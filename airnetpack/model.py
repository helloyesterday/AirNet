import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal

from airnetpack.interactions import SchNetInteraction
from airnetpack.base import AtomwiseReadout,CFconv,Aggregate
from airnetpack.cutoff import MollifierCutoff
from airnetpack.acsf import GaussianSmearing
from airnetpack.activations import Swish
from airnetpack.neighbors import AtomDistances

class GNN_Model(nn.Cell):
    r"""SchNet interaction block for modeling interactions of atomistic systems.

    Args:
        dim_atomembedding (int): number of features to describe atomic environments.
        num_rbf (int): number of input features of filter-generating networks.
        dim_filter (int): number of filters used in continuous-filter convolution.
        cutoff (float): cutoff radius.
        cutoff_network (nn.Module, optional): cutoff layer.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.

    """

    def __init__(
        self,
        num_atomtypes,
        num_rbf,
        dim_atomembedding,
        atom_indices=None,
        distance_expansion=None,
        cutoff_network=None,
    ):
        super().__init__()
        self.num_atomtypes=num_atomtypes
        self.dim_atomembedding=dim_atomembedding
        self.num_rbf=num_rbf
        self.distance_expansion = distance_expansion
        
        self.network_name='GNN_Model'

        # make a lookup table to store embeddings for each element (up to atomic
        # number max_z) each of which is a vector of size dim_atomembedding
        self.embedding = nn.Embedding(num_atomtypes, dim_atomembedding, True)
        
        self.fixed_atoms=False
        self.atom_indices=atom_indices
        self.atom_embedding=atom_indices
        if atom_indices is not None:
            set_atom_indices(atom_indices)

        self.cutoff_network=cutoff_network
        
        self.interactions = nn.CellList([])
        
        self.readout = None
        
    def set_atom_indices(self,atom_indices):
        self.fixed_atoms=True
        if type(atom_indices) is not Tensor:
            atom_indices = Tensor(atom_indices,ms.int32)
        self.atom_indices = F.expand_dims(atom_indices,0)
        self.atom_embedding=self.embedding(atom_indices)

    def construct(self, r_ij, neighbors=None, neighbor_mask=None, atom_indices=None):
        """Compute interaction output.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, dim_atomembedding) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            neighbor_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, dim_atomembedding) shape.

        """
        
        n_batch,n_atom,n_neigh = r_ij.shape
        
        # get atom embeddings for the input atomic numbers
        if atom_indices is None:
            if self.fixed_atoms:
                x_em = self.atom_embedding
            else:
                print('SchNet Error! The input term "atom_indices" must be given when the "fixed_atoms" is "False"')
                exit()
        else:
            x_em = self.embedding(atom_indices)
        
        em_batch,em_atom,em_in = x_em.shape
        if em_atom != n_atom:
            print('SchNet Error! the atoms\' number in r_ij and atom_indices are mismatch!')
        
        if em_batch != n_batch and em_batch == 1:
            broadcast_to = P.BroadcastTo((n_batch,em_atom,em_in))
            x_em = broadcast_to(x_em)
        
        # expand interatomic distances (for example, Gaussian smearing)
        if self.distance_expansion is None:
            rbf = F.expand_dims(r_ij,-1)
        else:
            rbf = self.distance_expansion(r_ij)
        
        # apply cutoff
        if self.cutoff_network is not None:
            cut = self.cutoff_network(r_ij)
            rbf = rbf * F.expand_dims(cut,-1)
        
        # continuous-filter convolution interaction block followed by Dense layer
        x = x_em
        for interaction in self.interactions:
            v = interaction(x, r_ij, rbf, neighbors, neighbor_mask)
            x = x + v
        
        if self.readout is not None:
            y = self.readout(x,neighbor_mask)
        else:
            y = x

        return y
        
class SchNet(GNN_Model):
    def __init__(
        self,
        num_atomtypes=100,
        num_rbf=25,
        dim_atomembedding=128,
        dim_filter=128,
        n_interactions=3,
        cutoff=5,
        atom_indices=None,
        distance_expansion=None,
        cutoff_network=MollifierCutoff(),
        normalize_filter=False,
        coupled_interactions=False,
        trainable_gaussians=False,
    ):
        super().__init__(
            num_atomtypes=num_atomtypes,
            num_rbf=num_rbf,
            dim_atomembedding=dim_atomembedding,
            atom_indices=atom_indices,
            distance_expansion=distance_expansion,
            cutoff_network=cutoff_network
            )
        self.network_name = 'SchNet'
        
        # layer for expanding interatomic distances in a basis
        if distance_expansion is None:
            self.distance_expansion = GaussianSmearing(
                0.0, cutoff, num_rbf, trainable=trainable_gaussians
            )

        cfconv_module = CFconv(num_rbf,dim_filter)
        # block for computing interaction
        if coupled_interactions:
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.CellList(
                [
                    SchNetInteraction(
                        dim_atomembedding=dim_atomembedding,
                        dim_filter=dim_filter,
                        cfconv_module=cfconv_module,
                        cutoff_network=cutoff_network,
                        activation=Swish(),
                        normalize_filter=normalize_filter,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.CellList(
                [
                    SchNetInteraction(
                        dim_atomembedding=dim_atomembedding,
                        dim_filter=dim_filter,
                        cfconv_module=cfconv_module,
                        cutoff_network=cutoff_network,
                        activation=Swish(),
                        normalize_filter=normalize_filter,
                    )
                    for _ in range(n_interactions)
                ]
            )

        # readout layer
        self.readout = AtomwiseReadout(dim_atomembedding,1,"sum",[32,],Swish())

class MolCalculator(nn.Cell):
    """SchNet architecture for learning representations of atomistic systems.

    Args:
        dim_atomembedding (int, optional): dimension of the vectors to describe atomic environments.
            This determines the size of each embedding vector; i.e. embeddings_dim.
        dim_filter (int, optional): number of filters used in continuous-filter convolution
        n_interactions (int, optional): number of interaction blocks.
        cutoff (float, optional): cutoff radius.
        n_gaussians (int, optional): number of Gaussian functions used to expand
            atomic distances.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        coupled_interactions (bool, optional): if True, share the weights across
            interaction blocks and filter-generating networks.
        return_intermediate (bool, optional): if True, `forward` method also returns
            intermediate atomic representations after each interaction block is applied.
        max_z (int, optional): maximum nuclear charge allowed in database. This
            determines the size of the dictionary of embedding; i.e. num_embeddings.
        cutoff_network (nn.Module, optional): cutoff layer.
        trainable_gaussians (bool, optional): If True, widths and offset of Gaussian
            functions are adjusted during training process.
        distance_expansion (nn.Module, optional): layer for expanding interatomic
            distances in a basis.
        charged_systems (bool, optional):

    References:
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        model,
        atom_indices=None,
        full_connect=False,
        distance_expansion=None,
        cutoff_network=None,
    ):
        super().__init__()
        
        self.predict=model
        dim_atomembedding=model.dim_atomembedding
        self.full_connect=full_connect
        
        if atom_indices is None:
            self.flexiable_atoms=True
            self.num_atoms=0
        else:
            self.flexiable_atoms=False
            if len(atom_indices.shape) == 1:
                self.num_atoms=len(atom_indices)
            elif len(atom_indices.shape) == 2:
                self.num_atoms=len(atom_indices[0])

            if type(atom_indices) is not Tensor:
                atom_indices = Tensor(atom_indices,ms.int32)

            self.atom_indices = atom_indices
            model.set_atom_indices(self.atom_indices)

        self.neighbors=None
        if self.full_connect:
            if self.flexiable_atoms is True:
                print("MolCalculator Error! The 'full_connect' flag cannot be True "+
                    "when the 'fixed_atoms' flag is 'True'")
                exit()
            if self.num_atoms <= 0:
                print("MolCalculator Error! The 'num_atoms' cannot be 0 "+
                    "when the 'full_connect' flag is 'True'")
                exit()
            list_neighbors = []
            all_ids = list(range(self.num_atoms))
            for i in range(self.num_atoms):
                neigh = all_ids.copy()
                neigh.pop(i)
                list_neighbors.append(neigh)
            self.neighbors=Tensor(list_neighbors,ms.int32)

        # layer for computing interatomic distances
        self.distances = AtomDistances()

    def construct(self,
        positions,
        neighbors=None,
        neighbor_mask=None,
        atom_indices=None):
        """Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.

        """

        # compute interatomic distance of every atom to its neighbors
        if neighbors is None:
            if self.full_connect:
                neighbors=self.neighbors
            else:
                print('MolCalculator Error! Then input term "neighbors" must be given when the "full_connect" flag is "False"')
                exit()

        r_ij = self.distances(positions,neighbors)
        y = self.predict(r_ij,neighbors,neighbor_mask,atom_indices)
        
        return y
