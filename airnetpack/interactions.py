import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from airnetpack.base import Aggregate

class SchNetInteraction(nn.Cell):
    r"""Continuous-filter convolution block used in SchNet module.

    Args:
        n_in (int): number of input (i.e. atomic embedding) dimensions.
        dim_filter (int): number of filter dimensions.
        filter_network (nn.Module): filter block.
        cutoff_network (nn.Module, optional): if None, no cut off function is used.
        activation (callable, optional): if None, no activation function is used.
        normalize_filter (bool, optional): If True, normalize filter to the number
            of neighbors when aggregating.
        axis (int, optional): axis over which convolution should be applied.

    """

    def __init__(
        self,
        dim_atomembedding,
        dim_filter,
        cfconv_module,
        cutoff_network=None,
        activation=None,
        normalize_filter=False,
        axis=-2,
    ):
        super().__init__()
        self.atomwise_bc = nn.Dense(dim_atomembedding, dim_filter,
            has_bias=False, activation=None)
        self.atomwise_ac1 = nn.Dense(dim_filter, dim_atomembedding,
            has_bias=True, activation=None)
        if activation is not None:
            self.atomwise_ac1.activation = activation
            self.atomwise_ac1.activation_flag = True
        self.atomwise_ac2 = nn.Dense(dim_atomembedding, dim_atomembedding,
            has_bias=True, activation=None)

        self.cfconv = cfconv_module
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def construct(self, x, r_ij, rbf, neighbors, pairwise_mask):
        """Compute convolution block.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            pairwise_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_out) shape.

        """

        # the 1st atom-wise module at interaction layer
        n_batch,n_atom,n_neigh = r_ij.shape
        
        if len(neighbors.shape) == 2:
            neigh_batch = 0
            neigh_atoms,neigh_nbh = neighbors.shape
        elif len(neighbors.shape) == 3:
            neigh_batch,neigh_atoms,neigh_nbh = neighbors.shape
        else:
            print('SchNetInteraction Error! The dimension of the Tensor "neighbors must be 3 or 2"')
            exit()
        
        if neigh_atoms != n_atom:
            print('SchNetInteraction Error! The number of atoms at "position" and "neighbors" mismatch ')
            exit()
        
        x_batch,x_atom,x_in = x.shape
        x = F.reshape(x,(-1,x_in))
        ax = self.atomwise_bc(x)
        ax = F.reshape(ax,(x_batch,x_atom,-1))

        # reshape y for element-wise multiplication by W
        if neigh_batch == 0:
            xx=F.gather(ax,neighbors,-2)
        elif neigh_batch == 1:
            xx=F.gather(ax,neighbors[0],-2)
        elif neigh_batch == x_batch:
            xlist=[]
            for i in range(x_batch):
                xi=F.gather(ax[i],neighbors[i],-2)
                xlist.append(F.expand_dims(xi,0))
            concat = P.Concat()
            xx = concat(tuple(xlist))
        else:
            print("SchNetInteraction Error! the size of 'neighbors' is not equal to the batch number")
            exit()

        # CFconv: pass expanded interactomic distances through filter block
        y = self.cfconv(xx,r_ij,rbf)
        # element-wise multiplication, aggregating and Dense layer
        y = self.agg(y, pairwise_mask)

        # the 2nd atom-wise module at interaction layer
        n_basis=y.shape[-1]
        y = F.reshape(y,(-1,n_basis))
        y = self.atomwise_ac1(y)
        # the 3rd atom-wise module at interaction layer
        y = self.atomwise_ac2(y)
        y = F.reshape(y,(n_batch,n_atom,-1))
        return y
