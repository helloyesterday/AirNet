import numpy as np
import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F

class AtomDistances(nn.Cell):
    r"""Layer for computing distance of every atom to its neighbors.

    Args:
        neighbors_fixed (bool, optional): if True, the `forward` method also returns
            normalized direction vectors.

    """

    def __init__(self):
        super().__init__()

    def construct(
        self, positions, neighbors, cell=None, cell_offsets=None, neighbor_mask=None
        ):
        r"""Compute distance of every atom to its neighbors.

        Args:
            positions (torch.Tensor): atomic Cartesian coordinates with
                (N_b x N_at x 3) shape.
            neighbors (torch.Tensor): indices of neighboring atoms to consider
                with (N_b x N_at x N_nbh) shape. If neighbors_fixed is True, the
                N_b is 1.
            cell (torch.tensor, optional): periodic cell of (N_b x 3 x 3) shape.
            cell_offsets (torch.Tensor, optional): offset of atom in cell coordinates
                with (N_b x N_at x N_nbh x 3) shape.
            neighbor_mask (torch.Tensor, optional): boolean mask for neighbor
                positions. Required for the stable computation of forces in
                molecules with different sizes.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh) shape.

        """

        # Construct auxiliary index vector
        # ~ n_dim = len(positions.shape)
        # ~ n_neigh_dim = len(neighbors.shape)
        n_batch,n_atom,n_xyz = positions.shape
        
        if len(neighbors.shape) == 2:
            neigh_batch = 0
            neigh_atoms,neigh_nbh = neighbors.shape
        elif len(neighbors.shape) == 3:
            neigh_batch,neigh_atoms,neigh_nbh = neighbors.shape
        else:
            print('Error! The dimension of the Tensor "neighbors must be 3 or 2"')
            exit()
        if neigh_atoms != n_atom:
            print('Error! The number of atoms at "position" and "neighbors" mismatch ')
            exit()
        
        # Get atomic positions of all neighboring indices
        
        if neigh_batch == 0:
            pos_xyz= F.gather(positions,neighbors,-2)
        elif neigh_batch == 1:
            pos_xyz= F.gather(positions,neighbors[0],-2)
        elif neigh_batch == n_batch:
            neigh_atoms = []
            for i in range(n_batch):
                xyz = F.gather(positions[i],neighbors[i],-2)
                neigh_atoms.append(F.expand_dims(xyz,0))
            concat = P.Concat()
            pos_xyz= concat(tuple(neigh_atoms))
        else:
            print("Error! the size of 'neighbors' is not equal to the batch number")
            exit()

        # Subtract positions of central atoms to get distance vectors
        dist_vec = pos_xyz - F.expand_dims(positions,-2)
        
        # add cell offset
        # ~ if cell is not None:
            # ~ B, A, N, D = cell_offsets.shape
            # ~ cell_offsets = F.reshape(cell_offsets,(B, A * N, D))
            # ~ bmm=P.BatchMatMul()
            # ~ offsets = bmm(cell_offsets,cell)
            # ~ offsets = F.reshape(offsets,(B, A, N, D))
            # ~ dist_vec += offsets

        # Compute vector lengths
        norm=nn.Norm(-1)
        distances = norm(dist_vec)
        
        # ~ zeroslike = P.ZerosLike()
        # ~ if neighbor_mask is not None:
            # ~ # Avoid problems with zero distances in forces (instability of square
            # ~ # root derivative at 0) This way is neccessary, as gradients do not
            # ~ # work with inplace operations, such as e.g.
            # ~ # -> distances[mask==0] = 0.0
            # ~ zero_distances = zeroslike(distances)
            # ~ distance=F.select(neighbor_mask,distance,zero_distances)

        # ~ if return_vecs:
            # ~ tmp_distances = zeroslike(distances)
            # ~ tmp_distances=F.select(neighbor_mask,distance,tmp_distances)

            # ~ if normalize_vecs:
                # ~ dist_vec = dist_vec / F.expand_dims(tmp_distances,-1)
            # ~ return distances, dist_vec
            
        return distances

class NeighborElements(nn.Cell):
    """
    Layer to obtain the atomic numbers associated with the neighboring atoms.
    """

    def __init__(self):
        super().__init__()

    def construct(self, atomic_numbers, neighbors):
        """
        Args:
            atomic_numbers (torch.Tensor): Atomic numbers (Nbatch x Nat x 1)
            neighbors (torch.Tensor): Neighbor indices (Nbatch x Nat x Nneigh)

        Returns:
            torch.Tensor: Atomic numbers of neighbors (Nbatch x Nat x Nneigh)
        """
            # Get molecules in batch
        n_batch = atomic_numbers.shape[0]
        # Construct auxiliary index
        
        # Get neighbors via advanced indexing
        numbers = []
        for i in range(n_batch):
            num = F.gather(atomic_numbers[i],neighbors[i],-2)
            numbers.append(F.expand_dims(num,0))
        concat = P.Concat()
        neighbor_numbers = concat(tuple(numbers))
        return neighbor_numberss
