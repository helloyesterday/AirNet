import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.api import ms_function

from airnetpack.cutoff import CosineCutoff

__all__ = [
    "GaussianSmearing",
    "RadialDistribution",
]

# radial_filter in RadialDistribution
class GaussianSmearing(nn.Cell):
    r"""Smear layer using a set of Gaussian functions.

    Args:
        start (float, optional): center of first Gaussian function, :math:`\mu_0`.
        stop (float, optional): center of last Gaussian function, :math:`\mu_{N_g}`
        n_gaussians (int, optional): total number of Gaussian functions, :math:`N_g`.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).
        trainable (bool, optional): If True, widths and offset of Gaussian functions
            are adjusted during training process.

    """

    def __init__(
        self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False
    ):
        super().__init__()
        # compute offset and width of Gaussian functions
        offset = Tensor(np.linspace(start, stop, n_gaussians),ms.float32)
        width = ( (stop-start) / (n_gaussians-1) ) * F.ones_like(offset)
        
        self.width = width
        self.offset = offset
        self.centered = centered
        
        if trainable:
            self.width = ms.Parameter(width,"widths")
            self.offset = ms.Parameter(offset,"offset")

    def construct(self, distances):
        """Compute smeared-gaussian distance values.

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.

        """
        nfuns = self.offset.size()
        ex_dis=F.expand_dims(distances,-1)
        if not self.centered:
            # compute width of Gaussian functions (using an overlap of 1 STDDEV)
            coeff = -0.5 / F.square(self.width)
            # Use advanced indexing to compute the individual components
            # ~ diff = distances[:, :, :, None] - offset[None, None, None, :]
            ex_offset=F.reshape(self.offset,(1,1,1,-1))
            diff = ex_dis - ex_offset
        else:
            # if Gaussian functions are centered, use offsets to compute widths
            coeff = -0.5 / F.square(self.offset)
            # if Gaussian functions are centered, no offset is subtracted
            diff = ex_dis
        # compute smear distance values
        exp = P.Exp()
        gauss = exp(coeff * F.square(diff))
        return gauss


class RadialDistribution(nn.Cell):
    """
    Radial distribution function used e.g. to compute Behler type radial symmetry functions.

    Args:
        radial_filter (callable): Function used to expand distances (e.g. Gaussians)
        cutoff_function (callable): Cutoff function
    """

    def __init__(self, radial_filter, cutoff_function=CosineCutoff):
        super().__init__()
        self.radial_filter = radial_filter
        self.cutoff_function = cutoff_function

    def construct(self, r_ij, elemental_weights=None, neighbor_mask=None):
        """
        Args:
            r_ij (torch.Tensor): Interatomic distances
            elemental_weights (torch.Tensor): Element-specific weights for distance functions
            neighbor_mask (torch.Tensor): Mask to identify positions of neighboring atoms

        Returns:
            torch.Tensor: Nbatch x Natoms x Nfilter tensor containing radial distribution functions.
        """

        nbatch, natoms, nneigh = r_ij.shape

        radial_distribution = self.radial_filter(r_ij)

        # If requested, apply cutoff function
        if self.cutoff_function is not None:
            cutoffs = self.cutoff_function(r_ij)
            radial_distribution = radial_distribution * F.expand_dims(cutoffs,-1)

        # Apply neighbor mask
        if neighbor_mask is not None:
            radial_distribution = radial_distribution * Tensor(F.expand_dims(
                neighbor_mask, -1
            ),ms.float32)

        # Weigh elements if requested
        if elemental_weights is not None:
            radial_distribution = (
                F.expand_dims(radial_distribution,-1)
                * F.expand_dims(elemental_weights,-2)
            )
            
        reduce_sum = P.ReduceSum()
        radial_distribution = reduce_sum(radial_distribution, 2)
        return F.reshape(radial_distribution,(nbatch, natoms, -1))
