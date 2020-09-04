import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F

__all__ = ["CosineCutoff", "MollifierCutoff", "HardCutoff", "get_cutoff_by_string"]


def get_cutoff_by_string(key):
    # build cutoff module
    if key == "hard":
        cutoff_network = HardCutoff
    elif key == "cosine":
        cutoff_network = CosineCutoff
    elif key == "mollifier":
        cutoff_network = MollifierCutoff
    else:
        raise NotImplementedError("cutoff_function {} is unknown".format(key))
    return cutoff_network


class CosineCutoff(nn.Cell):
    r"""Class of Behler cosine cutoff.

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): cutoff radius.

    """

    def __init__(self, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff

    def construct(self, distances):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor): values of interatomic distances.

        Returns:
            mindspore.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cos = P.Cos()
        cutoffs = 0.5 * (cos(distances * np.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).ms.float32()
        return cutoffs


class MollifierCutoff(nn.Cell):
    r"""Class for mollifier cutoff scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): Cutoff radius.
        eps (float, optional): offset added to distances for numerical stability.

    """

    def __init__(self, cutoff=5.0, eps=1.0e-7):
        super().__init__()
        self.cutoff = cutoff
        self.eps = eps

    def construct(self, distances):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor): values of interatomic distances.

        Returns:
            mindspore.Tensor: values of cutoff function.

        """
        mask = ((distances + self.eps) < self.cutoff)
        exponent = 1.0 - 1.0 / (1.0 - F.square(distances * mask / self.cutoff))
        exp = P.Exp()
        cutoffs = exp(exponent)
        cutoffs = cutoffs * mask
        return cutoffs


class HardCutoff(nn.Cell):
    r"""Class of hard cutoff.

    .. math::
       f(r) = \begin{cases}
        1 & r \leqslant r_\text{cutoff} \\
        0 & r > r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float): cutoff radius.

    """

    def __init__(self, cutoff=5.0):
        super().__init__()

    def construct(self, distances):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor): values of interatomic distances.

        Returns:
            mindspore.Tensor: values of cutoff function.

        """
        mask = (distances <= self.cutoff)
        return mask
