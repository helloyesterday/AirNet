from mindspore import nn
from mindspore.ops import operations as P

class ShiftedSoftplus(nn.Cell):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (mindspore.Tensor): input tensor.

    Returns:
        mindspore.Tensor: shifted soft-plus of input.

    """
    def __init__(self):
        super().__init__()
        self.softplus = P.Softplus()

    def construct(self,x):
        return self.softplus(x) - 0.6931471805599453

class Swish(nn.Cell):
    r"""Compute swish\SILU\SiL function.

    .. math::
       y_i = x_i / (1 + e^{-beta * x_i})

    Args:
        x (mindspore.Tensor): input tensor.

    Returns:
        mindspore.Tensor: shifted soft-plus of input.

    """
    def __init__(self):
        super().__init__()
        self.exp = P.Exp()

    def construct(self,x):
        return x/(1+self.exp(x))
