import mindspore as ms
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from airnetpack.blocks import MLP
from airnetpack.activations import Swish

__all__ = ["AtomwiseReadout", "CFconv", "ScaleShift", "Standardize", "Aggregate"]

class AtomwiseError(Exception):
    pass

class AtomwiseReadout(nn.Cell):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the
    energy.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of target property (default: 1)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        mean (torch.Tensor or None): mean of property
        stddev (torch.Tensor or None): standard deviation of property (default: None)
        atomref (torch.Tensor or None): reference single-atom properties. Expects
            an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
            property of element Z. The value of atomref[0] must be zero, as this
            corresponds to the reference property for for "mask" atoms. (default: None)
        outnet (callable): Network used for atomistic outputs. Takes schnetpack input
            dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)

    Returns:
        tuple: prediction for property

        If contributions is not None additionally returns atom-wise contributions.

        If derivative is not None additionally returns derivative w.r.t. atom positions.

    """

    def __init__(
        self,
        n_in,
        n_out=1,
        aggregation_mode="sum",
        n_neurons=[32,],
        activation=Swish(),
        mean=None,
        stddev=None,
        outnet=None,
    ):
        super().__init__()
        
        self.do_standardize = True
        if mean is None and stddev is None:
            self.do_standardize = False
        else:
            # build standardization layer
            self.standardize = ScaleShift(mean, stddev)
            mean = Tensor([0.0],ms.float32) if mean is None else mean
            stddev = Tensor([1.0],ms.float32) if stddev is None else stddev

        # build output network
        if outnet is None:
            self.out_net = nn.SequentialCell([
                MLP(n_in, n_out, n_neurons, activation),
            ])
        else:
            self.out_net = outnet

        # build aggregation layer
        if aggregation_mode == "sum":
            self.atom_pool = Aggregate(axis=1, mean=False)
        elif aggregation_mode == "avg":
            self.atom_pool = Aggregate(axis=1, mean=True)
        else:
            raise AtomwiseError(
                "{} is not a valid aggregation " "mode!".format(aggregation_mode)
            )

    def construct(self, inputs, atom_mask=None):
        r"""
        predicts atomwise property
        """

        n_batch, n_atom, n_basis = inputs.shape
        
        x = F.reshape(inputs,(-1,n_basis))
        # run prediction
        yi = self.out_net(x)
        yi = F.reshape(yi,(n_batch,n_atom,-1))
        if self.do_standardize:
            yi = self.standardize(yi)

        y = self.atom_pool(yi, atom_mask)
        
        # ~ n_out = y.shape[-1]
        # ~ if n_out == 1:
            # ~ y = F.reshape(y,(n_batch))
        
        return y
        
class CFconv(nn.Cell):
    def __init__(
        self,
        num_rbf,
        dim_filter,
        ):
        super().__init__()
        # filter block used in interaction block
        self.dense1 = nn.Dense(num_rbf, dim_filter)
        self.dense1.activation = Swish()
        self.dense1.activation_flag = True
        
        self.dense2 = nn.Dense(dim_filter, dim_filter)
                
    def construct(self,x,r_ij,rbf):
        n_batch, n_atom, n_nbh, n_rbf = rbf.shape
        rbf = F.reshape(rbf,(-1,n_rbf))
        
        d = self.dense1(rbf)
        W = self.dense2(d)
        W = F.reshape(W,(n_batch,n_atom,n_nbh,-1))

        y = x * W
        
        return y

class ScaleShift(nn.Cell):
    r"""Scale and shift layer for standardization.

    .. math::
       y = x \times \sigma + \mu

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.

    """

    def __init__(self, mean, stddev):
        super().__init__()
        self.mean = mean
        self.stdev = stddev

    def construct(self, inputs):
        """Compute layer output.

        Args:
            inputs (mindspore.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        y = inputs * self.stddev + self.mean
        return y


class Standardize(nn.Cell):
    r"""Standardize layer for shifting and scaling.

    .. math::
       y = \frac{x - \mu}{\sigma}

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.
        eps (float, optional): small offset value to avoid zero division.

    """

    def __init__(self, mean, stddev, eps=1e-9):
        super().__init__()
        self.mean = mean
        self.stddev = stddev
        self.eps = F.ones_like(stddev) * eps

    def construct(self, inputs):
        """Compute layer output.

        Args:
            inputs (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        # Add small number to catch divide by zero
        y = (inputs - self.mean) / (self.stddev + self.eps)
        return y


class Aggregate(nn.Cell):
    """Pooling layer based on sum or average with optional masking.

    Args:
        axis (int): axis along which pooling is done.
        mean (bool, optional): if True, use average instead for sum pooling.
        keepdim (bool, optional): whether the output tensor has dim retained or not.

    """

    def __init__(self, axis, mean=False, keepdim=False):
        super().__init__()
        self.average = mean
        self.axis = axis
        self.keepdim = keepdim

    def construct(self, inputs, mask=None):
        r"""Compute layer output.

        Args:
            input (torch.Tensor): input data.
            mask (torch.Tensor, optional): mask to be applied; e.g. neighbors mask.

        Returns:
            torch.Tensor: layer output.

        """
        # mask input
        if mask is not None:
            inputs = inputs * F.expand_dims(mask,-1)
        # compute sum of input along axis
        reduce_sum=P.ReduceSum(keep_dims=self.keepdim)
        maximum=P.Maximum()
        y = reduce_sum(inputs, self.axis)
        # compute average of input along axis
        if self.average:
            # get the number of items along axis
            if mask is not None:
                N = reduce_sum(mask, self.axis)
                N = maximum(N, other=F.ones_like(N))
            else:
                N = inputs.shape[self.axis]
            y = y / N
        return y
