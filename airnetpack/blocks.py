import mindspore
from mindspore import nn

# ~ from schnetpack import Properties
from airnetpack.activations import Swish

__all__ = ["MLP", "ElementalGate"]


class MLP(nn.Cell):
    """Multiple layer fully connected perceptron neural network.

    Args:
        n_in (int): number of input dimensions.
        n_out (int): number of output dimensions.
        layer_dims (list of int or int): number hidden layer dimensions.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers (int, optional): number of layers.
        activation (callable, optional): activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.

    """

    def __init__(
        self, n_in, n_out, layer_dims=None, activation=Swish()
        ):
        super().__init__()
        # get list of number of dimensions in input, hidden & output layers
        if layer_dims is None or len(layer_dims)==0:
            self.out_net=nn.Dense(n_in, n_out)
        else:
            # assign a Dense layer (with activation function) to each hidden layer
            nets=[]
            indim=n_in
            for ldim in layer_dims:
                layer = nn.Dense(indim, ldim)
                if activation is not None:
                    layer.activation = activation
                    layer.activation_flag = True
                nets.append(layer)
                indim=ldim
            # assign a Dense layer (without activation function) to the output layer
            nets.append(nn.Dense(indim, n_out, activation=None))
            # put all layers together to make the network
            self.out_net = nn.SequentialCell(nets)

    def construct(self, inputs):
        """Compute neural network output.

        Args:
            inputs (torch.Tensor): network input.

        Returns:
            torch.Tensor: network output.

        """
        return self.out_net(inputs)


class ElementalGate(nn.Cell):
    """
    Produces a Nbatch x Natoms x Nelem mask depending on the nuclear charges passed as an argument.
    If onehot is set, mask is one-hot mask, else a random embedding is used.
    If the trainable flag is set to true, the gate values can be adapted during training.

    Args:
        elements (set of int): Set of atomic number present in the data
        onehot (bool): Use one hit encoding for elemental gate. If set to False, random embedding is used instead.
        trainable (bool): If set to true, gate can be learned during training (default False)
    """

    def __init__(self, elements, onehot=True, trainable=False):
        super().__init__()
        self.trainable = trainable

        # Get the number of elements, as well as the highest nuclear charge to use in the embedding vector
        self.nelems = len(elements)
        maxelem = int(max(elements) + 1)

        self.gate = nn.Embedding(maxelem, self.nelems, onehot)

        # Set trainable flag
        if not trainable:
            self.gate.embedding_table.requires_grad = False

    def construct(self, atomic_numbers):
        """
        Args:0
            atomic_numbers (torch.Tensor): Tensor containing atomic numbers of each atom.

        Returns:
            torch.Tensor: One-hot vector which is one at the position of the element and zero otherwise.

        """
        return self.gate(atomic_numbers)
