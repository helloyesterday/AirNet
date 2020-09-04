import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import context

from airnetpack.model import SchNet,MolCalculator
from airnetpack.cutoff import MollifierCutoff

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

atomic_numbers = Tensor([11,0,0,0,0],ms.int32)
atomic_numbers = F.expand_dims(atomic_numbers,0)

positions = Tensor([[-0.0126981359, 1.0858041578, 0.0080009958],
                    [0.002150416, -0.0060313176, 0.0019761204],
                    [1.0117308433, 1.4637511618, 0.0002765748],
                    [-0.540815069, 1.4475266138, -0.8766437152],
                    [-0.5238136345, 1.4379326443, 0.9063972942]],
                    ms.float32)
positions = F.expand_dims(positions,0)
positions1 = positions + Tensor([1.,2.,3.],ms.float32)
concat = P.Concat()
positions = concat(tuple([positions,positions1]))
print(positions.shape)
print(positions)

model=SchNet()
mol = MolCalculator(model,atom_indices=atomic_numbers,full_connect=True,cutoff_network=MollifierCutoff())
out = mol(positions)
print(out)
