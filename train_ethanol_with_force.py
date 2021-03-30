import numpy as np
import time
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import dataset as ds
from mindspore.train import Model
from mindspore import context
from mindspore.train.callback import LossMonitor,SummaryCollector

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig,Callback
from mindspore.train.serialization import load_checkpoint,load_param_into_net
from mindspore.profiler import Profiler

from airnetpack.model import SchNet,AirNet,MolCalculator
from airnetpack.cutoff import MollifierCutoff
from airnetpack.acsf import LogGaussianDistribution
from airnetpack.activations import ShiftedSoftplus,Swish
from airnetpack.cutoff import CosineCutoff,SmoothCutoff
from airnetpack.train import SquareLoss,AbsLoss,ForceAbsLoss,MLoss
from airnetpack.train import WithForceLossCell,WithForceEvalCell
from airnetpack.train import Recorder,MAE,MSE

if __name__ == '__main__':

    # np.set_printoptions(threshold=np.inf)
    seed = 1111
    ms.set_seed(seed)

    summary_collector = SummaryCollector(summary_dir='./summary_dir')

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    # profiler = Profiler()

    mol_name='ethanol'
    train_file = './' + mol_name + '_train_1024.npz'
    valid_file = './' + mol_name + '_valid_128.npz'
    train_data = np.load(train_file)
    valid_data = np.load(valid_file)

    atomic_numbers = Tensor(train_data['z'],ms.int32)
    num_atom = atomic_numbers.size
    scale = float(train_data['scale'][0]) / num_atom
    shift = float(train_data['shift'][0])

    # mod = AirNet(min_rbf_dis=0.1,max_rbf_dis=8,num_rbf=32,n_interactions=3,dim_atomembedding=128,n_heads=8,max_cycles=1,use_time_embedding=True,fixed_cycles=True,public_filter=False,use_all_interactions=False,self_dis=0.01,cutoff_network=CosineCutoff)
    # mod = AirNet(min_rbf_dis=0.05,max_rbf_dis=10,num_rbf=32,rbf_sigma=0.2,n_interactions=3,dim_atomembedding=128,n_heads=8,max_cycles=1,use_time_embedding=True,fixed_cycles=True,public_filter=False,use_all_interactions=True)
    # mod = AirNet(min_rbf_dis=0.05,max_rbf_dis=10,num_rbf=32,rbf_sigma=0.2,n_interactions=3,dim_atomembedding=128,n_heads=8,max_cycles=5,use_time_embedding=True,fixed_cycles=True,public_filter=False,use_all_interactions=True,debug=True)
    mod = AirNet(min_rbf_dis=0.05,max_rbf_dis=10,num_rbf=32,rbf_sigma=0.2,n_interactions=3,dim_atomembedding=128,n_heads=8,max_cycles=1,use_time_embedding=True,fixed_cycles=True,public_filter=False,use_all_interactions=True,debug=False)
    # mod = AirNet(min_rbf_dis=0.05,max_rbf_dis=10,num_rbf=32,rbf_sigma=0.2,n_interactions=3,dim_atomembedding=128,n_heads=8,max_cycles=1,use_time_embedding=True,fixed_cycles=True,public_filter=False,use_all_interactions=False,debug=False)
    # mod = SchNet(max_rbf_dis=8,num_rbf=32,dim_atomembedding=128,dim_filter=128)
    net = MolCalculator(mod,atom_types=atomic_numbers,full_connect=True,scale=scale,shift=shift)

    network_name = mod.network_name

    tot_params = 0
    for i,param in enumerate(net.trainable_params()):
        tot_params += param.size
        print(i,param.name,param.shape)
        # print(i,param.asnumpy())
    print('Total parameters: ',tot_params)
    # print(net)

    print(scale,shift)

    n_epoch = 1024
    repeat_time = 1
    batch_size = 32

    R = Tensor(valid_data['R'][0:16],ms.float32)
    E = Tensor(valid_data['E'][0:16],ms.float32)
    out = net(R)
    for o,e in zip(out,E):
        print(o,e)

    ds_train = ds.NumpySlicesDataset({'R':train_data['R'],'F':train_data['F'],'E':train_data['E']},shuffle=True)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(repeat_time)

    ds_valid = ds.NumpySlicesDataset({'R':valid_data['R'],'F':valid_data['F'],'E':valid_data['E']},shuffle=False)
    ds_valid = ds_valid.batch(len(valid_data['E']))
    ds_valid = ds_valid.repeat(1)

    loss_opeartion = WithForceLossCell(net,SquareLoss(),SquareLoss())
    eval_opeartion = WithForceEvalCell(net)

    optim = nn.Adam(params=net.trainable_params(),learning_rate=1e-4)
    train_net=nn.TrainOneStepCell(loss_opeartion,optim)

    energy_mae = 'EnergyMAE'
    forces_mae = 'ForcesMAE'
    forces_mse = 'ForcesMSE'
    # model = Model(train_net,eval_network=eval_opeartion,metrics={energy_mae:MAE([2,3]),forces_mae:MAE([4,5]),forces_mse:MSE([4,5])},amp_level='O3')
    model = Model(train_net,eval_network=eval_opeartion,metrics={energy_mae:MAE([2,3]),forces_mae:MAE([4,5]),forces_mse:MSE([4,5])},amp_level='O0')

    params_name = mol_name + '_' + network_name
    config_ck = CheckpointConfig(save_checkpoint_steps=1024, keep_checkpoint_max=64)
    ckpoint_cb = ModelCheckpoint(prefix=params_name, directory=None, config=config_ck)

    record_file = mol_name + '_' + network_name + '-info.data'
    record_cb = Recorder(model, record_file, 1, eval_dataset=ds_valid)

    print("Start training ...")
    beg_time = time.time()
    model.train(n_epoch,ds_train,callbacks=[record_cb,ckpoint_cb,summary_collector],dataset_sink_mode=False)
    end_time = time.time()
    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)
    print ("Training Fininshed!")
    print ("Training Time: %02d:%02d:%02d" % (h, m, s))

    # profiler.analyse()