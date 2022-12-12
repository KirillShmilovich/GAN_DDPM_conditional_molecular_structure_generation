# -*- coding: utf-8 -*-

import mdshare
import mdtraj as md
import nglview as ng
import numpy as np

from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn 
from argparse import ArgumentParser

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, Dataset_traj, cycle, num_to_groups

import sys, os
sys.path.append('./denoising_diffusion_pytorch')

from Unets_MJ import Unet_xyz
from denoising_diffusion_pytorch_MJ import Unet, GaussianDiffusion
from denoising_diffusion_pytorch_1d_MJ import Unet1D


device = torch.device("cuda")

parser = ArgumentParser()

parser.add_argument("-f", "--folder", type=str, default='traj_AIB9')
parser.add_argument("-t", "--train", type=str, default='train_adp')
parser.add_argument("-c", "--cond", type=str, default='test_cond')
parser.add_argument("-v", "--test", type=str, default='test_adp')
parser.add_argument("-m", "--model", type=str, default='Unet')
parser.add_argument("-n", "--nsrvs", type=int, default=3)
parser.add_argument("-s", "--nsteps", type=int, default=1000)
parser.add_argument("-o", "--save", type=str, default='test')
parser.add_argument("-a", "--natoms", type=int, default=10)
parser.add_argument("--loss", type=str, default='l1')
parser.add_argument("--beta", type=str, default='linear')

parser.add_argument("--urandom", type=bool, default=False)
parser.add_argument("--usinu", type=bool, default=False)
parser.add_argument("--uself", type=bool, default=False)

parser.add_argument("--dtime", type=int, default=1000)
parser.add_argument("--init_dim", type=int, default=32)

args = parser.parse_args()

folder_name = args.folder
train_name = args.train 
cond_name = args.cond 
test_name = args.test
model_name = args.model
n_srvs = args.nsrvs
model_train_steps = args.nsteps
save_name = args.save
n_atoms = args.natoms
loss = args.loss
beta = args.beta

urandom = args.urandom
usinu = args.usinu
uself = args.uself

dtime = args.dtime
init_dim = args.init_dim

print(folder_name, train_name, model_name)

if model_name =='Unet':
    model = Unet(dim = init_dim, dim_mults = (1, 2, 4, 8), groups = 8).to(device) #(1, 2, 2, 4)
elif model_name == 'Unet_xyz':
    model =  Unet_xyz(dim = init_dim, n_conds=n_srvs, dim_mults = (1, 2, 2, 4), groups = 8).to(device)
elif model_name == 'Unet_cond':
    model = Unet_cond(dim = init_dim, n_conds=n_srvs, dim_mults = (1, 2, 2, 4), groups = 8).to(device)
elif model_name == 'Unet1D':
    model = model = Unet1D(dim = init_dim, dim_mults = (1, 2, 4, 8), channels=1,     
              self_condition=uself, random_fourier_features=urandom, 
              learned_sinusoidal_cond=usinu, learned_sinusoidal_dim=16) 
    
op_num = n_atoms*3 + n_srvs     
konw_op_num = n_srvs

model = nn.DataParallel(model)
model.to(device)

diffusion = GaussianDiffusion(
    model,                          # U-net model
    timesteps = dtime,               # number of diffusion steps
    unmask_number = konw_op_num,    # konw_op_num,  # the dimension of x2 in P(x1|x2)
    loss_type = loss,    #'l2'           # L1 or L2
    beta_schedule = beta
).to(device) 

#set training parameters
trainer = Trainer(
    diffusion,                                   # diffusion model
    folder = folder_name,                        # folder of trajectories
    system = train_name,         
    train_batch_size = 128,                      # training batch size
    train_lr = 1e-5,                             # learning rate
    train_num_steps = model_train_steps,       # total training steps
    gradient_accumulate_every = 1,               # gradient accumulation steps
    ema_decay = 0.995,                           # exponential moving average decay
    op_number = op_num,
    fp16 = False,
    save_name = save_name, # turn on mixed precision training with apex
)

# start training
trainer.train()

#prepare a dataloader to give samples from the conditional part of the distribution
batch_size = 5_005  #1280  #the number of samples generated in each batch

sample_ds = Dataset_traj(folder_name,  cond_name, n_conds=konw_op_num) 
sample_ds.max_data = trainer.ds.max_data
sample_ds.min_data = trainer.ds.min_data    #To ensure that the sample data is scaled in the same way as the training data

# both shuffle and pin_memory og set to true
sample_dl = cycle(data.DataLoader(sample_ds, batch_size = batch_size, shuffle=False, pin_memory=False)) 

# generate samples and save -- is this best way to do it?
num_sample = sample_ds.data.shape[0] # total number of samples

batches = num_to_groups(num_sample, batch_size)
all_ops_list = list(map(lambda n: trainer.ema_model.sample(
    trainer.op_number, batch_size=n, samples = next(sample_dl).cuda()[:n, :]), batches))

all_ops = torch.cat(all_ops_list, dim=0).cpu()
all_ops = trainer.rescale_sample_back(all_ops)

np.save(str(trainer.RESULTS_FOLDER / f'samples_final'), all_ops.numpy())