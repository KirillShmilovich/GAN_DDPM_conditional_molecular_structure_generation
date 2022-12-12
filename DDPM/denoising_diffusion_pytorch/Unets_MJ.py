import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam

import numpy as np
from tqdm import tqdm
from einops import rearrange

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# constants
SAVE_AND_SAMPLE_EVERY = 25000
UPDATE_EMA_EVERY = 10
PRINT_LOSS_EVERY = 200

#MODEL_INFO= '128-1-2-2-4-b128'
# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def generate_inprint_mask(n_batch, op_num, unmask_index = None):
    '''
    The mask will be True where we keep the true value and false where we want to infer the value
    So far it only supporting masking the right side of images

    '''
    mask = torch.zeros((n_batch, 1, op_num), dtype = bool)
    if not unmask_index == None:
        mask[:,:, unmask_index] = True
    return mask

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g

# building block modules
# groups are for batch norm only (not grouped convolutions)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )
    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.mlp(time_emb)[:, :, None]
        
        h = self.block2(h)
        return h + self.res_conv(x)
    
## MJ add back in simplified 2D versions of above block and resnet block

class Block_2D(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish())

    def forward(self, x, scale_shift = None):
        return self.block(x)

class ResnetBlock_2D(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        # this is a residual convolution

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x)
        h = self.block2(h)

        return h + self.res_conv(x)

    

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) l -> qkv b heads c l', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c l -> b (heads c) l', heads=self.heads)
        return self.to_out(out)

# model
# can import everything above frome denoising

class Unet_noattn(nn.Module):
    def __init__(self, dim, out_dim = None, dim_mults=(1, 2, 4, 8), groups = 8):
        super().__init__()
        
        # why start at 1? is eveyrthing  Nxdims?
        dims = [1, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        self.feature_dim = dim
        self.dim_mults = dim_mults
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        # add some resnet 2d here

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = dim, groups = groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim = dim, groups = groups),
                #Residual(Rezero(LinearAttention(dim_out))),  # why rezero?
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = dim, groups = groups)
        #self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = dim, groups = groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            # dim *2 bc concatenating
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = dim, groups = groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim = dim, groups = groups),
                #Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, 1)
        self.final_conv = nn.Sequential(
            Block(dim, dim, groups = groups),
            nn.Conv1d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        #import ipdb; ipdb.set_trace()
        
        # why do we need these lines?
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []
        size_list = []

        #for resnet, resnet2, attn, downsample in self.downs:
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)  # increase channel dim
            x = resnet2(x, t) # same dims
            #x = attn(x)       # same dims
            
            h.append(x)
            size_list.append(x.shape[-1])
            x = downsample(x) # decreases feature dim
            
        x = self.mid_block1(x, t)
        #x = self.mid_attn(x)
        x = self.mid_block2(x, t)
      
        # h contains info from downsampling
        #for resnet, resnet2, attn, upsample in self.ups:  
        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x[:,:,:size_list.pop()], h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            #x = attn(x)
            x = upsample(x)
            
        return self.final_conv(x[:,:,:size_list.pop()])
    
class simple_Conv(nn.Module):
    def __init__(self, dim, dim_in=1, dim_mults=(1, 2, 4, 8), groups=8):
        super().__init__()
        
        dim_out = dim
        self.feature_dim = dim
        self.dim_mults = dim_mults
        
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.res_down = ResnetBlock(dim_in, dim_out, time_emb_dim = dim, groups = groups)
        self.res_same = ResnetBlock(dim_out, dim_out, time_emb_dim = dim, groups = groups)
        #self.res_up = ResnetBlock(dim_out, dim_in, time_emb_dim=dim, groups = groups)
        
        self.final_conv = nn.Sequential(
            Block(dim_out, dim_out, groups = groups),
            nn.Conv1d(dim_out, dim_in, 1)
        )

    def forward(self, x, time):
        #import ipdb; ipdb.set_trace()
        
        # why do we need these lines?
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []
        size_list = []

        x = self.res_down(x, t)  # increase channel dim
        x = self.res_same(x, t)  # keep channel dim
        x = self.final_conv(x)   # decrease channel dim
        #x = self.res_up(x, t)    
            
        return x #self.final_conv(x[:,:,:size_list.pop()])
    
    
class simple_MLP(nn.Module):
  def __init__(self, dim, out_dim = None, n_layers=4):
      super().__init__()
      
      self.feature_dim = 32
      self.time_dim = dim
      self.n_layers = n_layers
        
      self.time_pos_emb = SinusoidalPosEmb(dim)
      self.dim_mults = [0, 1]
      
      self.mlp = nn.Sequential(
          nn.Linear(self.feature_dim + self.time_dim, 200),
          nn.SiLU(),
          nn.Linear(200, self.feature_dim)
      )
        
      self.mlp_time = nn.Sequential(
          nn.Linear(self.time_dim, self.time_dim*4),
          Mish(),
          nn.Linear(self.time_dim*4, self.time_dim)
        )
    
      # collect layers, embedding time in each
      self.nn_list = nn.ModuleList([])
      for n in range(self.n_layers):
          self.nn_list.append(nn.Sequential(
                              nn.Linear(self.feature_dim + self.time_dim, 200),
                              nn.SiLU(),
                              nn.Linear(200, self.feature_dim)
      ))
       

  def forward(self, x, time):
    #import ipdb; ipdb.set_trace()
    
    # pass t through its own mlp
    t = self.time_pos_emb(time)
    t = self.mlp_time(t)
    
    # remove channel dim from x
    x = x.squeeze()
    
    # iterate through nn_list
    for layer in self.nn_list:   #self.nn_list:
        x = layer(torch.cat([x, t], dim=-1))

    # restore channel time
    return x.unsqueeze(1)

            
            
class Unet_xyz(nn.Module):
    def __init__(self, dim, n_conds=0, out_dim = None, dim_mults=(1, 2, 4, 8), groups = 8):
        super().__init__()
            
        # first and last dims are > 1
        self.n_cond_dims = n_conds
        self.n_pnet_dims = 0
        self.n_channels_in = 3  + self.n_cond_dims + self.n_pnet_dims
        
        # should groups be 1 to start (all inputs convolved to all outputs)?
            
        # replace 1 with self.n_channels_in
        dims = [self.n_channels_in, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        self.feature_dim = dim
        self.dim_mults = dim_mults
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = dim, groups = groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim = dim, groups = groups),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = dim, groups = groups)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = dim, groups = groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = dim, groups = groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim = dim, groups = groups),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        # final out_dim should always be 3 if reducing back to xyz
        #out_dim = default(out_dim, 1)
        out_dim = 3
        
        self.final_conv = nn.Sequential(
            Block(dim, dim, groups = groups),
            nn.Conv1d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []
        size_list = []
        
        # reshape x in N x (3 + n_conds + n_pnet_dims)
        n_atoms = (x.shape[-1] - self.n_cond_dims) // 3 
     
        x_xyz = x[:, :, self.n_cond_dims:].reshape((-1, n_atoms, 3))
        x_xyz = torch.transpose(x_xyz, 1, 2)
        
        # expand conditions over channel dim
        x_cond = x[:, :, :self.n_cond_dims]
        x_cond_exp = torch.transpose(x_cond, 1, 2).expand(-1, -1, n_atoms)

        # concatenate the two over channel dims
        x = torch.cat([x_xyz, x_cond_exp], dim=1)
        #x = x_xyz
        #print(x.shape)
        
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            size_list.append(x.shape[-1])
            x = downsample(x)
            #print(x.shape)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
      
        for resnet, resnet2, attn, upsample in self.ups: 
            
            x = torch.cat((x[:,:,:size_list.pop()], h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)
            #print(x.shape)
            
        # outputs B X 3 X N (what we actually want in the end)
        x = self.final_conv(x[:,:,:size_list.pop()])
        #print(x.shape)
        
        # re-flatten for diffusion model format
        x = torch.transpose(x, 1, 2)
        x = x.reshape((-1, n_atoms*3))
            
        # need to shape back into N*3 + C x 1
        x = torch.cat([x_cond, x.reshape((-1, 1, 3*n_atoms))], dim=2)
            
        return x