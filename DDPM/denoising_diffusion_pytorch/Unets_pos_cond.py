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
    def __init__(self, dim, dim_out, *, time_emb_dim, cond_dim, groups = 8):
        super().__init__()
        
        # this is needed to map time (32 dim) to dim_out (>= 32)
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        )
        
        # map conditioning dim to dim_out
        self.mlp_cond = nn.Sequential(
            Mish(),
            nn.Linear(cond_dim, dim_out)
        )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb, cond):
        h = self.block1(x)
        
        # add time embedding
        h += self.mlp(time_emb)[:, :, None]
        
        # add cond embeding
        h += self.mlp_cond(cond)[:, :, None]
        
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
    
    
class Unet_cond(nn.Module):
    def __init__(self, dim, n_conds, out_dim = None, dim_mults=(1, 2, 4, 8), groups = 8):
        
        # first and last dims are > 1
        self.n_cond_dims = n_conds
        
        super().__init__()
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
        
        # larger mlp (in addition to resnet block)
        self.mlp_cond = nn.Sequential(
            nn.Linear(n_conds, n_conds * 4),
            Mish(),
            nn.Linear(n_conds * 4, n_conds)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = dim, cond_dim=self.n_cond_dims, groups = groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim = dim, cond_dim=self.n_cond_dims, groups = groups),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = dim, cond_dim=self.n_cond_dims, groups = groups)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = dim, cond_dim=self.n_cond_dims, groups = groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = dim, cond_dim=self.n_cond_dims, groups = groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim = dim, cond_dim=self.n_cond_dims, groups = groups),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, 1)
        self.final_conv = nn.Sequential(
            Block(dim, dim, groups = groups),
            nn.Conv1d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        t = self.time_pos_emb(time)
        t = self.mlp(t)
        
        # extract condition from input
        x = x[:, :, self.n_cond_dims:]
        cond_3d = x[:, :, :self.n_cond_dims]
        cond = cond_3d.squeeze()     
        
        # can add pos ecoding here as well
        c = self.mlp_cond(cond)

        h = []
        size_list = []
       
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t, c)
            x = resnet2(x, t, c)
            x = attn(x)
            h.append(x)
            size_list.append(x.shape[-1])
            x = downsample(x)
            #print('x', x.shape)
           
        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)
      
        for resnet, resnet2, attn, upsample in self.ups:        
            x = torch.cat((x[:,:,:size_list.pop()], h.pop()), dim=1)
            x = resnet(x, t, c)
            x = resnet2(x, t, c)
            x = attn(x)
            x = upsample(x)
            #print(x.shape)
            
        # final conv
        x = self.final_conv(x[:,:,:size_list.pop()])
        
        # recombine with original condition
        x = torch.cat([cond_3d, x], dim=2)
            
        return x

