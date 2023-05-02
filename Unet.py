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
from torchvision import transforms, utils
from PIL import Image
import numpy as np
from tqdm import tqdm
from einops import rearrange
import torchgeometry as tgm
import glob
import os
import errno
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch import linalg as LA
from sklearn.mixture import GaussianMixture
import torch
import torchvision
import cv2

def is_present(x):
    return x is not None

def default(val, d):
    if is_present(val):
        return val
    return d() if isfunction(d) else d

class Unet(nn.Module):
    def __init__(self,dim,out_dim = None, hidden_dim=(1, 2, 4, 8), channels = 3, with_time_embedding_amsss = True, residual = False):
        super().__init__()
        self.channels = channels
        self.residual = residual

        dims = [channels, *map(lambda m: dim * m, hidden_dim)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_embedding_amsss:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPositionalEmbedding_amsss(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvolutionNextBlocks(dim_in, dim_out, time_embbeding_dim_amsss = time_dim, norm = ind != 0),
                ConvolutionNextBlocks(dim_out, dim_out, time_embbeding_dim_amsss = time_dim),
                ResidualNetwork_amsss(PreNorm(dim_out, LinearAttention_amsss(dim_out))),
                Downsample_amsss(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvolutionNextBlocks(mid_dim, mid_dim, time_embbeding_dim_amsss = time_dim)
        self.mid_attn = ResidualNetwork_amsss(PreNorm(mid_dim, LinearAttention_amsss(mid_dim)))
        self.mid_block2 = ConvolutionNextBlocks(mid_dim, mid_dim, time_embbeding_dim_amsss = time_dim)

        for ind, (dim_in, dim_out) in enumerate(in_out[1:][::-1]):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvolutionNextBlocks(dim_out * 2, dim_in, time_embbeding_dim_amsss = time_dim),
                ConvolutionNextBlocks(dim_in, dim_in, time_embbeding_dim_amsss = time_dim),
                ResidualNetwork_amsss(PreNorm(dim_in, LinearAttention_amsss(dim_in))),
                Upsample_amsss(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvolutionNextBlocks(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        orig_x = x
        t = self.time_mlp(time) if is_present(self.time_mlp) else None

        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
        if self.residual:
            return self.final_conv(x) + orig_x

        return self.final_conv(x)
    

class SinusoidalPositionalEmbedding_amsss(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb_amsss = math.log(10000) / (half_dim - 1)
        emb_amsss = torch.exp(torch.arange(half_dim, device=device) * -emb_amsss)
        emb_amsss = x[:, None] * emb_amsss[None, :]
        emb_amsss = torch.cat((emb_amsss.sin(), emb_amsss.cos()), dim=-1)
        return emb_amsss
    
    
class ConvolutionNextBlocks(nn.Module):
    def __init__(self, dim, dim_out, *, time_embbeding_dim_amsss = None, multiply = 2, norm = True):
        super().__init__()
        if is_present(time_embbeding_dim_amsss):
            self.mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_embbeding_dim_amsss, dim)
            )
        else:
            self.mlp=None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * multiply, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(dim_out * multiply, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)
        if is_present(self.mlp):
            assert is_present(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.res_conv(x)
    

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
class ResidualNetwork_amsss(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
class LinearAttention_amsss(nn.Module):
    def __init__(self, dim, heads_amsss = 4, dim_head = 32):
        super().__init__()
        self.scale_amsss = dim_head ** -0.5
        self.heads_amsss = heads_amsss
        hidden_dim = dim_head * heads_amsss
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads_amsss), qkv)
        q = q * self.scale_amsss
        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads_amsss, x = h, y = w)
        return self.to_out(out)
    
def Upsample_amsss(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample_amsss(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)