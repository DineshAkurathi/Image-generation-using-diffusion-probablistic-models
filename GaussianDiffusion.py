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

class GaussianDiffusion(nn.Module):
    def __init__(self,denoise_fn,*,image_size,channels = 3,timesteps = 1000,loss_type = 'l1',train_routine = 'Final',sampling_routine='default',discrete=False):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        betas = cosine_beta_schedule_amsss(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        self.train_routine = train_routine
        self.sampling_routine = sampling_routine

    @torch.no_grad()
    def sample(self, batch_size = 16, img=None, t=None):

        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps
        dimmer=None

        xt = img
        direct_recons = None

        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x11bar_amsss = self.denoise_fn(img, step)
            x2_bar = self.getx2barfrom_xt_amsss(x11bar_amsss, img, step)
            dimmer=None
            if direct_recons is None:
                direct_recons = x11bar_amsss
            xt_bar = x11bar_amsss
            if t != 0:
                xt_bar = self.q_sample_amsss(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x11bar_amsss
            if t - 1 != 0:
                step2_amsss = torch.full((batch_size,), t - 2, dtype=torch.long).cuda()
                xt_sub1_bar = self.q_sample_amsss(x_start=xt_sub1_bar, x_end=x2_bar, t=step2_amsss)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        self.denoise_fn.train()

        return xt, direct_recons, img

    def getx2barfrom_xt_amsss(self, x1_bar, xt, t):
        return (
                (xt - extract_amsss(self.sqrt_alphas_cumprod, t, x1_bar.shape) * x1_bar) /
                extract_amsss(self.sqrt_one_minus_alphas_cumprod, t, x1_bar.shape)
        )

    @torch.no_grad()
    def gen_sample(self, batch_size=16, img=None, t=None):
        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps

        dimmer = None

        noise = img
        direct_recons = None

        if self.sampling_routine == 'ddim':
            while (t):
                dimmer = None
                step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
                x11bar_amsss = self.denoise_fn(img, step)
                x2_bar = self.getx2barfrom_xt_amsss(x11bar_amsss, img, step)
                if direct_recons == None:
                    direct_recons = x11bar_amsss

                xt_bar = x11bar_amsss
                if t != 0:
                    xt_bar = self.q_sample_amsss(x_start=xt_bar, x_end=x2_bar, t=step)

                xt_sub1_bar = x11bar_amsss
                if t - 1 != 0:
                    step2_amsss = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                    xt_sub1_bar = self.q_sample_amsss(x_start=xt_sub1_bar, x_end=x2_bar, t=step2_amsss)
                dimmer = None

                x = img - xt_bar + xt_sub1_bar
                img = x
                t = t - 1

        elif self.sampling_routine == 'x0_step_down':
            while (t):
                step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
                x11bar_amsss = self.denoise_fn(img, step)
                x2_bar = noise
                if direct_recons == None:
                    direct_recons = x11bar_amsss

                xt_bar = x11bar_amsss
                if t != 0:
                    xt_bar = self.q_sample_amsss(x_start=xt_bar, x_end=x2_bar, t=step)

                xt_sub1_bar = x11bar_amsss
                if t - 1 != 0:
                    step2_amsss = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                    xt_sub1_bar = self.q_sample_amsss(x_start=xt_sub1_bar, x_end=x2_bar, t=step2_amsss)
                dimmer = None

                x = img - xt_bar + xt_sub1_bar
                img = x
                t = t - 1

        return noise, direct_recons, img


    @torch.no_grad()
    def forwardbackward_amsss(self, batch_size=16, img=None, t=None, times=None, eval=True):
        self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        Forward_amsss = []
        Forward_amsss.append(img)

        noise = torch.randn_like(img)

        for i in range(t):
            with torch.no_grad():
                step = torch.full((batch_size,), i, dtype=torch.long, device=img.device)
                n_img = self.q_sample_amsss(x_start=img, x_end=noise, t=step)
                Forward_amsss.append(n_img)

        Backward_amsss = []
        img = n_img
        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
            x11bar_amsss = self.denoise_fn(img, step)
            x2_bar = noise #self.getx2barfrom_xt_amsss(x11bar_amsss, img, step)

            Backward_amsss.append(img)

            xt_bar = x11bar_amsss
            if t != 0:
                xt_bar = self.q_sample_amsss(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x11bar_amsss
            if t - 1 != 0:
                step2_amsss = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                xt_sub1_bar = self.q_sample_amsss(x_start=xt_sub1_bar, x_end=x2_bar, t=step2_amsss)

            x = img - xt_bar + xt_sub1_bar
            imgage = x
            t = t - 1

        return Forward_amsss, Backward_amsss, imgage


    @torch.no_grad()
    def all_sample(self, batch_size=16, img=None, t=None, times=None, eval=True):
        if eval:
            self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        X1_0_amsss, X2_0_amsss, X_t_amsss = [], [], []
        while (t):

            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x11bar_amsss = self.denoise_fn(img, step)
            x2_bar = self.getx2barfrom_xt_amsss(x11bar_amsss, img, step)


            X1_0_amsss.append(x11bar_amsss.detach().cpu())
            X2_0_amsss.append(x2_bar.detach().cpu())
            X_t_amsss.append(img.detach().cpu())

            xt_bar = x11bar_amsss
            if t != 0:
                xt_bar = self.q_sample_amsss(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x11bar_amsss
            if t - 1 != 0:
                step2_amsss = torch.full((batch_size,), t - 2, dtype=torch.long).cuda()
                xt_sub1_bar = self.q_sample_amsss(x_start=xt_sub1_bar, x_end=x2_bar, t=step2_amsss)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        return X1_0_amsss, X2_0_amsss, X_t_amsss

    def q_sample_amsss(self, x_start, x_end, t):
        return (
                extract_amsss(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_amsss(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_end
        )

    def p_losses_amsss(self, x_start, x_end, t):
        b, c, h, w = x_start.shape
        if self.train_routine == 'Final':
            x_mix = self.q_sample_amsss(x_start=x_start, x_end=x_end, t=t)
            x_recon = self.denoise_fn(x_mix, t)
            if self.loss_type == 'l1':
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

        return loss

    def forward(self, x1, x2, *args, **kwargs):
        b, c, h, w, device, img_size, = *x1.shape, x1.device, self.image_size
        a=b*c/c
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        dimmer=None
        return self.p_losses_amsss(x1, x2, t, *args, **kwargs)
    

def cosine_beta_schedule_amsss(timesteps, s = 0.008):
    steps = timesteps + 1
    dimmer=None
    x = torch.linspace(0, steps, steps)
    alphacum_prod_amsss = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphacum_prod_amsss = alphacum_prod_amsss / alphacum_prod_amsss[0]
    betas = 1 - (alphacum_prod_amsss[1:] / alphacum_prod_amsss[:-1])
    return torch.clip(betas, 0, 0.999)

def extract_amsss(a_amsss, t,xshape_amsss):
    b, *_ = t.shape
    dimmer = None
    out = a_amsss.gather(-1, t)
    return out.reshape(b, *((1,) * (len(xshape_amsss) - 1)))