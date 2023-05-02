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

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

def backwards_loss(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def updatedmodelaverages_amsss(self, ma_model, current_model):
        for current_params_amsss, motty_params_amsss in zip(current_model.parameters(), ma_model.parameters()):
            old_weight_amsss, up_weight_amsss = motty_params_amsss.data, current_params_amsss.data
            motty_params_amsss.data = self.updateaverages_amsss(old_weight_amsss, up_weight_amsss)

    def updateaverages_amsss(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def noiselikes_amsss(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    dimmer = None
    return repeat_noise() if repeat else noise()


class DatasetAug1_amsss(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for i in exts for p in Path(f'{folder}').glob(f'**/*.{i}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

def createfolder_amsss(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

from collections import OrderedDict
def remove_data_parallel(old_state_dict_amsss):
    new_statedict_amsss = OrderedDict()

    for k, v in old_state_dict_amsss.items():
        name = k.replace('.module', '')
        new_statedict_amsss[name] = v

    return new_statedict_amsss

def adjust_data_parallel(old_state_dict_amsss):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict_amsss.items():
        name = k.replace('denoise_fn.module', 'module.denoise_fn')
        new_state_dict[name] = v

    return new_state_dict

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        load_path = None,
        dataset = None,
        shuffle=True
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every


        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        if dataset == 'train':
            print(dataset, "DA used")
            self.ds = DatasetAug1_amsss(folder, image_size)
        else:
            print(dataset)
            self.ds = Dataset(folder, image_size)

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=16, drop_last=True))

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.fp16 = fp16

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema_amsss(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.updatedmodelaverages_amsss(self.ema_model, self.model)

    def save(self, itrs=None):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / f'model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model_{itrs}.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


    def add_title(self, path, title):
        image01 = cv2.imread(path)
        black = [0, 0, 0]
        constant = cv2.copyMakeBorder(image01, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        height = 20
        dimmer=None

        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)
        font = cv2.FONT_HERSHEY_SIMPLEX
        vcat_amsss = cv2.vconcat((violet, constant))

        cv2.putText(vcat_amsss, str(title), (violet.shape[1] // 2, height-2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat_amsss)



    def train(self):
        backwards = partial(backwards_loss, self.fp16)

        accuracy_loss = 0
        while self.step < self.train_num_steps:
            ulossy_amsss = 0
            for i in range(self.gradient_accumulate_every):
                data1_amsss = next(self.dl)
                data2_amsss = torch.randn_like(data1_amsss)

                data1_amsss, data2_amsss = data1_amsss.cuda(), data2_amsss.cuda()
                loss = torch.mean(self.model(data1_amsss, data2_amsss))
                if self.step % 100 == 0:
                    print(f'{self.step}: {loss.item()}')
                ulossy_amsss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)

            accuracy_loss = accuracy_loss + (ulossy_amsss/self.gradient_accumulate_every)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema_amsss()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = self.batch_size

                data1_amsss = next(self.dl)
                data2_amsss = torch.randn_like(data1_amsss)
                ogimage_amsss = data2_amsss.cuda()

                xt_amsss, direct_recons, allimages_amsss = self.ema_model.module.sample(batch_size=batches, img=ogimage_amsss)

                ogimage_amsss = (ogimage_amsss + 1) * 0.5
                utils.save_image(ogimage_amsss, str(self.results_folder / f'sample-og-{milestone}.png'), nrow=6)

                allimages_amsss = (allimages_amsss + 1) * 0.5
                utils.save_image(allimages_amsss, str(self.results_folder / f'sample-recon-{milestone}.png'), nrow = 6)

                direct_recons = (direct_recons + 1) * 0.5
                utils.save_image(direct_recons, str(self.results_folder / f'sample-direct_recons-{milestone}.png'), nrow=6)

                xt_amsss = (xt_amsss + 1) * 0.5
                utils.save_image(xt_amsss, str(self.results_folder / f'sample-xt-{milestone}.png'),
                                 nrow=6)

                accuracy_loss = accuracy_loss/(self.save_and_sample_every+1)
                print(f'Mean of last {self.step}: {accuracy_loss}')
                accuracy_loss=0
                self.save()
                if self.step % (self.save_and_sample_every * 100) == 0:
                    self.save(self.step)

            self.step += 1

        print('training completed')

    def test_from_data(self, extra_path, s_times=None):
        batches = self.batch_size
        ogimage_amsss = next(self.dl).cuda()
        X_0s, X_ts = self.ema_model.module.all_sample(batch_size=batches, img=ogimage_amsss, times=s_times)

        ogimage_amsss = (ogimage_amsss + 1) * 0.5
        utils.save_image(ogimage_amsss, str(self.results_folder / f'og-{extra_path}.png'), nrow=6)

        import imageio
        framest_amsss = []
        frames0_amsss = []
        logger=None

        for i in range(len(X_0s)):
            print(i)

            x_0 = X_0s[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), str(i))
            frames0_amsss.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png')))
            x_t = X_ts[i]
            allimages_amsss = (x_t + 1) * 0.5
            utils.save_image(allimages_amsss, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), str(i))
            framest_amsss.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png')))

        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), frames0_amsss)
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), framest_amsss)
        
        

    def sample_and_save_for_fid(self, noise=0):
        out_folder = f'{self.results_folder}_out'
        createfolder_amsss(out_folder)

        count = 0
        batchsize = 128
        for j in range(int(6400/batchsize)):

            data2_amsss = torch.randn(batchsize, 3, 128, 128)
            ogimage_amsss = data2_amsss.cuda()
            print(ogimage_amsss.shape)

            xt, direct_recons, allimages_amsss = self.ema_model.module.gen_sample(batch_size=batchsize, img=ogimage_amsss)

            for i in range(allimages_amsss.shape[0]):
                utils.save_image((allimages_amsss[i] + 1) * 0.5,
                                 str(f'{out_folder}/' + f'sample-x0-{count}.png'))


                count += 1