from diffusion.diffusion import Trainer
from diffusion.GaussianDiffusion import GaussianDiffusion
from diffusion.Unet import Unet
import torchvision
import os
import errno
import shutil
import argparse
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--train_steps', default=700000, type=int)
parser.add_argument('--save_folder', default='./cifar10_results_32', type=str)
parser.add_argument('--data_path', default='../cifar10_test/', type=str)
parser.add_argument('--load_path', default='./cifar10_model', type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='default', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--loss_type', default='l1', type=str)

args = parser.parse_args()
print(args)

model = Unet(
    dim = 64,
    hidden_dim = (1, 2, 4, 8),
    channels=3,
    with_time_embedding_amsss=not(args.remove_time_embed),
    residual=args.residual
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    channels = 3,
    timesteps = args.time_steps,
    loss_type = args.loss_type,
    train_routine = args.train_routine,
    sampling_routine = args.sampling_routine
).cuda()

diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

trainer = Trainer(
    diffusion,
    args.data_path,
    image_size = 32,
    train_batch_size = 10,
    train_lr = 2e-5,
    train_num_steps = args.train_steps,
    gradient_accumulate_every = 2,
    ema_decay = 0.995,
    fp16 = False,
    results_folder = args.save_folder,
    load_path = args.load_path,
    dataset = 'test'
)


trainer.sample_and_save_for_fid()