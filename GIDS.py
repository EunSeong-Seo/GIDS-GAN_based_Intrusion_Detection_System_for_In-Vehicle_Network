import argparse
import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from PIL import Image


# set a manual seed to prevent different result while every running

manualSeed = 999
print("Random seed:", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# set hyper parameters

data_root = "./CAN_image_dataset/"  # dataset root
workers = 2  # using thread numbers
batch_size = 128  # batch_size
nc = 1  # number of channel from input images
num_epochs = 20  # number of training epochs
lr = 0.0002  # learning rate
beta1 = 0.5  # hyperparameter for adam optimizer
ngpu = 1  # number of available gpu

# CAN image preprocessing
"""## I am writing this section ##"""
dataset = dset.ImageFolder(root=data_root)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=workers
)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Generator model implement


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


"""##I have to modfiy this code##"""


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input : N x channel noise x 1 x 1
            nn.ConvTranspose2d(256, 512, (3, 4), stride=1, bias=False),
            nn.ReLU(True),
            # second layer
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            # third layer
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            # fourth layer,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            # Final layer
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


netG = Generator(ngpu).to(device)

if (device.type == "cuda") and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

print(netG)


# Discriminator model implement


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = (ngpu,)
        self.main = nn.Sequential(
            nn.Conv2d(1, 1, (4, 3), stride=(2, 1), padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(1, 1, (4, 3), stride=(2, 1), padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(1, 1, (16, 48), stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


netD = Discriminator(ngpu).to(device)

if (device.type == "cuda") and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)

print(netD)
