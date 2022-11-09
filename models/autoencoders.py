import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import torch

from utils import print_debug


class Encoder(nn.Module):
    def __init__(self, variational=False, latent_size=None, num_channels=None, kernel_size=None, act=None, no_pool=False, debug=False) -> None:
        super().__init__()
        self.debug = debug
        self.variational = variational
        mult = 2 if variational else 1
        self.latent_size = latent_size
        self.no_pool = no_pool
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        if self.kernel_size == 3:
            padding = 1
        elif self.kernel_size == 5:
            padding = 2
        else:
            print(f'\n\nkernel_size {self.kernel_size} is not supported\n\n')
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, padding=padding)
        if self.no_pool:
            self.downsample1 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=2, padding=padding)
            self.downsample2 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=2, padding=padding)
        else:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(in_features=8*8*self.num_channels, out_features=mult*latent_size)
        self.act = act
        
    def forward(self, x):   # input x is (bs, 3, 32, 32) for cifar images
        print_debug(x, self.debug, name='\nEncoder: input')
        x = self.act(self.conv1(x))
        print_debug(x, self.debug, name='Encoder: after conv1')
        x = self.act(self.downsample1(x)) if self.no_pool else self.max_pool(x)
        print_debug(x, self.debug, name='Encoder: after downsample/max_pool')
        x = self.act(self.conv2(x))
        print_debug(x, self.debug, name='Encoder: after conv2')
        x = self.act(self.downsample2(x)) if self.no_pool else self.max_pool(x)
        print_debug(x, self.debug, name='Encoder: after downsample/max_pool')
        x = self.act(self.conv3(x))
        print_debug(x, self.debug, name='Encoder: after conv3')
        x = x.view(x.shape[0], -1)  # should be (bs, 8*8*self.num_channels)
        x = self.fc(x)  # should be (bs, mult*latent_size)
        print_debug(x, self.debug, name='Encoder: after fc')
        
        if self.variational:
            out = x.view(x.shape[0], self.latent_size, 2)
        else:
            out = x
        print_debug(x, self.debug, name='Encoder: after reshape')
        return out
    
class Decoder(nn.Module):
    def __init__(self, latent_size, num_channels=None, kernel_size=None, act=None, sigmoid=None, interpolation='bilinear', no_upsample=False, debug=False) -> None:
        super().__init__()
        self.debug = debug
        self.num_channels = num_channels
        self.use_sigmoid = sigmoid
        self.no_upsample = no_upsample
        self.kernel_size = kernel_size
        if self.kernel_size == 3:
            padding = 1
        elif self.kernel_size == 5:
            padding = 2
        else:
            print(f'\n\nkernel_size {self.kernel_size} is not supported\n\n')
        self.fc = nn.Linear(in_features=latent_size, out_features=self.num_channels*8*8)  # expand latent vector into 64x 8x8 feature maps
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=self.num_channels, out_channels=3, kernel_size=self.kernel_size, stride=1, padding=padding)
        if self.no_upsample:
            self.deconv1 = nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=2, padding=padding, output_padding=1)
            self.deconv2 = nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=2, padding=padding, output_padding=1)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode=interpolation)  # do we want upsample of deconv layer?
        self.act = act
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):   # x shape: (bs, latent_size) 
        print_debug(x, self.debug, name='Decoder: input')
        x = self.act(self.fc(x))
        x = x.view(x.shape[0], self.num_channels, 8, 8)
        print_debug(x, self.debug, name='Decoder: after fc')
        x = self.act(self.conv1(x))
        print_debug(x, self.debug, name='Decoder: after conv1')
        x = self.act(self.deconv1(x)) if self.no_upsample else self.upsample(x) # self.num_channelsx16x16
        print_debug(x, self.debug, name='Decoder: after deconv1/upsample')
        x = self.act(self.conv2(x))
        print_debug(x, self.debug, name='Decoder: after conv2')
        x = self.act(self.deconv2(x)) if self.no_upsample else self.upsample(x)  # self.num_channelsx32x32
        print_debug(x, self.debug, name='Decoder: after deconv1/upsample')
        x = self.conv3(x)     # 3x32x32
        print_debug(x, self.debug, name='Decoder: after conv3')
        
        if self.use_sigmoid:
            out = self.sigmoid(x)
        else:
            out = x
        print_debug(x, self.debug, name='Decoder: after sigmoid')
        
        return out
        
        
class Autoencoder(nn.Module):
    def __init__(
        self, 
        variational=False, 
        latent_size=256, 
        num_channels=64,
        kernel_size=3,
        act=nn.ReLU,
        sigmoid=True, 
        interpolation='bilinear',
        no_pool=False,
        no_upsample=False,
        device=None,
        debug=False,
        ) -> None:
        super().__init__()
        self.encoder = Encoder(
            variational=variational, 
            latent_size=latent_size, 
            num_channels=num_channels, 
            kernel_size=kernel_size, 
            act=act, 
            no_pool=no_pool, 
            debug=debug,
            )
        self.decoder = Decoder(
            latent_size, 
            num_channels=num_channels, 
            kernel_size=kernel_size, 
            act=act, 
            sigmoid=sigmoid, 
            interpolation=interpolation, 
            no_upsample=no_upsample, 
            debug=debug,
            )
        self.variational = variational
        self.latent_size = latent_size
        self.sigmoid = nn.Sigmoid()
        self.device = device
        
    def forward(self, x):
        # encode an image into two stats vectors (means and stds)
        # sample a latent vector from the stats vectors
        # decode the latent vector into an image
        # compute reconstruction_loss between decoded and original images
        # compute gaussian_loss between stats vector and Normal distribution stats (mean=0 and std=1)

        out = self.encoder(x)   # stats is a vector (latent_size, 2) with (mean, std) rows 
        
        if self.variational:
            self.stats = out
            means = self.stats[:, :, 0]
            log_vars = self.stats[:, :, 1]   # stds have to be positive
            stds = torch.exp(0.5 * log_vars)
        
            # reparametrization trick: sample a value as mean + std * N(0, 1)
            self.normal = torch.randn_like(means)
            assert self.normal.shape == means.shape == stds.shape
            self.latent_vector = means + stds * self.normal   # (num_images, 256)
        else:
            self.latent_vector = out
        
        out = self.decoder(self.latent_vector)
        
        return out
 
    def sample(self, name=None, num_images=8, return_samples=False):
        self.eval()
        latent_vector = torch.randn(size=(num_images, self.latent_size), device=self.device)
        out = self.decoder(latent_vector)
        if return_samples:
            return out
        grid = torchvision.utils.make_grid(out.cpu(), nrow=4).permute(1, 2, 0)
        plt.figure(figsize=(7,4.5))  # assuming 2x4 images
        plt.imshow(grid)
        plt.axis('off')
        plt.savefig(f'plots/samples_{name}.png')
        #plt.clf()  # not enough, need to actually close 
        plt.close()