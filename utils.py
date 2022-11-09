import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.nn as nn
import torchvision


def print_debug(x, debug=False, name=''):
    if debug:
        print(f'{name} shape: {list(x.shape)}\n\tvalues: {x.flatten()[:8]}')
        
        
def make_folder(name):
    if not os.path.isdir(name):
        os.mkdir(name)
        
        
def plot_image(image, name='image.png'):
    image = image.squeeze(0).permute(1,2,0)
    image = image - image.min()
    image = image / image.max()
    image = image.cpu().detach()

    plt.imshow(image)
    plt.show()
    plt.savefig('plots/'+name)
    
    
def plot_grid(model, input_images=None, name=''):
    model.eval()
    with torch.no_grad():
        reconstructed_images = model(input_images.to(next(model.parameters()).device))
    
    stack = torch.stack([input_images, reconstructed_images.cpu()], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(stack, normalize=True, nrow=input_images.shape[0]).permute(1, 2, 0)
    plt.figure(figsize=(7,4.5))  # assuming 4 images
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(f'plots/{name}.png')
    #plt.clf()
    plt.close()
    

def get_activation_function(act_str):
    if act_str == 'relu':
        act = nn.ReLU()
    elif act_str == 'leaky-relu':
        act = nn.LeakyReLU()
    elif act_str == 'elu':   # did not converge on cifar
        act = nn.ELU()
    elif act_str == 'selu':  # did not converge on cifar
        act = nn.SELU()
    elif act_str == 'gelu':
        act = nn.GELU()
    elif act_str == 'swish':
        act = nn.SiLU()
    elif act_str == 'mish':  # did not converge on cifar
        act = nn.Mish()
    else:
        raise(NotImplementedError)
    return act