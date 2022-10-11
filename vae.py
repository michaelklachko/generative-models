# %% 
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# replace all occurences of comma followed by non-white space character with comma+space: ,(?=[^\s])


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data_dir", default="data", type=str, help="path to save outputs")
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--wd", default=0.0, type=float)
    parser.add_argument("--beta", default=0.01, type=float)
    parser.add_argument("--evaluate", dest="evaluate", help="run FP32 evaluation as baseline", action="store_true")
    
    return parser


def get_data(dataset='CIFAR10', data_dir=None, num_samples=None, train_batch_size=None, test_batch_size=None):
    if dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST(root=os.path.abspath(data_dir), train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root=os.path.abspath(data_dir), train=False, transform=transform, download=False)
        
        if num_samples is not None and num_samples < 10000:
            np.random.seed(42)
            sampled_index=np.random.choice(10000, num_samples)
            test_dataset.data = torch.tensor(np.array(test_dataset.data)[sampled_index])
            test_dataset.targets = torch.tensor(np.array(test_dataset.targets)[sampled_index])

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    elif dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        
    return train_loader, test_loader


class Encoder(nn.Module):
    def __init__(self, latent_size) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)  TODO do we need a duplicate max_pool layer?
        self.fc = nn.Linear(in_features=8*8*256, out_features=2*latent_size)
        
    def forward(self, x):
        # input x is (bs, 3, 32, 32) for cifar images
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.conv3(x)

        x = x.reshape(x.shape[0], -1)  # should be (bs, 8*8*256)
        
        x = self.fc(x)  # should be (bs, 2*latent_size)
        
        out = x.reshape(x.shape[0], self.latent_size, 2)
        
        return out
    
class Decoder(nn.Module):
    def __init__(self, latent_size) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)  # TODO should we use smaller kernels here?
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")  # do we want upsample of deconv layer?
        self.init_size = int(np.sqrt(latent_size))
        
    def forward(self, x):
        # input shape: (bs, 1, 16, 16) assuming latent_size is 256, or 8, 8 if it's 64
        x = x.reshape(x.shape[0], 1, self.init_size, self.init_size)  # TODO do we want to reshape or view (faster)?
        #x = self.upsample(x)  # do we want to upsample or convolve first?
        x = self.conv1(x)
        x = self.upsample(x) 
        x = self.conv2(x)
        x = self.upsample(x) 
        out = self.conv3(x)
        
        return out
        
        
class VAE(nn.Module):
    def __init__(self, latent_size=None) -> None:
        super().__init__()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)
        self.latent_size = latent_size
        
    
    def forward(self, x):
        # encode an image into two stats vectors (means and stds)
        # sample a latent vector from the stats vectors
        # decode the latent vector into an image
        # compute reconstruction_loss between decoded and original images
        # compute gaussian_loss between stats vector and Normal distribution stats (mean=0 and std=1)

        self.stats = self.encoder(x)   # stats is a vector (latent_size, 2) with (mean, std) rows 
        
        means = self.stats[:, :, 0]
        stds = self.stats[:, :, 1]
        
        # reparametrization trick: sample a value as mean + std * N(0, 1)
        self.normal = torch.randn((x.shape[0], self.latent_size), device='cuda')
        assert self.normal.shape == means.shape == stds.shape
        latent_vector = means + stds * self.normal
        
        out = self.decoder(latent_vector)
        
        return out

# %%
parser = get_args_parser()
#args = get_args_parser().parse_args()
args, unknown = parser.parse_known_args()


train_dataloader, test_dataloader = get_data(
    data_dir=args.data_dir, 
    train_batch_size=args.train_batch_size, 
    test_batch_size=args.test_batch_size
    )

input_image = iter(test_dataloader).next()[0][0].reshape(1, 3, 32, 32)

latent_size = 64

vae = VAE(latent_size=latent_size).cuda()
 
reconstructed_image = vae(input_image.cuda())

# %%

def plot_image(image, name='image.png'):
    image = image.squeeze(0).permute(1,2,0)
    image = image - image.min()
    image = image / image.max()
    image = image.cpu().detach()

    plt.imshow(image)
    plt.show()
    plt.savefig(name)
    
plot_image(input_image, 'original_image.png')
plot_image(reconstructed_image, 'reconstructed_image.png')

# %%

# train

# create optimizer
optim = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=0.000)

reconstruction_loss = nn.MSELoss()
beta_loss = nn.KLDivLoss()

normal_stats = torch.zeros((args.train_batch_size, latent_size, 2), device='cuda')
normal_stats[:, :, 1] = 1

count = 0
vae.train()

for epoch in range(10):
    print(f'\n\nEpoch {epoch}\n\n')
    for image, label in train_dataloader:
        image = image.cuda()
        reconstructed = vae(image)
        
        rec_loss = reconstruction_loss(image, reconstructed)
        b_loss = beta_loss(vae.stats, normal_stats)
        loss = rec_loss + args.beta * b_loss.abs()
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        count += 1
        
        if count % 1000 == 0:
            print(count, rec_loss.item(), b_loss.item())
            name = f'reconstructed_i{count}_b{args.beta}_bs{args.train_batch_size}_lr{args.lr}_wd{args.wd}.png'
            vae.eval()
            reconstructed_image = vae(input_image.cuda())
            vae.train()
            plot_image(reconstructed_image, name)
        

# %%
