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
    parser.add_argument("--train_batch_size", default=50, type=int)
    parser.add_argument("--test_batch_size", default=50, type=int)
    parser.add_argument("--latent_size", default=64, type=int)
    parser.add_argument("--print_freq", default=1000, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--wd", default=0.0, type=float)
    parser.add_argument("--beta", default=0.01, type=float)
    parser.add_argument("--evaluate", dest="evaluate", help="run FP32 evaluation as baseline", action="store_true")
    parser.add_argument("--vae", dest="vae", help="use simple autoencoder, not VAE", action="store_true")
    parser.add_argument("--sigmoid", dest="sigmoid", help="apply sigmoid at the end", action="store_true")
    
    
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
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])
        transform_test = transforms.Compose([transforms.ToTensor(), ])

        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        
    return train_loader, test_loader


class Encoder(nn.Module):
    def __init__(self, vae=False, latent_size=None) -> None:
        super().__init__()
        self.vae = vae
        mult = 2 if vae else 1
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)  TODO do we need a duplicate max_pool layer?
        self.fc = nn.Linear(in_features=8*8*128, out_features=mult*latent_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # input x is (bs, 3, 32, 32) for cifar images
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        # x = self.conv3(x)
        # x = self.relu(x)

        x = x.reshape(x.shape[0], -1)  # should be (bs, 8*8*256)
        
        x = self.fc(x)  # should be (bs, 2*latent_size)
        
        if self.vae:
            out = x.reshape(x.shape[0], self.latent_size, 2)
        else:
            out = x
        
        return out
    
class Decoder(nn.Module):
    def __init__(self, latent_size) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)  # TODO should we use smaller kernels here?
        if latent_size == 64:
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=5, stride=1, padding=2)
        elif latent_size == 256:
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2)
        else:
            raise(NotImplementedError)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")  # do we want upsample of deconv layer?
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_size = int(np.sqrt(latent_size))
        
    def forward(self, x):
        # input shape: (bs, 1, 16, 16) assuming latent_size is 256, or 8, 8 if it's 64
        x = x.reshape(x.shape[0], 1, self.init_size, self.init_size)  # TODO do we want to reshape or view (faster)?
        #x = self.upsample(x)  # do we want to upsample or convolve first?
        x = self.conv1(x)
        x = self.relu(x)
        x = self.upsample(x) 
        x = self.conv2(x)
        if self.init_size == 16:
            pass
        elif self.init_size == 8:
            x = self.relu(x)
            x = self.upsample(x) 
            x = self.conv3(x)
        else:
            raise(NotImplementedError)
        
        if args.sigmoid:
            out = self.sigmoid(x)
        else:
            out = x
        
        return out
        
        
class VAE(nn.Module):
    def __init__(self, vae=False, latent_size=None) -> None:
        super().__init__()
        self.encoder = Encoder(vae=vae, latent_size=latent_size)
        self.decoder = Decoder(latent_size)
        self.vae = vae
        self.latent_size = latent_size
        
    
    def forward(self, x):
        # encode an image into two stats vectors (means and stds)
        # sample a latent vector from the stats vectors
        # decode the latent vector into an image
        # compute reconstruction_loss between decoded and original images
        # compute gaussian_loss between stats vector and Normal distribution stats (mean=0 and std=1)

        out = self.encoder(x)   # stats is a vector (latent_size, 2) with (mean, std) rows 
        
        if self.vae:
            self.stats = out
            means = self.stats[:, :, 0]
            stds = self.stats[:, :, 1]
        
            # reparametrization trick: sample a value as mean + std * N(0, 1)
            self.normal = torch.randn((x.shape[0], self.latent_size), device='cuda')
            assert self.normal.shape == means.shape == stds.shape
            self.latent_vector = means + stds * self.normal
        else:
            self.latent_vector = out
        
        out = self.decoder(self.latent_vector)
        
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

vae = VAE(vae=args.vae, latent_size=args.latent_size).cuda()

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

# train see https://github.com/orybkin/sigma-vae-pytorch

# create optimizer
optim = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=0.000)

reconstruction_loss = nn.MSELoss()
if args.vae:
    beta_loss = nn.KLDivLoss()

    normal_stats = torch.zeros((args.train_batch_size, args.latent_size, 2), device='cuda')
    normal_stats[:, :, 1] = 1

count = 0
total_rec_loss = 0
total_b_loss = 0
vae.train()

for epoch in range(args.epochs):
    print(f'\nEpoch {epoch}\n')

    for image, label in train_dataloader:
        image = image.cuda()
        reconstructed = vae(image)
        
        rec_loss = reconstruction_loss(image, reconstructed)
        total_rec_loss += rec_loss
        loss = rec_loss
        
        if args.vae:
            b_loss = beta_loss(vae.stats, normal_stats)
            loss += args.beta * b_loss.abs()
            total_rec_loss += b_loss
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        count += 1
        
        if count % args.print_freq == 0:
            # to combat vanishing KL loss:
            if args.vae:
                args.beta *= 1.5
            else:
                args.beta = '_no_vae'
            print(count, rec_loss.item()/args.print_freq, f'{b_loss.item()/args.print_freq}' if args.vae else '')
            name = f'reconstructed_i{count}_ls{args.latent_size}_b{args.beta}_bs{args.train_batch_size}_lr{args.lr}_wd{args.wd}.png'
            vae.eval()
            with torch.no_grad():
                reconstructed_image = vae(input_image.cuda())
            plot_image(reconstructed_image, name)
            vae.train()
            total_rec_loss = 0
            total_b_loss = 0
        

# %%
