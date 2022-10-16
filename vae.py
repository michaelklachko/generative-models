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
    parser.add_argument("--loss", default="mse", type=str, help="reconstruction loss function")
    parser.add_argument("--train_batch_size", default=50, type=int)
    parser.add_argument("--test_batch_size", default=50, type=int)
    parser.add_argument("--latent_size", default=64, type=int)
    parser.add_argument("--num_channels", default=64, type=int)
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
    def __init__(self, vae=False, latent_size=None, num_channels=None) -> None:
        super().__init__()
        self.vae = vae
        mult = 2 if vae else 1
        self.latent_size = latent_size
        self.num_channels = num_channels
        self.kernel_size = 3
        if self.kernel_size == 3:
            padding = 1
        elif self.kernel_size == 5:
            padding = 2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, padding=padding)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)  TODO do we need a duplicate max_pool layer?
        self.fc = nn.Linear(in_features=8*8*self.num_channels, out_features=mult*latent_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):   # input x is (bs, 3, 32, 32) for cifar images
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = x.reshape(x.shape[0], -1)  # should be (bs, 8*8*self.num_channels)
        x = self.fc(x)  # should be (bs, mult*latent_size)
        
        if self.vae:
            out = x.reshape(x.shape[0], self.latent_size, 2)
        else:
            out = x
        
        return out
    
class Decoder(nn.Module):
    def __init__(self, latent_size, num_channels=None) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = 3
        if self.kernel_size == 3:
            padding = 1
        elif self.kernel_size == 5:
            padding = 2
        self.fc = nn.Linear(in_features=latent_size, out_features=self.num_channels*8*8)  # expand latent vector into 64x 8x8 feature maps
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, padding=padding)  # TODO should we use smaller kernels here?
        self.conv2 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=self.num_channels, out_channels=3, kernel_size=self.kernel_size, stride=1, padding=padding)
        
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")  # do we want upsample of deconv layer?
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):   # x shape: (bs, latent_size) 
        x = self.fc(x)
        x = self.relu(x)
        x = x.reshape(x.shape[0], self.num_channels, 8, 8)  # TODO do we want to reshape or view (faster)?
        x = self.conv1(x)
        x = self.relu(x)
        x = self.upsample(x)  # self.num_channelsx16x16
        x = self.conv2(x)
        x = self.relu(x)
        x = self.upsample(x)  # self.num_channelsx32x32
        x = self.conv3(x)     # 3x32x32
        
        if args.sigmoid:
            out = self.sigmoid(x)
        else:
            out = x
        
        return out
        
        
class VAE(nn.Module):
    def __init__(self, vae=False, latent_size=256, num_channels=64) -> None:
        super().__init__()
        self.encoder = Encoder(vae=vae, latent_size=latent_size, num_channels=num_channels)
        self.decoder = Decoder(latent_size, num_channels=num_channels)
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
    

def plot_image(image, name='image.png'):
    image = image.squeeze(0).permute(1,2,0)
    image = image - image.min()
    image = image / image.max()
    image = image.cpu().detach()

    plt.imshow(image)
    plt.show()
    plt.savefig(name)
    
def plot_grid(model, input_images=None, name=''):
    model.eval()
    with torch.no_grad():
        reconstructed_images = model(input_images.cuda())
    
    stack = torch.stack([input_images, reconstructed_images.cpu()], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(stack, nrow=input_images.shape[0]).permute(1, 2, 0)
    plt.figure(figsize=(7,4.5))  # assuming 4 images
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(f'grid_{name}.png')
    #plt.clf()
    plt.close()
    
    
    
# %%
parser = get_args_parser()
#args = get_args_parser().parse_args()
args, unknown = parser.parse_known_args()


train_dataloader, test_dataloader = get_data(
    data_dir=args.data_dir, 
    train_batch_size=args.train_batch_size, 
    test_batch_size=args.test_batch_size
    )

vae = VAE(vae=args.vae, latent_size=args.latent_size, num_channels=args.num_channels).cuda()

input_image = iter(test_dataloader).next()[0][0].reshape(1, 3, 32, 32)
plot_image(input_image, 'original_image.png')
# reconstructed_image = vae(input_image.cuda())
# plot_image(reconstructed_image, 'reconstructed_image.png')

train_input_images = next(iter(train_dataloader))[0][:4]
test_input_images = next(iter(test_dataloader))[0][:4]

# %%

# train see https://github.com/orybkin/sigma-vae-pytorch

# create optimizer

optim = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=args.wd)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * len(train_dataloader))

if args.loss == 'mse':
    reconstruction_loss = nn.MSELoss()
elif args.loss == 'bce':
    reconstruction_loss = nn.BCELoss()
    
if args.vae:
    norm_loss = nn.KLDivLoss()

    normal_stats = torch.zeros((args.train_batch_size, args.latent_size, 2), device='cuda')
    normal_stats[:, :, 1] = 1

count = 0
total_rec_loss = 0
total_kl_loss = 0
vae.train()
print('\n\n')

for epoch in range(args.epochs):
    print(f'Epoch {epoch}  LR {lr_scheduler.get_last_lr()[0]:.5f}')

    for image, label in train_dataloader:
        image = image.cuda()
        reconstructed = vae(image)

        rec_loss = reconstruction_loss(reconstructed, image)
        total_rec_loss += rec_loss
        loss = rec_loss
        
        if args.vae:
            kl_loss = norm_loss(vae.stats, normal_stats)
            total_kl_loss += kl_loss.abs()
            loss += args.beta * kl_loss.abs()
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_scheduler.step()
        count += 1
        
        if count % args.print_freq == 0:
            # to combat vanishing KL loss:
            if args.vae:
                args.beta *= 1.05
                beta_str = f'{args.beta:.3f}'
            else:
                beta_str = '_no_vae'

            vae.eval()
            with torch.no_grad():
                reconstructed_image = vae(input_image.cuda())

            test_loss = reconstruction_loss(input_image.cuda(), reconstructed_image).item()
            kl_loss_str = f"kl beta {args.beta} loss {(total_kl_loss.item()/args.print_freq):.4f}" if args.vae else ""
            print(f'\t{count} losses: train {(total_rec_loss.item()/args.print_freq):.4f} test {test_loss:.4f} {kl_loss_str}')
            #name = f'reconstructed_i{count}_size{args.latent_size}_kl{beta_str}_bs{args.train_batch_size}_lr{args.lr:.5f}_wd{args.wd:.5f}.png'
            #plot_image(reconstructed_image, name)
            name = f'size{args.latent_size}_kl{beta_str}_bs{args.train_batch_size}_lr{args.lr:.5f}_wd{args.wd:.5f}.png'
            plot_grid(vae, train_input_images, name='train_'+name)
            plot_grid(vae, test_input_images, name='test_'+name)
            vae.train()
            total_rec_loss = 0
            total_kl_loss = 0
        

# %%
