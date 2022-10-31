# %% 
from copy import deepcopy
from genericpath import isfile
from turtle import forward
import numpy as np
from scipy.linalg import sqrtm
import os
from datetime import datetime
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# replace all occurences of comma followed by non-white space character with comma+space: ,(?=[^\s])


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data_dir", default="data", type=str, help="path to dataset")
    parser.add_argument("--checkpoint", default=None, type=str, help="path to model checkpoint")
    parser.add_argument("--fid_model_checkpoint", default='checkpoints/plain-ae_latent256_chan64_pool-stride_upsample-deconv_bs50_lr0.001_wd0.01_e100_full_model.pth', type=str, help="path to model checkpoint")
    parser.add_argument("--tag", default="", type=str, help="string to prepend when saving checkpoints")
    parser.add_argument("--debug", dest="debug", help="print out shapes and values of intermediate outputs", action="store_true")
    parser.add_argument("--loss", default="mse", type=str, help="reconstruction loss function")
    parser.add_argument("--train_batch_size", default=50, type=int)
    parser.add_argument("--test_batch_size", default=500, type=int)
    parser.add_argument("--latent_size", default=256, type=int)
    parser.add_argument("--num_channels", default=64, type=int)
    parser.add_argument("--kernel_size", default=3, type=int)
    parser.add_argument("--print_freq", default=1000, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--wd", default=0.0, type=float)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--act", default='gelu', type=str, help='relu, leaky-relu, elu, selu, gelu, swish, mish')
    parser.add_argument("--beta", default=0.01, type=float)
    parser.add_argument("--beta_mult", default=1.05, type=float)
    parser.add_argument("--sample", dest="sample", help="generate an image from a random latent vector", action="store_true")
    parser.add_argument("--evaluate", dest="evaluate", help="use pretrained model to reconstruct 4 images", action="store_true")
    parser.add_argument("--train", dest="train", help="train model", action="store_true")
    parser.add_argument("--variational", dest="variational", help="use simple autoencoder, not variational", action="store_true")
    parser.add_argument("--sigmoid", dest="sigmoid", help="apply sigmoid at the end", action="store_true")
    parser.add_argument("--mse", dest="mse", help="ise MSE loss instead of KL loss", action="store_true")
    parser.add_argument("--no_pool", dest="no_pool", help="don't use max pooling for downsampling, use stride=2 conv", action="store_true")
    parser.add_argument("--no_upsample", dest="no_upsample", help="don't use nn.Upsample for upsampling, use deconv", action="store_true")
    parser.add_argument("--interpolation", default="nearest", type=str, help=f"upsample interpolation mode in the decoder, "
                        f"'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'")
    
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


class Classifier(nn.Module):
    def __init__(self, args=None) -> None:
        super().__init__()
        self.debug = args.debug
        self.latent_size = args.latent_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(in_features=256*4*4, out_features=self.latent_size)
        self.fc_out = nn.Linear(in_features=self.latent_size, out_features=10)
        # self.fc1 = nn.Linear(in_features=128*8*8, out_features=512)
        # self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.bn4 = nn.BatchNorm1d(num_features=self.latent_size)
        # self.bn5 = nn.BatchNorm1d(num_features=512)
        self.dropout = nn.Dropout(p=args.dropout, inplace=False)
        self.act = args.act_fn
        
    def forward(self, x):
        print_debug(x, self.debug, name='\nClassifier: input')
        x = self.act(self.bn1(self.conv1(x)))
        print_debug(x, self.debug, name='Classifier: after conv1')
        x = self.max_pool(x)
        print_debug(x, self.debug, name='Classifier: after maxpool')
        x = self.act(self.bn2(self.conv2(x)))
        print_debug(x, self.debug, name='Classifier: after conv2')
        x = self.max_pool(x)
        print_debug(x, self.debug, name='Classifier: after maxpool')
        
        x = self.act(self.bn3(self.conv3(x)))
        print_debug(x, self.debug, name='Classifier: after conv3')
        x = self.max_pool(x)
        print_debug(x, self.debug, name='Classifier: after maxpool')
        x = x.view(-1, 4*4*256)
        print_debug(x, self.debug, name='Classifier: after reshape')
        x = self.act(self.bn4(self.fc(x)))
        print_debug(x, self.debug, name='Classifier: after fc')
        x = self.dropout(x)
        self.latent_vector = x
        x = self.fc_out(x)
        print_debug(x, self.debug, name='Classifier: after fc_out')

        # x = x.view(-1, 8*8*128)
        # print_debug(x, self.debug, name='Classifier: after reshape')
        # x = self.act(self.bn5(self.fc1(x)))
        # print_debug(x, self.debug, name='Classifier: after fc1')
        # x = self.dropout(x)
        # if not self.train:
        #     self.features = x
        # x = self.fc2(x)
        # print_debug(x, self.debug, name='Classifier: after fc2')
        
        return x
        
    

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
        debug=False,
        ) -> None:
        super().__init__()
        self.encoder = Encoder(variational=variational, latent_size=latent_size, num_channels=num_channels, kernel_size=kernel_size, act=act, no_pool=no_pool, debug=debug)
        self.decoder = Decoder(latent_size, num_channels=num_channels, kernel_size=kernel_size, act=act, sigmoid=sigmoid, interpolation=interpolation, no_upsample=no_upsample, debug=debug)
        self.variational = variational
        self.latent_size = latent_size
        self.sigmoid = nn.Sigmoid()
        
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
            self.normal = torch.randn((x.shape[0], self.latent_size), device=device)
            assert self.normal.shape == means.shape == stds.shape
            self.latent_vector = means + stds * self.normal   # (num_images, 256)
        else:
            self.latent_vector = out
        
        out = self.decoder(self.latent_vector)
        
        return out
 
    def sample(self, name=None, num_images=8, return_samples=False):
        self.eval()
        latent_vector = torch.randn(size=(num_images, model.latent_size), device=device)
        out = self.decoder(latent_vector)
        if return_samples:
            return out
        grid = torchvision.utils.make_grid(out.cpu(), nrow=4).permute(1, 2, 0)
        plt.figure(figsize=(7,4.5))  # assuming 2x4 images
        plt.imshow(grid)
        plt.axis('off')
        plt.savefig(f'plots/samples_{name}.png')
        #plt.clf()
        plt.close()
        

def compute_fid(model, images1, images2):
    from scipy.linalg import sqrtm
    # compute feature vectors for real and generated images, using model
    # compute feature-wise statistics (means and covariances) for the feature vectors
    # compute distance between statistics, using FID formula
    model.eval()
    with torch.no_grad():
        # need to pass input through the entire model (encoder, sampler, decoder) to get features (latent_vector)
        _ = model(images1)
        features1 = model.latent_vector.cpu().numpy()  # (bs, num_features)
        _ = model(images2)
        features2 = model.latent_vector.cpu().numpy()
    
    means1 = features1.mean(0)    # (bs, num_features) --> (num_features)
    means2 = features2.mean(0)
    
    # calculate mean and covariance statistics
    sigma1 = np.cov(features1, rowvar=False)
    sigma2 = np.cov(features2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((means1 - means2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def compute_fid2(model=None, images1=None, images2=None, eps=1e-6):
    # images1 are reference images from test dataset ('cifar', 'celeba', etc) OR a batch of images in torch format
    # images2 are samples from the model we want to evaluate
    # model can be an actual model object, or a string pointing to a saved model checkpoint - must be full model checkpoint, not a dict
    
    # https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    images1 = images1.to(images2.device)
    model = model.to(images2.device)
    
    # compute feature vectors for real and generated images, using model
    # compute feature-wise statistics (means and covariances) for the feature vectors
    # compute distance between statistics, using FID formula
    model.eval()
    with torch.no_grad():
        # need to pass input through the entire model (encoder, sampler, decoder) to get features (latent_vector)
        _ = model(images1)
        features1 = model.latent_vector.cpu().numpy()  # (bs, num_features)
        _ = model(images2)
        features2 = model.latent_vector.cpu().numpy()
    
    mu1 = features1.mean(0)    # (bs, num_features) --> (num_features)
    mu2 = features2.mean(0)
    
    sigma1 = np.cov(features1, rowvar=False)
    sigma2 = np.cov(features2, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f'fid calculation produces singular product, adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def compute_inception_score(classifier, images, N=3, alpha=0.7, debug=False):
    """
    Inception Score measures:
    - classifier confidence in detected objects in an image
    - diversity of detected objects in a batch of images
    
    Input is logits before softmax has been applied 
    
    0. Apply softmax to logits to produce output vector of class probabilities 
    
    Confidence score computation:
    1. Sort output values
    2. Compute ratio of the sum of top-N values to the total sum, for N=1 and N=5
    3. Weight the combination of the two: confidence score CS = a*S(N=1) + (1-a)*S(N=5), for a in [0, 1]
    
    Diversity score computation:
    1. Record highest predicted class ID for each image in the batch --> vector of class IDs
    2. Count frequency of each class ID  (ideally it should be ~num_images/num_classes)
    3. Apply confidence score algo to frequencies: diversity score DS = CS(freq_count(class_ids))
    
    Alternatively, we could:
        1. compute deviations from class_freq to num_images/num_classes for each class
        2. apply confidence score to this vector, or compute MSE/CE between freq vector and vector of num_images/num_classes
        
    Alternatively, in both cases (confidence and diversity), we can simply compute the std of the vector of
    interest (class probabilities or class ID frequencies)
    """
    
    # images shape: (batch_size, latent_size)
    device = images.device
    classifier = classifier.to(device)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(images)
        predictions = torch.softmax(logits, dim=1)  # predictions shape: (batch_size, num_classes)
        num_classes = predictions.shape[1]
        top1_ratio = (predictions.max(1)[0] / predictions.sum(1)).mean()
        
        if N > 1:
            sorted_predictions = torch.sort(predictions, dim=1, descending=True)  # returns two tensors: (sorted values (min to max), their orig indices).
            top1_ratio_test = (sorted_predictions[0][:, 0]).mean()  # don't need to divide by total_sum because it's 1 (because of softmax)
            assert torch.allclose(top1_ratio, top1_ratio_test, atol=1e-6)
            topN_ratio = (sorted_predictions[0][:, :N].sum(1)).mean()
            confidence_score = alpha * top1_ratio + (1-alpha) * topN_ratio
        else:
            confidence_score = top1_ratio
        
        # compute std or variance between predictions and means (1/num_classes)
        
        # means = torch.ones_like(predictions) / num_classes
        # mse = ((predictions - means) ** 2).sum(1) / num_classes
        # mse_ref = torch.nn.MSELoss()(predictions, means)
        # variance = predictions.var()
        # # var(unbiased=False) will match the mse, otherwise it will be divided by num_classes-1, not num_classes
        # vars = torch.var(predictions, dim=1, unbiased=False)  

        predicted_class_ids = torch.argmax(predictions, dim=1)  # should be vector of class ids
        class_frequencies = [0] * num_classes
        for class_id in predicted_class_ids:
            class_frequencies[class_id] += 1
        
        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        
        print(f'\nGenerated samples are classified as:')
        for l, f in zip(labels, class_frequencies):
            print(l, f)
        print()
        
        class_frequencies = torch.tensor(class_frequencies, dtype=torch.float, device=device)
        top1_ratio = class_frequencies.max() / class_frequencies.sum()
        
        if N > 1:
            sorted_class_frequencies = torch.sort(class_frequencies, descending=True)
            total_sums = sorted_class_frequencies[0].sum()
            top1_ratio_test = (sorted_class_frequencies[0][0] / total_sums)
            assert torch.allclose(top1_ratio, top1_ratio_test, atol=1e-6)
            topN_ratio = (sorted_class_frequencies[0][:N].sum() / total_sums)
            diversity_score1 = 1 - (alpha * top1_ratio + (1-alpha) * topN_ratio)
        else:
            diversity_score1 = 1 - top1_ratio
        
        # construct the batch with worst possible diversity of class IDs:
        min_diversity_batch = [0] * (num_classes - 1) + [predictions.shape[0]]
        min_diversity_batch = torch.tensor(min_diversity_batch, dtype=torch.float, device=device)
        # best diversity batch will have std = 0
        # compare current batch diversity to the worst diversity and invert (low score should indicate low diversity)
        diversity_score2 = 1 - class_frequencies.std() / min_diversity_batch.std()
        
    return 100*confidence_score, 100*diversity_score1, 100*diversity_score2
    
      
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
    
def print_debug(x, debug=False, name=''):
    if debug:
        print(f'{name} shape: {list(x.shape)}\n\tvalues: {x.flatten()[:8]}')


def compute_accuracy(outputs, labels):
    with torch.no_grad():
        batch_size = labels.size(0)
        pred = outputs.max(1)[1]
        acc = pred.eq(labels).sum().item() * 100.0 / batch_size
        return acc
    
def train_classifier(args):
    # python autoencoder.py --lr 0.001 --wd 0.1 --epochs 100 --dropout 0 --act relu
    print(f'\n\nTraining CIFAR-10 Classifier\n\n')
    
    classifier = Classifier(args=args).to(device)
    optim = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=args.epochs * len(train_dataloader))

    classifier.train()
    for epoch in range(args.epochs):
        classifier.train()
        tr_accs = []
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            loss = loss_fn(outputs, labels)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_scheduler.step()
            
            tr_acc = compute_accuracy(outputs, labels)
            tr_accs.append(tr_acc)
        
        train_acc = np.mean(tr_accs)
        test_acc = evaluate_classifier(classifier)
            
        print(f'Epoch {epoch}  train {train_acc:.2f}  test {test_acc:.2f}  LR {lr_scheduler.get_last_lr()[0]:.4f}')
        
    torch.save(classifier, f'checkpoints/cifar_classifier_{args.latent_size}.pth')
    # torch.save(classifier, f'checkpoints/{args.tag}cifar_classifier_{args.latent_size}.pth')
    return classifier

def evaluate_classifier(classifier):
    te_accs = []
    classifier.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            te_acc = compute_accuracy(outputs, labels)
            te_accs.append(te_acc)
    return np.mean(te_accs)


parser = get_args_parser()
args = get_args_parser().parse_args()
# use below for inline vscode cell execution
# args, unknown = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataloader, test_dataloader = get_data(
    data_dir=args.data_dir, 
    train_batch_size=args.train_batch_size, 
    test_batch_size=args.test_batch_size
    )

# load images from disk for plotting (otherwise train images are randomly picked every time)
if os.path.isfile('data/train_examples.pth') and os.path.isfile('data/test_examples.pth'):
    train_input_images = torch.load('data/train_examples.pth')
    test_input_images = torch.load('data/test_examples.pth')
else:
    train_input_images = next(iter(train_dataloader))[0]
    test_input_images = next(iter(test_dataloader))[0]
    torch.save(train_input_images, 'data/train_examples.pth')
    torch.save(test_input_images, 'data/test_examples.pth')
    
if test_input_images.shape[0] < args.latent_size:  # number of images should be >= number of features (to compute FID)
    print(f'\n\nNumber of test images {test_input_images.shape[0]} should be >= number of '
          f'features {args.latent_size} in order to compute FID properly (only applies to VAE)\n\n')
    if args.variational:
        raise(SystemExit)

if args.act == 'relu':
    act = nn.ReLU()
elif args.act == 'leaky-relu':
    act = nn.LeakyReLU()
elif args.act == 'elu':   # did not converge on cifar
    act = nn.ELU()
elif args.act == 'selu':  # did not converge on cifar
    act = nn.SELU()
elif args.act == 'gelu':
    act = nn.GELU()
elif args.act == 'swish':
    act = nn.SiLU()
elif args.act == 'mish':  # did not converge on cifar
    act = nn.Mish()
else:
    raise(NotImplementedError)

args.act_fn = act


if args.checkpoint is not None:
    print(f'\n\nLoading model checkpoint from {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint)
    saved_args = checkpoint['args']

    print(f'\n\nCurrent arguments:\n')
    for current_arg in vars(args):
        print(current_arg, getattr(args, current_arg))
    print(f'\n\nCheckpoint arguments:\n') 
    for saved_arg in vars(saved_args):
        print(saved_arg, getattr(saved_args, saved_arg))


model = Autoencoder(
    variational=args.variational, 
    latent_size=args.latent_size, 
    num_channels=args.num_channels,
    kernel_size=args.kernel_size,
    act=act,
    sigmoid=args.sigmoid, 
    interpolation=args.interpolation,
    no_pool=args.no_pool,
    no_upsample=args.no_upsample,
    debug=args.debug,
    ).to(device)

# train see https://github.com/orybkin/sigma-vae-pytorch

if args.loss == 'mse':
    reconstruction_loss = nn.MSELoss()
elif args.loss == 'bce':
    reconstruction_loss = nn.BCELoss()

beta = args.beta
init_epoch = 0
num_train_batches = len(train_dataloader)
num_test_batches = len(test_dataloader)

if args.variational:
    model_type = f'vae_{args.beta}x{args.beta_mult}'
    print(f'\n\nLoading pretrained VAE for FID computation from {args.fid_model_checkpoint}')
    fid_model = torch.load(args.fid_model_checkpoint)
    classifier_checkpoint = f'checkpoints/cifar_classifier_{args.latent_size}.pth'
    print(f'\n\nInstantiating Classifier for FID computation')
    if os.path.isfile(classifier_checkpoint):
        print(f'\n\nFound checkpoint at {classifier_checkpoint}, loading...')
        classifier = torch.load(classifier_checkpoint)
        test_acc = evaluate_classifier(classifier)
        print(f'\n\nCIFAR-10 Test Accuracy {test_acc:.2f}')
    else:
        print(f'\n\nClassifier checkpoint is not found at {classifier_checkpoint}')
        classifier = train_classifier(args)
        
    if isinstance(fid_model, dict) or isinstance(classifier, dict):
        raise NotImplementedError('\n\nmodel checkpoint must be a full saved model, not a state_dict\n\n')
else:
    model_type = 'plain-ae'
    
optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * len(train_dataloader))

if args.checkpoint is not None:
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    init_epoch = checkpoint['epoch']
    beta = checkpoint['current_beta']
    
pool_str = 'pool-stride' if args.no_pool else 'pool-max'
upsample_str = 'upsample-deconv' if args.no_upsample else f'upsample-{args.interpolation}'
    
experiment_str = args.tag + f'{model_type}_latent{args.latent_size}_chan{args.num_channels}_{pool_str}_{upsample_str}_bs{args.train_batch_size}_lr{args.lr}_wd{args.wd}_e{args.epochs}'
print(f'\n\n{experiment_str}\n\n')
    
if args.evaluate:
    print(f'\n\nEvaluating model')
    plot_grid(model, train_input_images[:4], name=experiment_str+'_train')
    plot_grid(model, test_input_images[:4], name=experiment_str+'_test')
    print(f'\n\nPlots saved to plots/{experiment_str}_train-test.png\n\n')
    
if args.sample:
    print(f'\n\nSampling 8 images from a random latent vector')
    model.sample(name=experiment_str)
    print(f'\n\nPlot saved to plots/samples_{experiment_str}.png\n\n')
    
if args.train:
    for epoch in range(init_epoch, args.epochs, 1):
        model.train()
        total_train_rec_loss = 0
        total_train_kl_loss = 0
        total_train_loss = 0
        
        for image, label in train_dataloader:
            image = image.to(device)
            reconstructed = model(image)

            train_rec_loss = reconstruction_loss(reconstructed, image)
            total_train_rec_loss += train_rec_loss
            train_loss = train_rec_loss
            
            if args.variational:
                mu = model.stats[:, :, 0]
                log_var = model.stats[:, :, 1]
                sigma = torch.exp(0.5 * log_var)
                if args.mse:
                    mu_loss = F.mse_loss(mu, torch.zeros_like(mu), reduction='none')
                    sigma_loss = F.mse_loss(sigma, torch.ones_like(sigma), reduction='none')
                    train_kl_loss = mu_loss.sum(1).mean(0) + sigma_loss.sum(1).mean(0)
                else:
                    # explained in https://arxiv.org/abs/1906.02691 
                    train_kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                total_train_kl_loss += train_kl_loss
                train_loss += beta * train_kl_loss
                
            optim.zero_grad()
            train_loss.backward()
            optim.step()
            lr_scheduler.step()
            
        # to combat vanishing KL loss:
        if args.variational:
            beta *= args.beta_mult

        # compute test loss (on test dataset)
        model.eval()
        with torch.no_grad():
            total_test_rec_loss = 0
            total_test_kl_loss = 0
            total_test_loss = 0
            latent_mean = 0
            latent_std = 0
            for test_image, label in test_dataloader:
                test_image = test_image.to(device)
                test_reconstructed = model(test_image)

                test_rec_loss = reconstruction_loss(test_reconstructed, test_image)
                total_test_rec_loss += test_rec_loss
                
                if args.variational:
                    mu = model.stats[:, :, 0]
                    log_var = model.stats[:, :, 1]
                    sigma = torch.exp(0.5 * log_var)
                    if args.mse:
                        mu_loss = F.mse_loss(mu, torch.zeros_like(mu), reduction='none')
                        sigma_loss = F.mse_loss(sigma, torch.ones_like(sigma), reduction='none')
                        test_kl_loss = mu_loss.sum(1).mean(0) + sigma_loss.sum(1).mean(0)
                    else:
                        test_kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                    total_test_kl_loss += test_kl_loss
                    latent_mean += mu.abs().mean()
                    latent_std += sigma.abs().mean()
                    
            total_test_loss = total_test_rec_loss + beta * total_test_kl_loss if args.variational else total_test_rec_loss
           
        if args.variational:
            samples = model.sample(num_images=args.test_batch_size, return_samples=True)
            #fid1 = compute_fid(model, test_input_images, samples)
            fid_vae = compute_fid2(model=fid_model, images1=test_input_images, images2=samples)  # use pretrained cifar model to compute features
            #fid_vae_dynamic = compute_fid2(model=model, images1=test_input_images, images2=samples)
            
            fid_classifier = compute_fid2(classifier, images1=test_input_images, images2=samples)
            #print(f'\n\tFID computed on {args.test_batch_size} feature vectors ({args.latent_size} features): {fid1:.2f} {fid2:.2f}\n')
            inception_score = compute_inception_score(classifier, samples)
            confidence, diversity1, diversity2 = inception_score
            
            fid_vae_ref = compute_fid2(model=fid_model, images1=test_input_images, images2=image)  # use pretrained cifar model to compute features
            #fid_vae_dynamic = compute_fid2(model=model, images1=test_input_images, images2=samples)
            
            fid_classifier_ref = compute_fid2(classifier, images1=test_input_images, images2=image)
            #print(f'\n\tFID computed on {args.test_batch_size} feature vectors ({args.latent_size} features): {fid1:.2f} {fid2:.2f}\n')
            inception_score_ref = compute_inception_score(classifier, image, debug=True)
            confidence_ref, diversity1_ref, diversity2_ref = inception_score_ref
            
            fid_str = f'  fid {fid_classifier:.1f}/{fid_vae:.1f} (ref {fid_classifier_ref:.1f}/{fid_vae_ref:.1f})  confidence {confidence:.1f} (ref {confidence_ref:.1f}) diversity {diversity1:.1f}/{diversity2:.1f} (ref {diversity1_ref:.1f}/{diversity2_ref:.1f})'
        else:
            fid_str = '' 
            
        latent_stats_str = f'  mean {(latent_mean/num_test_batches):.4f} std {(latent_std/num_test_batches):.4f}' if args.variational else ''    
        kl_loss_str = f'  kl train {(1000*beta*total_train_kl_loss/num_train_batches):.2f} test {(1000*beta*total_test_kl_loss/num_test_batches):.2f}' if args.variational else ""
        loss_str = f'losses:  train {(1000*total_train_rec_loss/num_train_batches):.2f} test {(1000*total_test_rec_loss/num_test_batches):.2f}{kl_loss_str}'
        changes_str = f'  LR {lr_scheduler.get_last_lr()[0]:.5f}' + (f' beta {beta:.4f}' if args.variational else '')
        time_str = f'{str(datetime.now())[:-7]}'
        print(f'{time_str}  Epoch {epoch:>3d}   {loss_str}{latent_stats_str}{fid_str}{changes_str}')
        
        plot_grid(model, train_input_images[:4], name=experiment_str+'_train')
        plot_grid(model, test_input_images[:4], name=experiment_str+'_test')
                
        # checkpointing:
        checkpoint = {}
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['optimizer'] = optim.state_dict()
        checkpoint['lr_scheduler'] = lr_scheduler.state_dict() 
        checkpoint['epoch'] = epoch
        checkpoint['args'] = args
        checkpoint['current_beta'] = beta
        checkpoint['losses'] = loss_str
        
        path = 'checkpoints/' + experiment_str + ".pth"
        torch.save(checkpoint, path)
        
        model.sample(name=experiment_str)
    
    # torch.save(model, 'checkpoints/' + experiment_str + "_full_model.pth")


# TODO:
# 1. FID scores - number of images (orig/generated batches) should be greater than latent size  - DONE
# 2. Inception Score - WIP
# 2. Different act functions - DONE
# 3. Tensorboard support
# 4. Plots of loss, FID, mean, var
# 5. save logs of train output
# 6. add CelebA dataset
# %%
