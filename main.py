from copy import deepcopy
import numpy as np
import os
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloaders import get_cifar
from utils import get_activation_function, plot_grid
from models.classifiers import evaluate_classifier, train_classifier
from models.autoencoders import Autoencoder, Encoder, Decoder
from metrics import compute_fid2, compute_inception_score

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
    

parser = get_args_parser()
args = get_args_parser().parse_args()
# use below for inline vscode cell execution
# args, unknown = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataloader, test_dataloader = get_cifar(
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

args.act_fn = get_activation_function(act_str=args.act)


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
    act=args.act_fn,
    sigmoid=args.sigmoid, 
    interpolation=args.interpolation,
    no_pool=args.no_pool,
    no_upsample=args.no_upsample,
    device=device,
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
        test_acc = evaluate_classifier(classifier, test_dataloader=test_dataloader, device=device)
        print(f'\n\nCIFAR-10 Test Accuracy {test_acc:.2f}')
    else:
        print(f'\n\nClassifier checkpoint is not found at {classifier_checkpoint}')
        classifier = train_classifier(args, classifier=None, train_dataloader=train_dataloader, test_dataloader=test_dataloader, device=device)
        
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