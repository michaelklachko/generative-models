from copy import deepcopy
import numpy as np
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloaders import get_cifar
from utils import get_activation_function, plot_grid, make_folder
from models.classifiers import evaluate_classifier, train_classifier
from models.autoencoders import Autoencoder
from metrics import compute_fid, compute_inception_score
from arguments import get_args_parser

# replace all occurences of comma followed by non-white space character with comma+space: ,(?=[^\s])


def train_one_epoch(args, model, epoch=None, beta=None, train_dataloader=None, optim=None, lr_scheduler=None, device=None, writer=None):
    model.train()
    total_train_rec_loss = 0
    total_train_kl_loss = 0
    count = 0
    
    for image, label in train_dataloader:
        image = image.to(device)
        reconstructed = model(image)

        train_rec_loss = F.mse_loss(reconstructed, image)
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

            total_train_kl_loss += beta * train_kl_loss
            train_loss += beta * train_kl_loss
            
        optim.zero_grad()
        train_loss.backward()
        optim.step()
        lr_scheduler.step()
        
        if writer is not None:
            i = epoch * len(train_dataloader) + count
            writer.add_scalar('Train Loss/KL', train_kl_loss, i)
            writer.add_scalar('Train Loss/Reconstruction', train_rec_loss, i)
            writer.add_scalar('Train Loss/Total', train_loss, i)
            writer.add_scalar('Train Loss/beta', beta, i)
            
        count += 1
        
    # to combat vanishing KL loss:
    if args.variational:
        beta *= args.beta_mult
    
    avg_train_rec_loss = total_train_rec_loss / len(train_dataloader)
    avg_train_kl_loss = total_train_kl_loss / len(train_dataloader)
    return beta, avg_train_rec_loss, avg_train_kl_loss


def compute_test_loss(args, model, beta=None, test_dataloader=None, device=None):
    # compute test loss (on test dataset)
    model.eval()
    with torch.no_grad():
        total_test_rec_loss = 0
        total_test_kl_loss = 0
        latent_mean = 0
        latent_std = 0
        
        for test_image, label in test_dataloader:
            test_image = test_image.to(device)
            test_reconstructed = model(test_image)

            test_rec_loss = F.mse_loss(test_reconstructed, test_image)
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
                
        
        num_test_batches = len(test_dataloader)
        avg_test_kl_loss = beta * total_test_kl_loss / num_test_batches
        avg_test_rec_loss = total_test_rec_loss / num_test_batches
        avg_latent_mean = latent_mean / num_test_batches
        avg_latent_std = latent_std / num_test_batches
        
        return avg_latent_mean, avg_latent_std, avg_test_rec_loss, avg_test_kl_loss
    
    
def compute_metrics(autoencoder=None, classifier=None, real_images=None, samples=None, debug=False):
    #fid_ae_dynamic = compute_fid2(model=model_we_are_training, images1=real_images, images2=samples)
    fid_ae = compute_fid(model=autoencoder, images1=real_images, images2=samples)  # use pretrained cifar model to compute features
    fid_classifier = compute_fid(model=classifier, images1=real_images, images2=samples)
    confidence, diversity = compute_inception_score(classifier, samples, debug=debug)   
    return fid_classifier, fid_ae, confidence, diversity


def main(args):
    # create necessary directories as needed
    for dir_name in ['test', 'checkpoints', 'data', 'plots', 'logs']:
        make_folder(dir_name)

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

    beta = args.beta
    init_epoch = 0
            
    if args.variational:
        model_type = f'vae_{args.beta}x{args.beta_mult}'
        if os.path.isfile(args.ae_checkpoint):
            print(f'\n\nLoading pretrained VAE for FID computation from {args.ae_checkpoint}')
            plain_ae = torch.load(args.ae_checkpoint)
        else:
            print(f'\n\nTo compute FID score using a plain autoencoder, provide valid path to checkpoint, or train one:')
            print(f'\n\npython main.py --latent_size 256 --sigmoid --wd 0.01 --epochs 100 --train --no_upsample --no_pool\n\n')
            raise(SystemExit)
        
        if args.classifier_checkpoint is None:
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
            
        if isinstance(plain_ae, dict) or isinstance(classifier, dict):
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
    print(f'\n\n{experiment_str}')

    if args.log_tb:
        # launch with 'tensorboard --host 0.0.0.0 --logdir args.tb_dir' on the gateway and go to gateway_ip:6006
        writer = SummaryWriter(args.tb_dir + '/' + experiment_str)
        # TODO should we save this writer to checkpoints?
    else:
        writer = None
        
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
        if args.variational:
            fid_classifier, fid_ae, confidence, diversity = compute_metrics(
                autoencoder=plain_ae, 
                classifier=classifier, 
                real_images=train_input_images, 
                samples=test_input_images, 
                debug=True,
            )
            ref_metrics_str = f'Reference (real images) metrics:  FID (classifier/AE): ' + \
                            f'{fid_classifier:.1f}/{fid_ae:.1f}  confidence {confidence:.1f}  diversity {diversity:.1f}\n\n'
        else:
            ref_metrics_str = ''
        
        print(f'\n\nTraining {model_type} model on CIFAR-10 images\n\n{ref_metrics_str}')
        
        for epoch in range(init_epoch, args.epochs, 1):
            beta, avg_train_rec_loss, avg_train_kl_loss = train_one_epoch(
                args,
                model, 
                epoch=epoch,
                beta=beta, 
                train_dataloader=train_dataloader, 
                optim=optim, 
                lr_scheduler=lr_scheduler,
                device=device,
                writer=writer,
            )

            avg_latent_mean, avg_latent_std, avg_test_rec_loss, avg_test_kl_loss = compute_test_loss(
                args, 
                model, 
                beta=beta, 
                test_dataloader=test_dataloader, 
                device=device,
            )
            avg_test_loss = avg_test_rec_loss + avg_test_kl_loss if args.variational else avg_test_rec_loss
            
            if args.variational:
                samples = model.sample(num_images=args.test_batch_size, return_samples=True)
                fid_classifier, fid_ae, confidence, diversity = compute_metrics(
                    autoencoder=plain_ae, 
                    classifier=classifier, 
                    real_images=test_input_images, 
                    samples=samples, 
                    debug=True,
                )
                metrics_str = f'  FID (classifier/AE): {fid_classifier:.1f}/{fid_ae:.1f}  confidence {confidence:.1f}  diversity {diversity:.1f}'
            else:
                metrics_str = ''
                
            if writer is not None:
                i = epoch * len(train_dataloader)
                writer.add_scalar('Test Loss/KL', avg_test_kl_loss, i)
                writer.add_scalar('Test Loss/Reconstruction', avg_test_rec_loss, i)
                writer.add_scalar('Test Loss/Total', avg_test_loss, i)
                writer.add_scalar('Test Loss/beta', beta, i)
                writer.add_scalar('Latent Vector/mean', avg_latent_mean, i)
                writer.add_scalar('Latent Vector/std', avg_latent_std, i)
                writer.add_scalar('FID/classifier', fid_classifier, i)
                writer.add_scalar('FID/plain_ae', fid_ae, i)
                writer.add_scalar('Metrics/confidence', confidence, i)
                writer.add_scalar('Metrics/diversity', diversity, i)
                
            latent_stats_str = f'  mean {avg_latent_mean:.4f} std {avg_latent_std:.4f}' if args.variational else ''    
            kl_loss_str = f'  kl train {1000*avg_train_kl_loss:.2f} test {1000*avg_test_kl_loss:.2f}' if args.variational else ''
            loss_str = f'losses:  train {1000*avg_train_rec_loss:.2f} test {1000*avg_test_rec_loss:.2f}{kl_loss_str}'
            changes_str = f'  LR {lr_scheduler.get_last_lr()[0]:.5f}' + (f' beta {beta:.6f}' if args.variational else '')
            time_str = f'{str(datetime.now())[:-7]}'
            print(f'{time_str}  Epoch {epoch:>3d}   {loss_str}{latent_stats_str}{metrics_str}{changes_str}')
            
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
        
        if not os.path.isfile(args.ae_checkpoint):  # save full model checkpoint 
            torch.save(model, 'checkpoints/' + experiment_str + "_full_model.pth")


if __name__ == "__main__":
    # use below for inline vscode cell execution
    # parser = get_args_parser()
    # args, unknown = parser.parse_known_args()
    args = get_args_parser().parse_args()
    main(args)
    
    # TODO:
    # 2. Inception Score - official one
    # 3. Tensorboard support - DONE
    # 5. save logs of train output
    # 6. add CelebA dataset

    # add vae-vq
    # implement MUSIQ
    # add larger models (resnets)
