# generative-models
Pytorch implementation of popular generative models (early work in progress)

## Goal 
Learn about generative models and apply them to music generation (in raw waveform domain)

## Usage  
See `arguments.py` for explanation of CLI arguments and default values.

Train a small plain autoencoder with latent vector size 256 on cifar-10 images for 100 epochs:
`python main.py --latent_size 256 --sigmoid --wd 0.01 --epochs 100 --train --no_upsample --no_pool`

Train a small VAE on cifar-10 images, generates samples, computes metrics (FID, inception score, etc), logs data for tensorboard, etc) 
`python main.py --latent_size 256 --sigmoid --wd 0.01 --epochs 100 --train --variational --beta 5e-6 --no_upsample --no_pool --beta_mult 1 --tag test_tag_`

This assumes a plain autoencoder has already been trained (see above) and the corresponding checkpoint is available in checkpoints folder (saved automatically)
