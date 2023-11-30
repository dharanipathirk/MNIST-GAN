#!/bin/bash
set -e

# make sure to activate the conda environment
# conda activate MNIST-GAN

echo "Running dcGAN"
python train.py --config ./configs/dcgan.json

echo "Running conditionalGAN"
python train.py --config ./configs/cgan.json

echo "Running encoderCGAN"
python train.py --config ./configs/ecgan.json

echo "Running dcGAN-wloss"
python train.py --config ./configs/dcgan_w_loss.json

echo "Running conditionalGAN-wloss"
python train.py --config ./configs/cgan_w_loss.json

echo "Running encoderCGAN-wloss"
python train.py --config ./configs/ecgan_w_loss.json


# echo "Running experimental models"
# python train.py --config ./configs/experimental.json
