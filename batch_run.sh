#!/bin/bash
set -e

# make sure to activate the conda environment
# conda activate MNIST-GAN

echo "Running config"
python train.py --config ./configs/config.json
