#!/bin/bash
set -e

# make sure to activate the conda environment
# conda activate MNIST-GAN

# echo "Running config1"
# python train.py --config ./configs/config1.json

# echo "Running config2"
# python train.py --config ./configs/config2.json

echo "Running config3"
python train.py --config ./configs/config3.json

# echo "Running config4"
# python train.py --config ./configs/config4.json
