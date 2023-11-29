import argparse
import json

import lightning as L
import torchsummary

from dataloaders.mnist_datamodule import MNISTDataModule
from models.dcgan import DCGAN

# L.seed_everything(420)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN')
    parser.add_argument(
        '--config', type=str, required=True, help='Path to the config file'
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path) as json_file:
        return json.load(json_file)


def initialize_trainer(config):
    return L.Trainer(
        max_epochs=config['max_epochs'],
        logger=L.pytorch.loggers.TensorBoardLogger(
            config['log_dir'], name=config['logger_name']
        ),
    )


def print_model_summary(model, config):
    print('Generator Summary:')
    torchsummary.summary(model.generator, (config['latent_dim'], 1, 1), device='cpu')
    print('\nDiscriminator Summary:')
    torchsummary.summary(
        model.discriminator, (config['channels'], 28, 28), device='cpu'
    )


def main(config_path):
    config = load_config(config_path)

    dm = MNISTDataModule(config['dm'])
    model = DCGAN(config['model'])

    trainer = initialize_trainer(config['trainer'])

    print_model_summary(model, config['model'])

    trainer.fit(model, dm)


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
