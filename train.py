import argparse
import json

import lightning as L
import torchsummary

from dataloaders.mnist_datamodule import MNISTDataModule
from models.cgan import ConditionalGAN
from models.dcgan import DCGAN
from models.ecgan import EncoderCGAN

L.seed_everything(911)


# Parsing command line arguments for configuration
def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN')
    parser.add_argument(
        '--config', type=str, required=True, help='Path to the config file'
    )
    return parser.parse_args()


# Loading configuration from a JSON file
def load_config(config_path):
    with open(config_path) as json_file:
        return json.load(json_file)


# Initializing the Lightning trainer with configurations
def initialize_trainer(config):
    return L.Trainer(
        max_epochs=config['max_epochs'],
        logger=L.pytorch.loggers.TensorBoardLogger(
            config['log_dir'], name=config['logger_name']
        ),
    )


# Printing summary of the model
def print_model_summary(model, config):
    if config['model_type'] == 'ECGAN':
        print('Encoder Summary:')
        torchsummary.summary(model.encoder, (config['channels'], 28, 28), device='cpu')
    print('Generator Summary:')
    torchsummary.summary(model.generator, (config['latent_dim'], 1, 1), device='cpu')
    print('\nDiscriminator Summary:')
    torchsummary.summary(
        model.discriminator, (config['channels'], 28, 28), device='cpu'
    )


def main(config_path):
    config = load_config(config_path)

    dm = MNISTDataModule(config['datamodule'])

    if config['model']['model_type'] == 'DCGAN':
        model = DCGAN(config['model'])
        print_model_summary(model, config['model'])

    elif config['model']['model_type'] == 'CGAN':
        model = ConditionalGAN(config['model'])
        # no model summary for CGAN since the dummy variables used by the torchsummary for labels are float instead of int

    elif config['model']['model_type'] == 'ECGAN':
        model = EncoderCGAN(config['model'])
        print_model_summary(model, config['model'])

    trainer = initialize_trainer(config['trainer'])

    trainer.fit(model, dm)


if __name__ == '__main__':
    args = parse_args()
    main(args.config)
