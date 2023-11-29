import random

import lightning as L
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.calculate_stats = config['calculate_stats']
        self.default_mean = config['default_mean']
        self.default_std = config['default_std']
        self.num_workers = config['num_workers']
        self.mean = self.default_mean
        self.std = self.default_std
        self.use_augmentation = config['use_augmentation']
        self.augmentations_per_image = config['augmentations_per_image']
        self.augmentation_transforms = [
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(
                degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
            ),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.6),
        ]

    def prepare_data(self):
        # Download only once
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

        if self.calculate_stats:
            # Initialize a dataset with transform just for calculating mean and std
            mnist_for_calculation = MNIST(
                self.data_dir,
                train=True,
                download=False,
                transform=transforms.ToTensor(),
            )
            self.mean, self.std = self.calculate_mean_std(mnist_for_calculation)

    def setup(self, stage=None):
        # Update transform with calculated or default mean and std
        self.basic_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((self.mean,), (self.std,))]
        )

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(
                self.data_dir, train=True, transform=self.basic_transform
            )
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.basic_transform
            )

    def train_dataloader(self):
        if self.use_augmentation:
            return DataLoader(
                self.mnist_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.custom_collate_fn,
            )
        else:
            return DataLoader(
                self.mnist_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    @staticmethod
    def calculate_mean_std(dataset):
        loader = DataLoader(dataset, batch_size=1000, num_workers=1, shuffle=False)
        mean = 0.0
        std = 0.0
        total_images = 0

        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images += batch_samples

        mean /= total_images
        std /= total_images
        return mean.item(), std.item()

    # Custom collate function to apply augmentation multiple times
    def custom_collate_fn(self, batch):
        augmented_batch = []
        for image, label in batch:
            augmented_batch.append((image, label))

            for _ in range(self.augmentations_per_image):
                random_transform = random.choice(self.augmentation_transforms)
                augmented_image = random_transform(image)
                augmented_batch.append((augmented_image, label))

        images, labels = zip(*augmented_batch)
        return torch.stack(images), torch.tensor(labels)
