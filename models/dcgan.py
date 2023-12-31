import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import make_grid

# import torchvision.utils as vutils


# simple convolutional generator
class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = nn.Sequential(
            nn.ConvTranspose2d(config['latent_dim'], 1024, 3, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 3, 2, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, config['channels'], 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


# simple convolutional discriminator
class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = nn.Sequential(
            nn.Conv2d(config['channels'], 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 3, 2, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 3, 1, 0, bias=False),
        )

    def forward(self, img):
        out = self.model(img)
        # apply sigmoid when not using W loss
        return (
            out.view(-1, 1)
            if self.config['use_wasserstein']
            else torch.sigmoid(out).view(-1, 1)
        )


# deep convolutional GAN
class DCGAN(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Disabling automatic optimization so that we can use multiple optimizers
        self.automatic_optimization = False
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

    def forward(self, z):
        return self.generator(z)

    # BCE loss for standard training
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch

        # to view real images in tensorboard (uncomment import torchvision.utils as vutils)
        # grid = vutils.make_grid(imgs, normalize=True)
        # self.logger.experiment.add_image('real_images', grid, self.current_epoch)

        # Sample noise
        z = torch.randn(imgs.shape[0], self.config['latent_dim'], 1, 1)
        z = z.type_as(imgs)

        # Access optimizers
        opt_g, opt_d = self.optimizers()

        if self.config['use_wasserstein']:
            # Training with W loss
            # training frequency of generator
            if self.global_step % self.config['n_critic_steps'] != 0:
                # Train generator
                self.toggle_optimizer(opt_g)
                self.generated_imgs = self(z)
                # generator W loss
                g_loss = -torch.mean(self.discriminator(self.generated_imgs))
                opt_g.zero_grad()
                self.manual_backward(g_loss)
                opt_g.step()
                self.log(
                    'generator_loss',
                    g_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                self.untoggle_optimizer(opt_g)
            else:
                # Train critic
                self.toggle_optimizer(opt_d)
                # critic W loss
                d_loss = -torch.mean(self.discriminator(imgs)) + torch.mean(
                    self.discriminator(self(z))
                )
                opt_d.zero_grad()
                self.manual_backward(d_loss)
                opt_d.step()
                self.log(
                    'critic_loss',
                    d_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                self.untoggle_optimizer(opt_d)

        else:
            # Train generator
            self.toggle_optimizer(opt_g)
            self.generated_imgs = self(z)
            # generator BCE loss
            g_loss = self.adversarial_loss(
                self.discriminator(self.generated_imgs),
                torch.ones((imgs.size(0), 1), device=self.device),
            )
            opt_g.zero_grad()
            self.manual_backward(g_loss)
            opt_g.step()
            self.log(
                'generator_loss',
                g_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.untoggle_optimizer(opt_g)

            # Train discriminator
            self.toggle_optimizer(opt_d)
            real_loss = self.adversarial_loss(
                self.discriminator(imgs),
                torch.ones((imgs.size(0), 1), device=self.device),
            )
            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()),
                torch.zeros((imgs.size(0), 1), device=self.device),
            )
            # discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()
            self.log(
                'discriminator_loss',
                d_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.untoggle_optimizer(opt_d)

    # optimizers for generator and discriminator
    def configure_optimizers(self):
        g_lr = self.config['g_lr']
        d_lr = self.config['d_lr']
        b1 = self.config['b1']
        b2 = self.config['b2']

        opt_g = Adam(self.generator.parameters(), lr=g_lr, betas=(b1, b2))
        opt_d = Adam(self.discriminator.parameters(), lr=d_lr, betas=(b1, b2))
        return opt_g, opt_d

    def validation_step(self, batch, batch_idx):
        # Generating a fixed number of images with random noise
        z = torch.randn(8, self.config['latent_dim'], 1, 1)
        z = z.type_as(batch[0])
        generated_imgs = self(z)
        return generated_imgs

    def on_validation_epoch_end(self):
        # Generating and logging a grid of images at the end of each val epoch
        z = torch.randn(8, self.config['latent_dim'], 1, 1)
        z = z.type_as(next(self.generator.parameters()))
        sample_imgs = self(z)
        grid = make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

    def on_after_backward(self):
        # weight clipping when using W loss
        if self.config['use_wasserstein']:
            with torch.no_grad():
                for p in self.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)
