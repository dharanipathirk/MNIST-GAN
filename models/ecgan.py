import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import make_grid

# import torchvision.utils as vutils


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = nn.Sequential(
            nn.Conv2d(config['channels'], 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, config['latent_dim']),
        )

    def forward(self, img):
        return self.model(img)


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
        return (
            out.view(-1, 1)
            if self.config['use_wasserstein']
            else torch.sigmoid(out).view(-1, 1)
        )


class EncoderCGAN(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.automatic_optimization = False
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.encoder = Encoder(config)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch

        # to view real images in tensorboard (uncomment import torchvision.utils as vutils)
        # grid = vutils.make_grid(imgs, normalize=True)
        # self.logger.experiment.add_image('real_images', grid, self.current_epoch)

        #  encode real images to latent space
        z_real = self.encoder(imgs)
        z_real = z_real.view(imgs.size(0), self.config['latent_dim'], 1, 1)

        # Access optimizers
        opt_g, opt_d = self.optimizers()

        if self.config['use_wasserstein']:
            # WGAN loss computation
            if self.global_step % self.config['n_critic_steps'] != 0:
                # Train generator
                self.toggle_optimizer(opt_g)
                self.generated_imgs = self(z_real)
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
                d_loss = -torch.mean(self.discriminator(imgs)) + torch.mean(
                    self.discriminator(self(z_real))
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
            self.generated_imgs = self(z_real)
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
                self.discriminator(self(z_real).detach()),
                torch.zeros((imgs.size(0), 1), device=self.device),
            )
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

    def configure_optimizers(self):
        g_lr = self.config['g_lr']
        d_lr = self.config['d_lr']
        b1 = self.config['b1']
        b2 = self.config['b2']

        opt_g = Adam(self.generator.parameters(), lr=g_lr, betas=(b1, b2))
        opt_d = Adam(self.discriminator.parameters(), lr=d_lr, betas=(b1, b2))
        return opt_g, opt_d

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        # Encode real images to latent space
        z_encoded = self.encoder(imgs)
        z_encoded = z_encoded.view(imgs.size(0), self.config['latent_dim'], 1, 1)
        generated_imgs = self.generator(z_encoded)
        return generated_imgs

    def on_validation_epoch_end(self):
        # Select a batch of real images
        val_loader = self.trainer.datamodule.val_dataloader()

        real_imgs, _ = next(iter(val_loader))
        real_imgs = real_imgs[:8]
        real_imgs = real_imgs.type_as(next(self.generator.parameters()))

        z_encoded = self.encoder(real_imgs)
        z_encoded = z_encoded.view(8, self.config['latent_dim'], 1, 1)
        sample_imgs = self.generator(z_encoded)

        grid = make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

    def on_after_backward(self):
        if self.config['use_wasserstein']:
            with torch.no_grad():
                for p in self.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)
