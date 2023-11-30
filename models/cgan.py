import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import make_grid

# import torchvision.utils as vutils


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.label_emb = nn.Embedding(config['n_classes'], config['n_classes'])

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                config['latent_dim'] + config['n_classes'], 1024, 3, 1, 0, bias=False
            ),
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

    def forward(self, z, labels):
        # Embed labels and concatenate with noise vector
        labels = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        z = torch.cat([z, labels], dim=1)
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.label_emb = nn.Embedding(config['n_classes'], config['n_classes'])

        self.model = nn.Sequential(
            nn.Conv2d(
                config['channels'] + config['n_classes'], 256, 4, 2, 1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 3, 2, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 3, 1, 0, bias=False),
        )

    def forward(self, img, labels):
        labels = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        labels = labels.expand(-1, -1, img.size(2), img.size(3))
        img = torch.cat([img, labels], dim=1)
        out = self.model(img)
        return (
            out.view(-1, 1)
            if self.config['use_wasserstein']
            else torch.sigmoid(out).view(-1, 1)
        )


class ConditionalGAN(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.automatic_optimization = False
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

    def forward(self, z, labels):
        return self.generator(z, labels)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch

        # to view real images in tensorboard (uncomment import torchvision.utils as vutils)
        # grid = vutils.make_grid(imgs, normalize=True)
        # self.logger.experiment.add_image('real_images', grid, self.current_epoch)

        # Sample noise
        z = torch.randn(imgs.shape[0], self.config['latent_dim'], 1, 1)
        z = z.type_as(imgs)

        # Access optimizers
        opt_g, opt_d = self.optimizers()

        if self.config['use_wasserstein']:
            # WGAN loss computation
            if self.global_step % self.config['n_critic_steps'] != 0:
                # Train generator
                self.toggle_optimizer(opt_g)
                self.generated_imgs = self(z, labels)
                g_loss = -torch.mean(self.discriminator(self.generated_imgs, labels))
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
                d_loss = -torch.mean(self.discriminator(imgs, labels)) + torch.mean(
                    self.discriminator(self(z, labels), labels)
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
            self.generated_imgs = self(z, labels)
            g_loss = self.adversarial_loss(
                self.discriminator(self.generated_imgs, labels),
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
                self.discriminator(imgs, labels),
                torch.ones((imgs.size(0), 1), device=self.device),
            )
            fake_loss = self.adversarial_loss(
                self.discriminator(self(z, labels).detach(), labels),
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
        z = torch.randn(8, self.config['latent_dim'], 1, 1, device=self.device)
        val_labels = torch.randint(
            0, self.config['n_classes'], (8,), device=self.device
        )
        z = z.type_as(batch[0])
        generated_imgs = self(z, val_labels)
        return generated_imgs

    def on_validation_epoch_end(self):
        z = torch.randn(8, self.config['latent_dim'], 1, 1, device=self.device)
        val_labels = torch.randint(
            0, self.config['n_classes'], (8,), device=self.device
        )
        z = z.type_as(next(self.generator.parameters()))
        sample_imgs = self(z, val_labels)
        grid = make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

    def on_after_backward(self):
        if self.config['use_wasserstein']:
            with torch.no_grad():
                for p in self.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)
