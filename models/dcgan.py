import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import make_grid

from models.base.discriminator import Discriminator
from models.base.generator import Generator


class DCGAN(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.automatic_optimization = False
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.validation_z = torch.randn(8, config['latent_dim'], 1, 1)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch

        # Sample noise
        z = torch.randn(imgs.shape[0], self.config['latent_dim'], 1, 1)
        z = z.type_as(imgs)

        # Access optimizers
        opt_g, opt_d = self.optimizers()

        # Train generator
        self.generated_imgs = self(z)
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

        # Train discriminator
        real_loss = self.adversarial_loss(
            self.discriminator(imgs), torch.ones((imgs.size(0), 1), device=self.device)
        )
        fake_loss = self.adversarial_loss(
            self.discriminator(self(z).detach()),
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

    def configure_optimizers(self):
        g_lr = self.config['g_lr']
        d_lr = self.config['d_lr']
        b1 = self.config['b1']
        b2 = self.config['b2']

        opt_g = Adam(self.generator.parameters(), lr=g_lr, betas=(b1, b2))
        opt_d = Adam(self.discriminator.parameters(), lr=d_lr, betas=(b1, b2))
        return opt_g, opt_d

    def validation_step(self, batch, batch_idx):
        z = torch.randn(8, self.config['latent_dim'], 1, 1)
        z = z.type_as(batch[0])
        generated_imgs = self(z)
        return generated_imgs

    def on_validation_epoch_end(self):
        z = torch.randn(8, self.config['latent_dim'], 1, 1)
        z = z.type_as(next(self.generator.parameters()))
        sample_imgs = self(z)
        grid = make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)