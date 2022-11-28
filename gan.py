import torch
from torch import nn
from pytorch_lightning import LightningModule


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.SiLU())
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 200),
            *block(200, 200),
            *block(200, 200),
            nn.Linear(200, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, output_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(output_dim, 200),
            nn.SiLU(),
            nn.Linear(200, 200),
            nn.SiLU(),
            nn.Linear(200, 200),
            nn.SiLU(),
            nn.Linear(200, 1),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity


class WGANGP(LightningModule):
    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        n_samples: int,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        latent_dim: int = 50,
        lr: float = 5e-5,
        **kwargs,
    ):
        super().__init__()
        self.out_dim = x_dim
        self.save_hyperparameters()

        # networks
        self.generator = Generator(
            latent_dim=self.hparams.latent_dim + c_dim, output_dim=self.out_dim
        )
        self.discriminator = Discriminator(output_dim=self.out_dim + c_dim)

        self.z_val = torch.randn(n_samples, self.hparams.latent_dim)

    def forward(self, z, c):
        return self.generator(torch.cat((z, c), dim=-1))

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples

        alpha = torch.rand((real_samples.size(0), 1), device=real_samples.device)

        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),  # fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, c = batch

        # sample noise
        z = torch.randn(x.size(0), self.latent_dim, device=x.device)
        z = z.type_as(x)

        # train generator
        if optimizer_idx == 0:

            fake = self(z, c)
            fake = torch.cat((fake, c), dim=-1)
            g_loss = -torch.mean(self.discriminator(fake))

            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        elif optimizer_idx == 1:
            fake = self(z, c)

            # Real images
            real = torch.cat((x, c), dim=-1)
            real_validity = self.discriminator(real)
            # Fake images
            fake = torch.cat((fake, c), dim=-1)
            fake_validity = self.discriminator(fake)
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(real.data, fake.data)
            # Adversarial loss
            d_loss = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                + self.hparams.lambda_gp * gradient_penalty
            )

            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=self.hparams.lr)
        opt_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.hparams.lr)

        return (
            {"optimizer": opt_g, "frequency": 1},
            {"optimizer": opt_d, "frequency": self.hparams.n_critic},
        )

    # def on_epoch_end(self):
    #    if self.current_epoch % 100 == 0:
    #        c_val = self.trainer.datamodule.data[1].to(self.device)
    #        z_val = self.z_val.to(self.device)
    #        with torch.no_grad():
    #            pos_val = self.forward(z_val, c_val).cpu().numpy()
    #        pos_val = scaler.inverse_transform(pos_val)
    #        trj_val = md.Trajectory(
    #            pos_val.reshape(pos_val.shape[0], -1, 3), topology=trj.top
    #        )
    #        trj_val = trj_val.atom_slice(heavy_idxs)
    #        pdists = np.concatenate([pdist(xyz)[None] for xyz in trj_val.xyz])
    #        projected_data = tica_model.transform(pdists)

    #        pyemma.plots.plot_free_energy(
    #            *projected_data[:, :2].T, levels=np.linspace(0, 10, 15)
    #        )
    #        plt.xlim(-3.3430330710483647, 6.949510139450872)
    #        plt.ylim(-5.115580727712025, 1.74860872294618)
    #        plt.title(f"Epoch #{self.current_epoch}")
    #        plt.show()
