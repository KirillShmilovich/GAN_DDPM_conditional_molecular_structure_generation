import torch
from pytorch_lightning import LightningModule
from GAN_modules import (
    SimpleGenerator,
    SimpleDiscriminator,
    ConvDiscriminator,
    ConvGenerator,
)
from pathlib import Path


class WGANGP(LightningModule):
    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        n_samples: int,
        gen_network_type: str = "simple",
        dis_network_type: str = "simple",
        gen_hidden_dim: int = 256,
        dis_hidden_dim: int = 512,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        latent_dim: int = 100,
        lr: float = 5e-5,
        opt: str = "rmsprop",
        log_eval_metrics: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.gen_network_type.lower() == "simple":
            GenNetwork = SimpleGenerator
        elif self.hparams.gen_network_type.lower() == "conv":
            GenNetwork = ConvGenerator
        else:
            raise NotImplementedError

        if self.hparams.dis_network_type.lower() == "simple":
            DisNetwork = SimpleDiscriminator
        elif self.hparams.dis_network_type.lower() == "conv":
            DisNetwork = ConvDiscriminator
        else:
            raise NotImplementedError

        self.generator = GenNetwork(
            latent_dim=self.hparams.latent_dim + self.hparams.c_dim,
            output_dim=self.hparams.x_dim,
            hidden_dim=self.hparams.gen_hidden_dim,
        )
        self.discriminator = DisNetwork(
            output_dim=self.hparams.x_dim + self.hparams.c_dim,
            hidden_dim=self.hparams.dis_hidden_dim,
        )

        self.z_val = torch.randn(self.hparams.n_samples, self.hparams.latent_dim)

    def forward(self, z, c):
        x = self.generator(torch.cat((z, c), dim=-1))
        return x

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
        z = torch.randn(x.size(0), self.hparams.latent_dim, device=x.device)
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
            d_loss_was = torch.mean(fake_validity) - torch.mean(
                real_validity
            )  # Wasserstein loss
            gp = self.hparams.lambda_gp * gradient_penalty  # gradient penalty
            d_loss = d_loss_was + gp  # full loss

            self.log("d_loss_was", d_loss_was)
            self.log("gp", gp)
            self.log("d_loss", d_loss, prog_bar=True)

            return d_loss

    def configure_optimizers(self):
        opt = self.hparams.opt.lower()
        if opt == "rmsprop":
            opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=self.hparams.lr)
            opt_d = torch.optim.RMSprop(
                self.discriminator.parameters(), lr=self.hparams.lr
            )
        elif opt == "adam":
            opt_g = torch.optim.Adam(
                self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9)
            )
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9)
            )
        else:
            raise NotImplementedError

        return (
            {"optimizer": opt_g, "frequency": 1},
            {"optimizer": opt_d, "frequency": self.hparams.n_critic},
        )

    def on_train_epoch_end(self):
        if self.current_epoch % self.hparams.log_eval_metrics == 0:
            self.eval()
            c_val = self.trainer.datamodule.eval_data[1].to(self.device)
            z_val = self.z_val.to(self.device)
            with torch.no_grad():
                pos_val = self.forward(z_val, c_val).cpu().numpy()
            pos_val = self.trainer.datamodule.xyz_scaler.inverse_transform(pos_val)
            pos_val = pos_val.reshape(pos_val.shape[0], -1, 3)
            eval_metrics_dict, trj_val = self.trainer.datamodule.get_eval_metrics(
                pos_val
            )
            for k, v in eval_metrics_dict.items():
                self.log(k, v)

            trj_save_path = Path(self.trainer.logger.log_dir + "/trjs")
            trj_save_path.mkdir(parents=True, exist_ok=True)

            trj_val.save_xyz(str(trj_save_path / f"trj_epoch_{self.current_epoch}.xyz"))
