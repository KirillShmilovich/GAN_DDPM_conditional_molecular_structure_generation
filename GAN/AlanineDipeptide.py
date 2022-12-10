from GAN_datamodule import AlanineDipeptide
from GAN_model import WGANGP
from pytorch_lightning.cli import LightningCLI


def cli_main():
    class MyCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.x_dim", "model.x_dim", apply_on="instantiate")
            parser.link_arguments("data.c_dim", "model.c_dim", apply_on="instantiate")
            parser.link_arguments(
                "data.n_samples", "model.n_samples", apply_on="instantiate"
            )

    cli = MyCLI(
        WGANGP,
        AlanineDipeptide,
        seed_everything_default=4321,
        run=False,  # used to de-activate automatic fitting.
        trainer_defaults={
            "accelerator": "gpu",
            "max_epochs": 100000,
            "enable_progress_bar": False,
            # "limit_val_batches": 0,
            # "num_sanity_val_steps": 0,
            # "check_val_every_n_epoch": 500,
            # "logger": TensorBoardLogger(save_dir="./",name="PentaPeptideBackBoneLogs"),
        },
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    # predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)
    # print(predictions[0])


if __name__ == "__main__":
    cli_main()
