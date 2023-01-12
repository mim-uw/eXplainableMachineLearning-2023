import hydra
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.loggers import LightningLoggerBase


@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(config):
    print(config)
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    module: LightningModule = hydra.utils.instantiate(config.module)
    logger: LightningLoggerBase = hydra.utils.instantiate(config.logger)
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=[logger])
    trainer.fit(model=module, datamodule=datamodule)


if __name__ == "__main__":
    main()
