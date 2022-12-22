from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, AUROC


class Module(LightningModule):
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()
        self.net = model

        self.save_hyperparameters(logger=False, ignore=["net"])

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_auc = AUROC(num_classes=2)
        self.val_auc = AUROC(num_classes=2)
        self.test_auc = AUROC(num_classes=2)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation aucuracy
        self.val_auc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        self.val_auc_best.reset()

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)

        loss = self.criterion(logits, y)
        return loss, logits, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, logits, targets = self.step(batch)

        self.train_loss(loss)
        self.train_auc(logits, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "logits": logits, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        self.log("train/auc", self.train_auc.compute(), prog_bar=True)
        self.train_auc.reset()
        self.train_loss.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logits, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_auc(logits, targets)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "logits": logits, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        auc = self.val_auc.compute()  # get current val auc
        self.val_auc_best(auc)  # update best so far val auc

        self.log("val/auc_best", self.val_auc_best.compute(), prog_bar=True)
        self.log("val/auc", self.val_auc.compute(), prog_bar=True)
        self.val_auc.reset()
        self.val_loss.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
