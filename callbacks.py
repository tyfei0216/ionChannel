from typing import Any, Mapping

import pytorch_lightning as L
import torch
import torchmetrics
from torch.optim import Optimizer

import models


class LambdaUpdate(L.Callback):
    def __init__(self, warmup=5000, check=3000):
        self.warmup = warmup
        self.check = check
        self.acc = torchmetrics.Accuracy(task="binary")

        self.cnt = 0
        self.outputs = []

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: models.IonBaseclf,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: torch.Any,
        batch_idx: int,
    ) -> None:
        self.outputs.append(pl_module.training_step_outputs[-1])
        # return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_before_optimizer_step(
        self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Optimizer
    ) -> None:
        if self.warmup > 0:
            self.warmup -= 1
        else:
            self.cnt += 1
        if self.cnt == self.check:
            self.cnt = 0
            scores = torch.concatenate([x["y"] for x in self.outputs])
            y = torch.concatenate([x["true_label"] for x in self.outputs])

            self.outputs.clear()
            acc = self.acc(scores, y)
            pl_module.updateLambda(acc)
        # return super().on_before_optimizer_step(trainer, pl_module, optimizer)


class DatasetAugmentationUpdate(L.Callback):
    def __init__(self):
        pass

    def on_before_optimizer_step(
        self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Optimizer
    ) -> None:

        for i in trainer.train_dataloader():
            i.step()
