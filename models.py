import esm
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.autograd import Function

# from torchmetrics import Metric


class GradientR(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
        return grad_input, None


class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return GradientR.apply(x, self.alpha)


class ionclf(L.LightningModule):
    def __init__(
        self,
        esm_model,
        addadversial=True,
        lamb=0.1,
        lr=5e-4,
        step_lambda=True,
        step=1.5,
        max_lambda=6,
        thres=0.95,
        p=0.2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["esm_model"])

        self.num_layers = esm_model.num_layers
        self.embed_dim = esm_model.embed_dim
        self.attention_heads = esm_model.attention_heads
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.alphabet_size = len(self.alphabet)
        self.addadversial = addadversial

        self.lamb = lamb
        self.step_lambda = step_lambda
        self.step = step
        self.max_lambda = max_lambda
        self.thres = thres
        self.p = p
        self.update_epoch = False

        self.lr = lr

        self.esm_model = esm_model

        self.cls = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            # nn.Dropout(p=self.p),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.LayerNorm(self.embed_dim // 4),
            # nn.Dropout(p=self.p),
            nn.GELU(),
            nn.Linear(self.embed_dim // 4, 1),
        )

        self.dis = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.LayerNorm(self.embed_dim // 4),
            nn.GELU(),
            nn.Linear(self.embed_dim // 4, 1),
        )

        self.reverse = GradientReversal(1)

        # if unfix is None:
        #     self.fixParameters()
        # else:
        #     self.fixParameters(unfix)

        self.acc = torchmetrics.Accuracy(task="binary")

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        representations = self.esm_model(x, repr_layers=[self.num_layers])

        x = representations["representations"][self.num_layers][:, 0]
        x1 = self.reverse(x)
        pre = self.cls(x)
        pre = F.sigmoid(pre)

        y = self.dis(x1)
        y = F.sigmoid(y)

        return pre, y

    def _common_training_step(self, batch):
        X1, y, X2 = batch
        y_pre, dis_pre_x1 = self(X1)
        _y, dis_pre_x2 = self(X2)

        loss1 = F.binary_cross_entropy(y_pre.squeeze(), y.float())
        loss2 = F.binary_cross_entropy(
            dis_pre_x1, torch.zeros_like(dis_pre_x1)
        ) + F.binary_cross_entropy(dis_pre_x2, torch.ones_like(dis_pre_x1))

        if self.addadversial:
            loss = loss1 + loss2 * self.lamb
        else:
            loss = loss1

        return loss, loss1, loss2, y_pre, y

    def training_step(self, batch, batch_idx):

        loss, loss1, loss2, y_pre, y = self._common_training_step(batch)

        acc = self.acc(y_pre.squeeze(), y)

        self.log_dict(
            {"predict loss": loss1.item(), "adversial loss": loss2.item(), "acc": acc},
            prog_bar=True,
            on_step=True,
        )
        self.training_step_outputs.append(
            {
                "loss": loss.detach().cpu(),
                "y": y_pre.detach().squeeze().cpu(),
                "true_label": y.cpu(),
            }
        )

        return loss

    def _common_epoch_end(self, outputs):

        loss = torch.stack([x["loss"] for x in outputs]).mean()
        scores = torch.concatenate([x["y"] for x in outputs])
        y = torch.concatenate([x["true_label"] for x in outputs])

        outputs.clear()
        return loss, self.acc(scores, y)

    def on_training_epoch_end(self):

        loss, acc = self._common_epoch_end(self.training_step_outputs)

        # print("finish training epoch, loss %f, acc %f"%(loss, acc))
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": acc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.update_epoch = True

    def validation_step(self, batch, batch_idx):

        loss, loss1, loss2, y_pre, y = self._common_training_step(batch)

        acc = self.acc(y_pre.squeeze(), y)

        self.log_dict(
            {"predict loss": loss1.item(), "adversial loss": loss2.item(), "acc": acc}
        )

        self.validation_step_outputs.append(
            {"loss": loss.cpu(), "y": y_pre.squeeze().cpu(), "true_label": y.cpu()}
        )

        return loss

    def on_validation_epoch_end(self):
        loss, acc = self._common_epoch_end(self.validation_step_outputs)
        # print("finish validating, loss %f, acc %f"%(loss, acc))
        self.log_dict(
            {
                "validate_loss": loss,
                "validate_acc": acc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        if self.step_lambda and self.update_epoch:
            self.update_epoch = False
            if self.thres[1] < acc:
                self.lamb = min(self.lamb * self.step, self.max_lambda)
            if self.thres[0] > acc:
                self.lamb = max(self.lamb / self.step, 0.1)

    def test_step(self, batch, batch_idx):
        x = batch
        y_pre, _ = self(x)
        return y_pre

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list):
            if len(batch) == 3:
                X1, y, X2 = batch
            elif len(batch) == 2:
                X1, y = batch
            else:
                raise ValueError
        else:
            X1 = batch
            y = None
        # print(X1, len)
        pre, _ = self(X1)
        pre = pre.squeeze()
        if y is None:
            return pre
        else:
            return pre, y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )

        return optimizer


def fixParameters(esm_model, unfix=["9", "10", "11"]):
    for i, j in esm_model.named_parameters():
        flag = 1
        for k in unfix:
            if k in i:
                flag = 0

        if flag == 1:
            j.requires_grad = False
        else:
            j.requires_grad = True

    return esm_model


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim) * std_dev)
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def _set_submodule(submodule, module_path, new_module):
    tokens = module_path.split(".")
    for token in tokens[:-1]:
        submodule = getattr(submodule, token)
    setattr(submodule, tokens[-1], new_module)


def addlora(esm_model, layers, ranks, alphas):
    # if layers is None:
    #     layers = [str(i) for i in range(12)]
    for i, j in esm_model.named_modules():
        if isinstance(j, nn.Linear):
            # print(i)
            # res = [False]
            # res.extend([t in i for t in layers])
            # res = reduce(lambda x, y: x or y, res)
            for layer, rank, alpha in zip(layers, ranks, alphas):
                if str(layer) in i:
                    _set_submodule(
                        esm_model,
                        i,
                        LinearWithLoRA(j, rank, alpha),
                    )
    return esm_model
