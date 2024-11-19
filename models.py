import esm
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.autograd import Function
from torch.optim.optimizer import Optimizer

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


class Linearcls(nn.Module):
    """simple linear classifier

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self, input_dim=1536, take_embed="first", dropout=-1, p0=None, output_dim=1
    ):
        super().__init__()

        assert take_embed in ["first", "mean", "max"]
        self.embed_dim = input_dim
        self.dropout = dropout
        self.take_embed = take_embed
        self.output_dim = output_dim

        self.l1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        self.l2 = nn.Linear(self.embed_dim // 2, self.embed_dim // 4)
        self.l3 = nn.Linear(self.embed_dim // 4, output_dim)
        self.ln1 = nn.LayerNorm(self.embed_dim // 2)
        self.ln2 = nn.LayerNorm(self.embed_dim // 4)
        if p0 is None:
            self.p0 = None
        else:
            self.p0 = nn.Dropout(p0)
        if dropout > 0 and dropout < 1:
            self.dropout1 = nn.Dropout(p=self.dropout)
            self.dropout2 = nn.Dropout(p=self.dropout)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x: torch.Tensor):

        if self.take_embed == "first":
            x = x[:, 0]
        elif self.take_embed == "mean":
            x = torch.mean(x, dim=1)
        elif self.take_embed == "max":
            x = x.transpose(1, 2)
            x = F.adaptive_max_pool1d(x, 1)

        if self.p0 is not None:
            x = self.p0(x)

        x = self.l1(x)
        x = self.ln1(x)
        if self.dropout1 is not None:
            x = self.dropout1(x)
        x = F.gelu(x)
        x = self.l2(x)
        x = self.ln2(x)
        if self.dropout2 is not None:
            x = self.dropout2(x)
        x = F.gelu(x)
        x = self.l3(x)
        # print("lin", x.shape)
        if self.output_dim == 1:
            return x
        else:
            return x[:, 0], x[:, 1:]


class CNNcls(nn.Module):
    """CNN based classifier for esm2/3 extracted features

    Args:
        nn (_type_): _description_
    """

    def __init__(self, input_dim=1536, pool="max", output_dim=1):
        super().__init__()

        self.embed_dim = input_dim
        self.output_dim = output_dim
        assert pool in ["max", "avg"]
        self.pool = pool

        self.cl1 = nn.Conv1d(self.embed_dim, self.embed_dim // 2, kernel_size=5)
        self.cl2 = nn.Conv1d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=5)
        self.cl3 = nn.Conv1d(self.embed_dim // 4, self.embed_dim // 8, kernel_size=5)

        self.ll1 = nn.Linear(self.embed_dim // 8, self.embed_dim // 16)
        self.ll2 = nn.Linear(self.embed_dim // 16, output_dim)

    def forward(self, x: torch.Tensor):

        x = x.transpose(1, 2)

        x = self.cl1(x)
        x = F.gelu(x)
        x = self.cl2(x)
        x = F.gelu(x)
        x = self.cl3(x)
        x = F.gelu(x)

        if self.pool == "max":
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        elif self.pool == "avg":
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        else:
            raise NotImplementedError
        # print(x.shape)
        x = self.ll1(x)
        x = F.gelu(x)
        x = self.ll2(x)
        # print("cnn", x.shape)
        if self.output_dim == 1:
            return x
        else:
            return x[:, 0], x[:, 1:]


class IonBaseclf(L.LightningModule):
    """
    base class for the ion channel classifier, handles training process and
    domain adaptation

    Args:
        L (_type_): _description_

    Raises:
        NotImplementedError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    def __init__(
        self,
        addadversial=True,
        lamb=0.1,
        lr=5e-4,
        lr_backbone=None,
        step_lambda=True,
        step=1.5,
        max_lambda=6,
        thres=0.95,
        weight_decay=0.005,
        additional_label_weights=[],
    ):
        super().__init__()
        self.addadversial = addadversial
        if not isinstance(additional_label_weights, torch.Tensor):
            additional_label_weights = torch.tensor(
                additional_label_weights, requires_grad=False
            )
        # self.additional_label_weights = additional_label_weights
        additional_label_weights = torch.nn.Parameter(additional_label_weights)
        self.register_parameter("additional_label_weights", additional_label_weights)
        self.additional_label_weights.requires_grad = False

        self.lamb = lamb
        self.step_lambda = step_lambda
        self.step = step
        self.max_lambda = max_lambda
        self.thres = thres
        self.update_epoch = False
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_backbone = lr_backbone

        self.acc = torchmetrics.Accuracy(task="binary")

        self.last_train_step = 0

        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.load_freeze = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _common_training_step(self, batch):
        X1, y, X2 = batch
        y_pre, dis_pre_x1 = self(X1)
        _y, dis_pre_x2 = self(X2)
        # print(y_pre, y)
        if len(y.size()) == 2:
            y = y.squeeze(0)

        if len(self.additional_label_weights) == 0:
            loss1 = F.binary_cross_entropy(y_pre.squeeze(-1), y.float())
        else:
            if y.dim() == 1:
                y1 = y[[0]].unsqueeze(0)
                y2 = y[1:].unsqueeze(0)
            elif y.dim() == 2:
                y1 = y[:, [0]]
                y2 = y[:, 1:]

            y_pre1, y_pre2 = y_pre
            if y_pre1.dim() == 1:
                y_pre1 = y_pre1.unsqueeze(0)
            if y_pre2.dim() == 1:
                y_pre2 = y_pre2.unsqueeze(0)

            loss1 = F.binary_cross_entropy(y_pre1, y1.float())
            # print(y_pre1.shape, self.additional_label_weights[None, :].shape)
            q = self.additional_label_weights[None, :].repeat(y_pre1.shape[0], 1)
            q[y2 < 0] = 0.0
            q.require_grad = False
            y2[y2 < 0] = 0
            y2[y2 > 1] = 1
            # print(y_pre2, y2, q)
            loss1_a = F.binary_cross_entropy(
                y_pre2,
                y2.float(),
                weight=self.additional_label_weights,
            )
            # print("finish loss")
            y_pre = y_pre1  # .squeeze(-1)
            y = y1.squeeze(-1)

        loss2 = F.binary_cross_entropy(
            dis_pre_x1, torch.zeros_like(dis_pre_x1)
        ) + F.binary_cross_entropy(dis_pre_x2, torch.ones_like(dis_pre_x1))

        if self.addadversial:
            loss = loss1 + loss2 * self.lamb
        else:
            loss = loss1

        if len(self.additional_label_weights) > 0:
            loss += loss1_a
            return loss, (loss1, loss1_a), loss2, y_pre, y
        else:
            return loss, loss1, loss2, y_pre, y

    def training_step(self, batch, batch_idx):

        loss, loss1, loss2, y_pre, y = self._common_training_step(batch)

        # acc = self.acc(y_pre.squeeze(-1), y)
        if len(self.additional_label_weights) == 0:
            self.training_step_outputs.append(
                {
                    "loss": loss.detach().cpu(),
                    "loss1": loss1.detach().cpu(),
                    "loss2": loss2.detach().cpu(),
                    "y": y_pre.detach().squeeze(-1).cpu(),
                    "true_label": y.cpu(),
                }
            )
        else:
            self.training_step_outputs.append(
                {
                    "loss": loss.detach().cpu(),
                    "loss1": loss1[0].detach().cpu(),
                    "loss1_a": loss1[1].detach().cpu(),
                    "loss2": loss2.detach().cpu(),
                    "y": y_pre.detach().squeeze(-1).cpu(),
                    "true_label": y.cpu(),
                }
            )
        return loss

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:

        loss = torch.stack(
            [x["loss"] for x in self.training_step_outputs[self.last_train_step :]]
        ).mean()
        loss1 = torch.stack(
            [x["loss1"] for x in self.training_step_outputs[self.last_train_step :]]
        ).mean()
        if len(self.additional_label_weights) > 0:
            loss1_a = torch.stack(
                [
                    x["loss1_a"]
                    for x in self.training_step_outputs[self.last_train_step :]
                ]
            ).mean()
        loss2 = torch.stack(
            [x["loss2"] for x in self.training_step_outputs[self.last_train_step :]]
        ).mean()
        scores = torch.concatenate(
            [x["y"] for x in self.training_step_outputs[self.last_train_step :]]
        )
        y = torch.concatenate(
            [
                x["true_label"]
                for x in self.training_step_outputs[self.last_train_step :]
            ]
        )
        acc = self.acc(scores, y)

        self.last_train_step = len(self.training_step_outputs)

        if len(self.additional_label_weights) == 0:
            self.log_dict(
                {
                    "train predict loss": loss1,
                    "train adversial loss": loss2,
                    "train loss": loss,
                    "train acc": acc,
                },
                logger=True,
                prog_bar=True,
            )
        else:
            self.log_dict(
                {
                    "train predict loss": loss1,
                    "train additional loss": loss1_a,
                    "train adversial loss": loss2,
                    "train loss": loss,
                    "train acc": acc,
                },
                logger=True,
                prog_bar=True,
            )

    def _common_epoch_end(self, outputs):

        if len(outputs) == 0:
            return 0, 0
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        scores = torch.concatenate([x["y"] for x in outputs])
        y = torch.concatenate([x["true_label"] for x in outputs])

        outputs.clear()
        return loss, self.acc(scores, y)

    def on_training_epoch_end(self):

        loss, acc = self._common_epoch_end(self.training_step_outputs)

        print("finish training epoch, loss %f, acc %f" % (loss, acc))
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": acc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.last_train_step = 0

        self.update_epoch = True

    def validation_step(self, batch, batch_idx):

        loss, loss1, loss2, y_pre, y = self._common_training_step(batch)

        acc = self.acc(y_pre.squeeze(-1), y)

        # self.log_dict(
        #     {"predict loss": loss1.item(), "adversial loss": loss2.item(), "acc": acc},
        #     logger=True,
        # )

        self.validation_step_outputs.append(
            {"loss": loss.cpu(), "y": y_pre.squeeze(-1).cpu(), "true_label": y.cpu()}
        )

        return loss

    def on_validation_epoch_end(self):
        loss, acc = self._common_epoch_end(self.validation_step_outputs)
        print("finish validating, loss %f, acc %f" % (loss, acc))
        self.log_dict(
            {
                "validate_loss": loss,
                "validate_acc": acc,
            },
            on_step=False,
            logger=True,
            on_epoch=True,
            prog_bar=False,
        )

        # not updated after batch
        # if self.step_lambda and self.update_epoch:
        #     self.update_epoch = False
        #     if self.thres[1] < acc:
        #         self.lamb = min(self.lamb * self.step, self.max_lambda)
        #     if self.thres[0] > acc:
        #         self.lamb = max(self.lamb / self.step, 0.1)

    def updateLambda(self, acc):
        print("updating self lambda")
        if self.step_lambda:
            if self.thres[1] < acc:
                self.lamb = min(self.lamb * self.step, self.max_lambda)
            if self.thres[0] > acc:
                self.lamb = max(self.lamb / self.step, 0.1)

        print("lambda", self.lamb)

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
        print("get training optimizer")
        if self.lr_backbone is not None:
            l1 = []
            l2 = []
            for i, j in self.named_parameters():
                if j.requires_grad:
                    if "esm" in i:
                        l1.append(j)
                    else:
                        l2.append(j)

            param_dicts = [
                {
                    "params": l1,
                    "lr": self.lr_backbone,
                },
                {
                    "params": l2,
                    "lr": self.lr,
                },
            ]
            return torch.optim.Adam(param_dicts, weight_decay=self.weight_decay)

        if self.load_freeze is None:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            l1 = list(self.dis.parameters())
            l2 = list(self.clf.parameters())
            l1.extend(l2)
            optimizer = torch.optim.Adam(
                l1,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            for need in self.load_freeze:
                params = []
                for i, j in self.named_parameters():
                    flag = 1
                    for k in need:
                        if k not in i:
                            flag = 0
                            break
                    if flag == 1:
                        params.append(j)
                optimizer.add_param_group({"params": params, "lr": self.lr})

        return optimizer


class IonclfESM3(IonBaseclf):
    def __init__(
        self,
        esm_model,
        embed_dim=1536,
        addadversial=True,
        lamb=0.1,
        lr=5e-4,
        lr_backbone=None,
        step_lambda=True,
        step=1.5,
        max_lambda=6,
        thres=0.95,
        p=0.2,
        weight_decay=0.005,
        addition_label_weights=[],
        clf="linear",
        clf_params={},
        dis="linear",
        dis_params={},
    ) -> None:
        super().__init__(
            addadversial=addadversial,
            lamb=lamb,
            lr=lr,
            lr_backbone=lr_backbone,
            step_lambda=step_lambda,
            step=step,
            max_lambda=max_lambda,
            thres=thres,
            weight_decay=weight_decay,
            additional_label_weights=addition_label_weights,
        )

        self.embed_dim = embed_dim
        self.addadversial = addadversial
        self.p = p
        self.reverse = GradientReversal(1)

        self.esm_model = esm_model

        assert clf in ["linear", "cnn"]
        assert dis in ["linear", "cnn"]
        if "output_dim" not in clf_params:
            clf_params["output_dim"] = 1 + len(self.additional_label_weights)
        if clf == "cnn":
            self.clf = CNNcls(**clf_params)
        else:
            self.clf = Linearcls(**clf_params)
        if dis == "cnn":
            self.dis = CNNcls(**dis_params)
        else:
            self.dis = Linearcls(**dis_params)

    def forward(self, input_dict):

        for i in ["seq_t", "structure_t", "ss8_t", "sasa_t"]:
            if i not in input_dict:
                input_dict[i] = None
            else:
                if len(input_dict[i].size()) == 1:
                    input_dict[i] = input_dict[i].unsqueeze(0)

        representations = self.esm_model(
            sequence_tokens=input_dict["seq_t"],
            structure_tokens=input_dict["structure_t"],
            ss8_tokens=input_dict["ss8_t"],
            sasa_tokens=input_dict["sasa_t"],
        )

        x = representations.embeddings  # [:, 0]
        x1 = self.reverse(x)
        x1 = x
        pre = self.clf(x)
        if len(self.additional_label_weights) == 0:
            pre = F.sigmoid(pre)
        else:
            pre = (F.sigmoid(pre[0]), F.sigmoid(pre[1]))

        y = self.dis(x1)
        y = F.sigmoid(y)

        return pre, y

    def on_save_checkpoint(self, checkpoint):
        backbones = []
        for i, j in self.named_parameters():
            if "esm" in i and not j.requires_grad:
                backbones.append(i)
        for i in backbones:
            del checkpoint["state_dict"][i]


class IonclfESM2(IonBaseclf):
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
        weight_decay=0.005,
        p=0.2,
    ) -> None:
        super().__init__(
            addadversial=addadversial,
            lamb=lamb,
            lr=lr,
            step_lambda=step_lambda,
            step=step,
            max_lambda=max_lambda,
            thres=thres,
            weight_decay=weight_decay,
        )

        self.save_hyperparameters(ignore=["esm_model"])

        self.num_layers = esm_model.num_layers
        self.embed_dim = esm_model.embed_dim
        self.attention_heads = esm_model.attention_heads
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.alphabet_size = len(self.alphabet)

        self.p = p

        self.esm_model = esm_model

        self.cls = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.Dropout(p=self.p),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.LayerNorm(self.embed_dim // 4),
            nn.Dropout(p=self.p),
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

    def forward(self, x):
        representations = self.esm_model(x, repr_layers=[self.num_layers])

        x = representations["representations"][self.num_layers][:, 0]
        x1 = self.reverse(x)
        pre = self.cls(x)
        pre = F.sigmoid(pre)

        y = self.dis(x1)
        y = F.sigmoid(y)

        return pre, y


class SelfAttention(nn.Module):
    def __init__(self, channels, n_head):
        super(SelfAttention, self).__init__()
        self.channels = channels
        # self.size = size
        self.n_head = n_head
        assert channels % n_head == 0
        self.mha = nn.MultiheadAttention(channels, n_head, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # x = x.swapaxes(1, 2)
        # batch, length, channel = x.shape
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value  # .swapaxes(2, 1)


class SeqTransformer(nn.Module):
    def __init__(self, embed_dim=32, pos_dim=32):
        super().__init__()
        self.in_dim = embed_dim
        self.step_dim = embed_dim * 2
        self.pos_dim = pos_dim

        pe = self.inipos(self.pos_dim)
        pe.requires_grad = False
        self.register_buffer("pe", pe)

        self.embed = nn.Embedding(35, self.in_dim)

        self.inc = nn.Linear(self.in_dim + self.pos_dim, self.step_dim)
        self.sa1 = SelfAttention(self.step_dim, 4)
        self.sa2 = SelfAttention(self.step_dim, 4)
        self.sa3 = SelfAttention(self.step_dim, 4)
        self.sa4 = SelfAttention(self.step_dim, 4)
        self.sa5 = SelfAttention(self.step_dim, 4)

    def inipos(self, channels):
        inv_freq = 1.0 / (
            (2000 * 10) ** (torch.arange(0, channels, 2).float() / channels)
        )  # .to(self.device)
        t = torch.arange(0, 2000)[:, None]  # .to(self.device)
        # print(t.shape, inv_freq.shape)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        # print(pos_enc_a.shape)
        pos_enc = torch.cat(
            [pos_enc_a, pos_enc_b], dim=1
        )  # [None, :, :].repeat(x.shape, 1, 1)

        return pos_enc  # .transpose(0, 1)

    def pos_encoding(self, t, channels, freq=1000):
        inv_freq = 1.0 / (freq ** (torch.arange(0, channels, 2).float() / channels)).to(
            self.device
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, input_dict):
        x = input_dict["seq_t"]
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        # print(x)
        x = self.embed(x)
        x = torch.cat(
            [x, self.pe[None, : x.shape[1], :].repeat(x.shape[0], 1, 1)], dim=2
        )
        x = self.inc(x)
        x = self.sa1(x)
        x = self.sa2(x)
        x = self.sa3(x)
        x = self.sa4(x)
        x = self.sa5(x)
        return x


class IonclfBaseline(IonBaseclf):
    def __init__(
        self,
        embed_dim=128,
        pos_dim=32,
        addadversial=True,
        lamb=0.1,
        lr=5e-4,
        step_lambda=True,
        step=1.5,
        max_lambda=6,
        thres=0.95,
        p=0.2,
        weight_decay=0.005,
        clf="linear",
        dis="linear",
    ) -> None:
        super().__init__(
            addadversial=addadversial,
            lamb=lamb,
            lr=lr,
            step_lambda=step_lambda,
            step=step,
            max_lambda=max_lambda,
            thres=thres,
            weight_decay=weight_decay,
        )

        self.feature_extract = SeqTransformer(embed_dim, pos_dim)

        self.p = p
        self.reverse = GradientReversal(1)

        assert clf in ["linear", "cnn"]
        assert dis in ["linear", "cnn"]
        if clf == "cnn":
            self.clf = CNNcls(input_dim=embed_dim * 2)
        else:
            self.clf = Linearcls(input_dim=embed_dim * 2)
        if dis == "cnn":
            self.dis = CNNcls(input_dim=embed_dim * 2)
        else:
            self.dis = Linearcls(input_dim=embed_dim * 2)

    def forward(self, input_dict):
        # print("forward")

        x = self.feature_extract(input_dict)
        x1 = self.reverse(x)
        x1 = x
        pre = self.clf(x)
        pre = F.sigmoid(pre)

        y = self.dis(x1)
        y = F.sigmoid(y)

        return pre, y


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
