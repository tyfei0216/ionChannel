import esm
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from attr import dataclass
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


class Linearlayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, layer_norm=False, activate="gelu"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if layer_norm is not None:
            self.ln = nn.LayerNorm(out_dim)
        else:
            self.ln = None
        if dropout > 0 and dropout < 1:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if activate == "gelu":
            self.activate = nn.GELU()
        elif activate == "relu":
            self.activate = nn.ReLU()
        elif activate == "leakyrelu":
            self.activate = nn.LeakyReLU()
        else:
            self.activate = nn.Identity()
            # raise ValueError("activate %s not supported" % acivate)
        # self.activate = activate

    def forward(self, x):
        x = self.linear(x)
        if self.ln is not None:
            x = self.ln(x)
        x = self.activate(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Linearcls(nn.Module):
    """simple linear classifier

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_dim=1536,
        take_embed="first",
        dropout=-1,
        p0=None,
        output_dim=1,
        hidden_dim=256,
        hidden_layer=-1,
        activate="gelu",
        layer_norm=False,
    ):
        super().__init__()

        assert take_embed in ["first", "mean", "max"]
        self.embed_dim = input_dim
        self.dropout = dropout
        self.take_embed = take_embed
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        if hidden_layer == -1:
            self.l1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
            self.l2 = nn.Linear(self.embed_dim // 2, self.embed_dim // 4)
            self.l3 = nn.Linear(self.embed_dim // 4, output_dim)
            self.ln1 = nn.LayerNorm(self.embed_dim // 2)
            self.ln2 = nn.LayerNorm(self.embed_dim // 4)

            if dropout > 0 and dropout < 1:
                self.dropout1 = nn.Dropout(p=self.dropout)
                self.dropout2 = nn.Dropout(p=self.dropout)
            else:
                self.dropout1 = None
                self.dropout2 = None
            self.layers = None
        else:

            in_dims = [input_dim] + [hidden_dim] * (hidden_layer)
            output_dims = [hidden_dim] * (hidden_layer + 1)
            self.layers = nn.ModuleList(
                [
                    Linearlayer(
                        in_dims[i],
                        output_dims[i],
                        dropout=dropout,
                        layer_norm=layer_norm,
                        activate=activate,
                    )
                    for i in range(len(in_dims))
                ]
            )
            self.output = nn.Linear(hidden_dim, output_dim)

        if p0 is None:
            self.p0 = None
        else:
            self.p0 = nn.Dropout(p0)

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

        if self.layers is None:
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
            return x

        for layer in self.layers:
            x = layer(x)
        # print("lin", x.shape)
        x = self.output(x)
        return x
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


class ListMLE(nn.Module):
    def __init__(self, max_val=30, eps=1e-8):
        super().__init__()
        self.max_val = max_val
        self.eps = eps

    def forward(self, x):
        while torch.max(x) > self.max_val:
            x = x / 2
        sumx = x.exp().cumsum(0)
        sumx = torch.log(sumx + self.eps) - x

        return torch.sum(sumx)


@dataclass
class IonclfConfig:
    stage: str = "base_learning"

    addadversial: bool = True
    lambda_ini: float = 0.1
    step_lambda: float = 1.5
    max_lambda: float = 6
    lambda_thres: float = 0.95
    dis_loss: str = "bce"
    dis: str = "linear"
    dis_params: dict = {}

    lr: float = 5e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 0.005

    clf: str = "linear"
    clf_params: dict = {}

    addition_clf: str = "linear"
    addition_clf_params: dict = {}
    additional_label_weights: list[float] = []
    weight_step: float = 1.2
    weight_max: float = 15.0
    pos_weights: list[float] = None

    logits: float = -1.0
    mask_weight: float = 0.0
    logit_dim: int = 64

    list_freq: float = 1.0
    list_learning_lambda: float = 1.0
    mean_target: float = None
    std_target: float = None
    list_activate: str = "None"
    max_val: float = 30.0
    eps: float = 1e-8
    cache_list: bool = True

    bayes_predict: int = -1


class IonBaseclf(L.LightningModule):
    """
    base class for the ion channel classifier, handles training process and
    domain adaptation

    Args:
        L (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(
        self,
        config: IonclfConfig,
    ):
        super().__init__()
        # print(config.pos_weights, config.additional_label_weights)
        self.addadversial = config.addadversial

        if not isinstance(config.additional_label_weights, torch.Tensor):
            config.additional_label_weights = torch.tensor(
                config.additional_label_weights, requires_grad=False
            )
        # self.additional_label_weights = additional_label_weights

        additional_label_weights = torch.nn.Parameter(config.additional_label_weights)
        self.register_parameter("additional_label_weights", additional_label_weights)
        self.additional_label_weights.requires_grad = False

        self.weight_step = config.weight_step
        self.weight_max = config.weight_max
        if config.pos_weights is None:
            config.pos_weights = torch.ones_like(config.additional_label_weights)

        if not isinstance(config.pos_weights, torch.Tensor):
            pos_weights = torch.tensor(config.pos_weights, requires_grad=False)

        # print(pos_weights)
        pos_weights = torch.nn.Parameter(pos_weights)
        self.register_parameter("pos_weights", pos_weights)
        self.pos_weights.requires_grad = False
        # print(self.pos_weights, self.additional_label_weights)
        assert len(self.pos_weights) == len(self.additional_label_weights)
        # print(pos_weights.shape, additional_label_weights.shape)
        # assert len(pos_weights) == len(additional_label_weights)

        self.addadversial = config.addadversial
        self.reverse = GradientReversal(1)

        self.lamb = config.lambda_ini
        self.step_lambda = config.step_lambda
        self.max_lambda = config.max_lambda
        self.thres = config.lambda_thres
        self.update_epoch = False
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.lr_backbone = config.lr_backbone

        self.acc = torchmetrics.Accuracy(task="binary")

        self.last_train_step = 0

        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.load_freeze = None

        self.dis_loss = config.dis_loss

        self.listmle = ListMLE(max_val=config.max_val, eps=config.eps)

        self.list_learning_lambda = config.list_learning_lambda

        self.cache_list = config.cache_list

        self.embedding_cache = {}

        self.mean_target = config.mean_target
        self.std_target = config.std_target
        self.list_freq = config.list_freq
        self.list_activate = config.list_activate

        self.bayes_predict = config.bayes_predict

        assert config.stage in ["base_learning", "active_learning"]

        self.stage = config.stage
        print("initized model for %s stage" % config.stage)

        assert config.clf in ["linear", "cnn"]
        assert config.dis in ["linear", "cnn"]

        if config.clf == "cnn":
            self.clf = CNNcls(**config.clf_params)
        else:
            self.clf = Linearcls(**config.clf_params)

        if config.dis == "cnn":
            self.dis = CNNcls(**config.dis_params)
        else:
            self.dis = Linearcls(**config.dis_params)

        if config.addition_clf is not None:
            assert config.addition_clf in ["linear", "cnn"]
            self.additional_clf = Linearcls(**config.addition_clf_params)
        else:
            self.additional_clf = None

        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

        self.logits = config.logits

        self.mask_weight = config.mask_weight

        self.logit_dim = config.logit_dim

    def getScore(self, input_dict, idx=None):
        """only used for active learning stage
            return the score of the input data for List loss

        Args:
            input_dict (_type_): _description_
        """
        # print(idx)
        with torch.no_grad():
            if idx is not None and self.cache_list:
                if idx in self.embedding_cache:
                    # print("cached")
                    x = self.embedding_cache[idx]
                else:
                    # print("uncached", idx)
                    x = self.getEmbedding(input_dict)
                    self.embedding_cache[idx] = x.detach()
            else:
                x = self.getEmbedding(input_dict)
        pre = self.clf(x)
        return pre

    def getEmbedding(self, x, return_logits=False):
        raise NotImplementedError

    def forward(self, input_dict):
        if self.logits >= 0.0:
            x, h = self.getEmbedding(input_dict, return_logits=True)
        else:
            x = self.getEmbedding(input_dict)

        x1 = self.reverse(x)
        pre = self.clf(x)
        if len(self.additional_label_weights) == 0:
            pre = F.sigmoid(pre)
        else:
            additionalpre = self.additional_clf(x)
            pre = (F.sigmoid(pre), F.sigmoid(additionalpre))

        y = self.dis(x1)
        if self.dis_loss == "bce":
            y = F.sigmoid(y)
        if self.logits >= 0.0:
            return pre, y, h
        return pre, y

    def _common_training_step(self, batch):
        X1, y, X2 = batch
        if self.logits < 0.0:
            y_pre, dis_pre_x1 = self(X1)
            _y, dis_pre_x2 = self(X2)
            loss_logit = None
        else:
            y_pre, dis_pre_x1, logits1 = self(X1)
            _y, dis_pre_x2, logits2 = self(X2)
            # print(logits1.shape, X1["ori_seq_t"].shape, X1["mask"].shape)
            loss_logit1 = self.cross_entropy(
                logits1.view(-1, self.logit_dim), X1["ori_seq_t"].view(-1).long()
            )
            mask1 = 1 - X1["mask"].float()
            mask1 += self.mask_weight
            loss_logit1 = torch.sum(loss_logit1 * mask1) / (torch.sum(mask1) + 1e-8)
            loss_logit2 = F.cross_entropy(
                logits2.view(-1, self.logit_dim), X2["ori_seq_t"].view(-1).long()
            )
            mask2 = 1 - X2["mask"].float()
            mask2 += self.mask_weight
            loss_logit2 = torch.sum(loss_logit2 * mask2) / (torch.sum(mask2) + 1e-8)
            loss_logit = (loss_logit1 + loss_logit2) / 2

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
            qq = self.pos_weights[None, :].repeat(y_pre1.shape[0], 1)
            q[y2 < 0] = 0.0
            q.require_grad = False
            qq.require_grad = False
            qq[y2 < 0] = 1.0
            y2[y2 < 0] = 0
            y2[y2 > 1] = 1

            # print(qq)
            # print(y_pre2, y2, q)
            loss1_a = F.binary_cross_entropy(
                y_pre2,
                y2.float(),
                weight=q * qq,  # self.additional_label_weights,
            )
            # print("finish loss")
            y_pre = y_pre1  # .squeeze(-1)
            y = y1.squeeze(-1)

        if self.dis_loss == "bce":
            loss2 = F.binary_cross_entropy(
                dis_pre_x1, torch.zeros_like(dis_pre_x1)
            ) + F.binary_cross_entropy(dis_pre_x2, torch.ones_like(dis_pre_x1))
        elif self.dis_loss == "wes":
            loss2 = torch.mean(dis_pre_x1 - dis_pre_x2)
        else:
            raise NotImplementedError

        if self.addadversial:
            loss = loss1 + loss2 * self.lamb
        else:
            loss = loss1

        if len(self.additional_label_weights) > 0:
            loss += loss1_a
            loss1 = (loss1, loss1_a)

        if loss_logit is not None:
            loss += loss_logit * self.logits
            loss2 = (loss2, loss_logit)

        return loss, loss1, loss2, y_pre, y

    # def getScore(self, input_dict, idx):
    #     raise NotImplementedError

    def _list_training_step(self, batch):

        loss_list = 0

        scores = []
        for i, x in enumerate(batch):
            # print(x["id"][0])

            s = self.getScore(x, x["id"][0]).squeeze()
            # if i == 42:
            #     print(x["seq_t"][0][12], s)
            scores.append(s)
        scores = torch.stack(scores)

        if self.mean_target is not None:
            loss_list += (torch.mean(F.sigmoid(scores)) - self.mean_target) ** 2

        if self.std_target is not None:
            loss_list += (torch.std(F.sigmoid(scores)) - self.std_target) ** 2

        if self.stage == "active_learning":
            if self.list_activate == "None":
                # print(scores.requires_grad)
                loss_list += self.listmle(scores)
            elif self.list_activate == "sigmoid":
                loss_list += self.listmle(F.sigmoid(scores))
            else:
                raise ValueError("list activate %s not supported" % self.list_activate)
            # loss = self._active_training_step(batch)
            # return loss
            # loss += self.listmle(scores)

        return loss_list

    def training_step(self, batch, batch_idx):

        # base learning step
        loss, loss1, loss2, y_pre, y = self._common_training_step(batch[:3])

        # if len(batch) == 3:
        #     loss, loss1, loss2, y_pre, y = self._common_training_step(batch)
        # else:
        #     loss, loss1, loss2, y_pre, y = self._common_training_step(batch[:3])

        if self.logits >= 0.0:
            loss2, loss_logit = loss2

        train_output_dict = {
            "loss": loss.detach().cpu(),
            "loss2": loss2.detach().cpu(),
            "y": y_pre.detach().squeeze(-1).cpu(),
            "true_label": y.cpu(),
        }

        if self.logits >= 0.0:
            train_output_dict["loss_logit"] = loss_logit.detach().cpu()

        # acc = self.acc(y_pre.squeeze(-1), y)
        if len(self.additional_label_weights) == 0:
            train_output_dict["loss1"] = loss1.detach().cpu()
        else:
            train_output_dict["loss1"] = loss1[0].detach().cpu()
            train_output_dict["loss1_a"] = loss1[1].detach().cpu()

        # list learning step
        # r = np.random.uniform(0.001, 0.999)
        if len(batch) == 4:
            r = np.random.uniform(0.001, 0.999)
            if self.stage == "active_learning" or r < self.list_freq:
                loss_list = self._list_training_step(batch[3])
                if self.stage == "active_learning":
                    train_output_dict["loss_list"] = loss_list.detach().cpu()
                loss = loss * self.list_learning_lambda + loss_list
                # loss = loss_list
                # loss = loss_list
                train_output_dict["loss"] = loss.detach().cpu()
        # print("train loss:   ", train_output_dict["loss"])
        self.training_step_outputs.append(train_output_dict)
        return loss

    # def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
    #     self.temp_cache = self.cache_list
    #     self.cache_list = True

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:

        loss = []
        loss1 = []
        loss1_a = []
        loss_list = []
        loss_logit = []
        loss2 = []
        scores = []
        y = []
        for i in self.training_step_outputs[self.last_train_step :]:
            loss.append(i["loss"])
            loss1.append(i["loss1"])
            loss2.append(i["loss2"])
            scores.append(i["y"])
            y.append(i["true_label"])
            if "loss1_a" in i:
                loss1_a.append(i["loss1_a"])
            if "loss_list" in i:
                loss_list.append(i["loss_list"])
            if "loss_logit" in i:
                loss_logit.append(i["loss_logit"])

        loss = torch.stack(loss).mean()
        loss1 = torch.stack(loss1).mean()
        loss2 = torch.stack(loss2).mean()
        scores = torch.cat(scores)
        y = torch.cat(y)
        acc = self.acc(scores, y)
        log_dict = {
            "predict loss": loss1,
            "adversial loss": loss2,
            "loss": loss,
            "acc": acc,
        }

        if len(loss_list) > 0:
            loss_list = torch.stack(loss_list).mean()
            log_dict["list loss"] = loss_list
        if len(loss_logit) > 0:
            loss_logit = torch.stack(loss_logit).mean()
            log_dict["logits loss"] = loss_logit
        if len(loss1_a) > 0:
            loss1_a = torch.stack(loss1_a).mean()
            log_dict["additional loss"] = loss1_a

        self.last_train_step = len(self.training_step_outputs)

        self.log_dict(
            log_dict,
            logger=True,
            prog_bar=True,
        )

    def _common_epoch_end(self, outputs):

        if len(outputs) == 0:
            return 0, 0

        loss = []
        scores = []
        y = []
        loss_mle = []
        loss_logit = []

        for i in outputs:
            if "loss" in i:
                loss.append(i["loss"])
            if "loss_list" in i:
                loss_mle.append(i["loss_list"])
            if "true_label" in i:
                y.append(i["true_label"])
            if "y" in i:
                scores.append(i["y"])
            if "loss_logit" in i:
                loss_logit.append(i["loss_logit"])

        loss = torch.stack([i["loss"] for i in outputs]).mean()
        y = torch.concatenate([i["true_label"] for i in outputs])
        scores = torch.concatenate([i["y"] for i in outputs])
        # print(scores, y)
        outputs.clear()
        losses = {"loss": loss}
        if len(loss_mle) > 0:
            loss_mle = torch.stack(loss_mle).mean()
            losses["list_loss"] = loss_mle
        if len(loss_logit) > 0:
            loss_logit = torch.stack(loss_logit).mean()
            losses["logit"] = loss_logit
        return losses, self.acc(scores, y).cpu().item()
        # if len(loss_mle) == 0:
        #     return loss, self.acc(scores, y).cpu()
        # else:
        #     loss_mle = torch.stack(loss_mle).mean()
        #     return (loss.item(), loss_mle.item()), self.acc(scores, y).cpu().item()

    def on_training_epoch_end(self):

        losses, acc = self._common_epoch_end(self.training_step_outputs)

        if isinstance(losses, dict):
            print("\nfinish training epoch acc %f, loss:" % acc, losses, "\n")
            d = {}
            for i in losses:
                d["train_" + i] = losses[i]
            d["train_acc"] = acc
            self.self.log_dict(
                d,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        else:
            print("finish training epoch, loss %f, acc %f" % (losses, acc))
            self.log_dict(
                {
                    "train_loss": losses,
                    "train_acc": acc,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        self.last_train_step = 0

        self.update_epoch = True

    def validation_step(self, batch, batch_idx):

        # if len(batch) == 3:
        #     loss, loss1, loss2, y_pre, y = self._common_training_step(batch)
        # else:
        loss, loss1, loss2, y_pre, y = self._common_training_step(batch[:3])

        validation_output_dict = {
            "loss": loss.cpu(),
            "y": y_pre.squeeze(-1).cpu(),
            "true_label": y.cpu(),
        }

        if len(batch) == 4:
            r = np.random.uniform(0.001, 0.999)
            if self.stage == "active_learning" or r < self.list_freq:
                loss_list = self._list_training_step(batch[3])
                loss = loss * self.list_learning_lambda + loss_list
                if self.stage == "active_learning":
                    validation_output_dict["loss_list"] = loss_list.cpu()
                validation_output_dict["loss"] = loss.cpu()

        self.validation_step_outputs.append(validation_output_dict)

        return loss

    def on_validation_epoch_end(self):

        # self.cache_list = self.temp_cache
        # if not self.cache_list:
        #     self.embedding_cache.clear()

        losses, acc = self._common_epoch_end(self.validation_step_outputs)
        if isinstance(losses, dict):

            print("\n finish validating acc: %f loss:" % acc, losses, "\n")
            d = {}
            for i in losses:
                d["validate_" + i] = losses[i]
            d["validate_acc"] = acc
            self.log_dict(
                d,
                on_step=False,
                logger=True,
                on_epoch=True,
                prog_bar=False,
            )
        else:
            print("\n finish validating, loss %f, acc %f \n" % (losses, acc))
            self.log_dict(
                {"validate_loss": losses, "validate_acc": acc},
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

        if self.weight_step > 1.0:
            if self.thres[0] > acc:
                if (
                    torch.sum(self.additional_label_weights) * self.weight_step
                    < self.weight_max
                ):
                    self.additional_label_weights *= self.weight_step

        if self.step_lambda > 1.0:
            if self.thres[1] < acc:
                self.lamb = min(self.lamb * self.step_lambda, self.max_lambda)
            if self.thres[0] > acc:
                self.lamb = max(self.lamb / self.step_lambda, 0.1)

        print("lambda", self.lamb)

    def test_step(self, batch, batch_idx):
        x = batch
        y_pre, _ = self(x)
        return y_pre

    def predict_batch(self, batch):
        # print(batch, self(batch))
        if self.logits >= 0.0:
            pre, _, _ = self(batch)
        else:
            pre, _ = self(batch)

        if len(self.additional_label_weights) == 0:
            pre0 = pre.squeeze()
            pre = pre0
        else:
            pre0 = pre[0].squeeze()
            pre1 = pre[1].squeeze()
            pre = (pre0, pre1)
        return pre

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list):
            if len(batch) == 3:
                X1, y, X2 = batch
            elif len(batch) == 2:
                X1, y = batch
            elif len(batch) == 4:
                X1, y, X2, _ = batch
            else:
                raise ValueError
        else:
            X1 = batch
            y = None
        # print(X1, len)
        if self.bayes_predict > 0:
            self.train()
            pre = []
            pre0 = []
            for _ in range(self.bayes_predict):
                res = self.predict_batch(X1)
                if isinstance(res, tuple):
                    pre.append(res[0])
                    pre0.append(res[1])
                else:
                    pre.append(res)
            pre = torch.stack(pre)
            pre = torch.mean(pre, 0)
            if len(pre0) > 0:
                pre0 = torch.stack(pre0)
                pre0 = torch.mean(pre0, 0)
                pre = (pre, pre0)
        else:
            self.eval()
            pre = self.predict_batch(X1)
            # pre = torch.cat([pre0, pre1], 1)
        # pre = pre.squeeze()
        if y is None:
            return pre
        else:
            return pre, y

    def configure_optimizers(self):
        print("get training optimizer")

        # lower learning weights for the backbone lora and even lower for backbone
        if self.lr_backbone is not None:
            l1 = []
            l2 = []
            l3 = []
            for i, j in self.named_parameters():
                if j.requires_grad:
                    if "esm" in i:
                        # sequence_heads or lora weights
                        if "lora" in i or "sequence_head" in i:
                            l3.append(j)
                        # original esm weights or or norm weights
                        else:
                            l1.append(j)
                    else:
                        # classifier weights
                        l2.append(j)

            param_dicts = [
                {
                    "params": l1,
                    "lr": self.lr_backbone / 2,
                },
                {
                    "params": l3,
                    "lr": self.lr_backbone,
                },
                {
                    "params": l2,
                    "lr": self.lr,
                },
            ]
            return torch.optim.Adam(param_dicts, weight_decay=self.weight_decay)

        # if self.lr_backbone is not None:
        #     l1 = []
        #     l2 = []
        #     for i, j in self.named_parameters():
        #         if j.requires_grad:
        #             if "esm" in i:
        #                 l1.append(j)
        #             else:
        #                 l2.append(j)

        #     param_dicts = [
        #         {
        #             "params": l1,
        #             "lr": self.lr_backbone,
        #         },
        #         {
        #             "params": l2,
        #             "lr": self.lr,
        #         },
        #     ]
        #     return torch.optim.Adam(param_dicts, weight_decay=self.weight_decay)

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
        config: IonclfConfig,
    ) -> None:
        super().__init__(config)

        self.esm_model = esm_model

    def getEmbedding(self, input_dict, return_logits=False):

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

        x = representations.embeddings
        h = representations.sequence_logits
        if return_logits:
            return x, h

        return x

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
        config: IonclfConfig,
        num_layers=33,
    ) -> None:
        super().__init__(
            config=config,
        )

        self.num_layers = num_layers

        self.esm_model = esm_model

    def getEmbedding(self, x, return_logits=False):
        representations = self.esm_model(x, repr_layers=[self.num_layers])

        x = representations["representations"][self.num_layers][:, 0]
        h = representations["logits"]
        if return_logits:
            return x, h
        return x

    def on_save_checkpoint(self, checkpoint):
        backbones = []
        for i, j in self.named_parameters():
            if "esm" in i and not j.requires_grad:
                backbones.append(i)
        for i in backbones:
            del checkpoint["state_dict"][i]


class IonclfESMC(IonBaseclf):
    def __init__(
        self,
        esm_model,
        config: IonclfConfig,
    ) -> None:
        super().__init__(
            config=config,
        )
        self.save_hyperparameters(ignore=["esm_model"])

        self.esm_model = esm_model

    def getEmbedding(self, input_dict, return_logits=False):

        assert "seq_t" in input_dict
        inputs = input_dict["seq_t"]
        if len(inputs.size()) == 1:
            inputs = inputs.unsqueeze(0)

        representations = self.esm_model(
            sequence_tokens=inputs,
        )

        x = representations.embeddings
        h = representations.sequence_logits
        if return_logits:
            return x, h
        return x

    # def forward(self, input_dict):

    #     # print(input_dict)

    #     assert "seq_t" in input_dict
    #     inputs = input_dict["seq_t"]
    #     if len(inputs.size()) == 1:
    #         inputs = inputs.unsqueeze(0)

    #     # print(inputs, inputs.shape)

    #     representations = self.esm_model(
    #         sequence_tokens=inputs,
    #     )

    #     x = representations.embeddings  # [:, 0]
    #     # x = x.to(torch.float32)
    #     x1 = self.reverse(x)
    #     pre = self.clf(x)
    #     if len(self.additional_label_weights) == 0:
    #         pre = F.sigmoid(pre)
    #     else:
    #         additionalpre = self.additional_clf(x)
    #         pre = (F.sigmoid(pre), F.sigmoid(additionalpre))

    #     y = self.dis(x1)
    #     if self.dis_loss == "bce":
    #         y = F.sigmoid(y)
    #     return pre, y

    def on_save_checkpoint(self, checkpoint):
        backbones = []
        for i, j in self.named_parameters():
            if "esm" in i and not j.requires_grad:
                backbones.append(i)
        for i in backbones:
            del checkpoint["state_dict"][i]

    # def updateLambda(self, acc):
    #     super().updateLambda(acc)
    #     if self.weight_step > 1.0:
    #         if self.thres[0] > acc:
    #             if (
    #                 torch.sum(self.additional_label_weights) * self.weight_step
    #                 < self.weight_max
    #             ):
    #                 self.additional_label_weights = (
    #                     self.additional_label_weights * self.weight_step
    #                 )

    # def configure_optimizers(self):
    #     print("get training optimizer")
    #     if self.lr_backbone is not None:
    #         l1 = []
    #         l2 = []
    #         l3 = []
    #         for i, j in self.named_parameters():
    #             if j.requires_grad:
    #                 if "esm" in i:
    #                     if "lora" in i:
    #                         l3.append(j)
    #                     else:
    #                         l1.append(j)
    #                 else:
    #                     l2.append(j)

    #         param_dicts = [
    #             {
    #                 "params": l1,
    #                 "lr": self.lr_backbone,
    #             },
    #             {
    #                 "params": l3,
    #                 "lr": self.lr_backbone / 10,
    #             },
    #             {
    #                 "params": l2,
    #                 "lr": self.lr,
    #             },
    #         ]
    #         return torch.optim.Adam(param_dicts, weight_decay=self.weight_decay)
    #     else:
    #         return super().configure_optimizers()


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

        self.output_head = nn.Sequential(
            nn.Linear(self.step_dim, self.step_dim),
            nn.ReLU(),
            nn.Linear(self.step_dim, self.step_dim),
            nn.ReLU(),
            nn.Linear(self.step_dim, 35),
        )

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
        h = self.output_head(x)
        return x, h


class IonclfBaseline(IonBaseclf):
    def __init__(
        self,
        config: IonclfConfig,
        embed_dim=128,
        pos_dim=32,
    ) -> None:
        super().__init__(
            config=config,
        )

        self.feature_extract = SeqTransformer(embed_dim, pos_dim)

    def getEmbedding(self, input_dict, return_logits=False):
        x, h = self.feature_extract(input_dict)
        if return_logits:
            return x, h
        return x


def fixParameters(model, unfix=["9", "10", "11"], fix=[]):
    for i, j in model.named_parameters():
        flag = 1
        for k in unfix:
            if str(k) in i:
                flag = 0

        for k in fix:
            if str(k) in i:
                flag = 1

        if flag == 1:
            j.requires_grad = False
        else:
            j.requires_grad = True

    return model


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, dtype=torch.float):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).to(dtype))
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank, dtype=dtype) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim, dtype=dtype) * std_dev)
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha, dtype=torch.float):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha, dtype
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def _set_submodule(submodule, module_path, new_module):
    tokens = module_path.split(".")
    for token in tokens[:-1]:
        submodule = getattr(submodule, token)
    setattr(submodule, tokens[-1], new_module)


def addlora(esm_model, layers, ranks, alphas, dtype=torch.float):
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
                        LinearWithLoRA(j, rank, alpha, dtype=dtype),
                    )
    return esm_model
