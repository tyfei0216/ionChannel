import torch
import esm 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import numpy as np

# from torchmetrics import Metric

import torchmetrics

from torch.autograd import Function

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
            grad_input = - alpha*grad_output
        return grad_input, None


class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return GradientR.apply(x, self.alpha)

class ionclf(L.LightningModule):
    def __init__(self, esm_model, unfix = None, addadversial=True, lamb=0.1, lr=5e-4) -> None:
        super().__init__()
        self.num_layers = esm_model.num_layers 
        self.embed_dim = esm_model.embed_dim 
        self.attention_heads = esm_model.attention_heads 
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b") 
        self.alphabet_size = len(self.alphabet)
        self.addadversial = addadversial
        self.lamb =  lamb
        self.lr = lr

        self.esm_model = esm_model 

        self.cls = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim // 2), 
                                  nn.LayerNorm(self.embed_dim // 2), 
                                  nn.GELU(), 
                                  nn.Linear(self.embed_dim // 2, self.embed_dim // 4), 
                                  nn.LayerNorm(self.embed_dim // 4), 
                                  nn.GELU(), 
                                  nn.Linear(self.embed_dim // 4, 1)
                                  )
        
        self.dis = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim // 2), 
                                  nn.LayerNorm(self.embed_dim // 2), 
                                  nn.GELU(), 
                                  nn.Linear(self.embed_dim // 2, self.embed_dim // 4), 
                                  nn.LayerNorm(self.embed_dim // 4), 
                                  nn.GELU(), 
                                  nn.Linear(self.embed_dim // 4, 1)
                                  )

        self.reverse = GradientReversal(1)

        if unfix is None:
            self.fixParameters() 
        else:
            self.fixParameters(unfix)

        self.acc = torchmetrics.Accuracy(task="binary")

        self.training_step_outputs = []
        self.validation_step_outputs = []



    def fixParameters(self, unfix=["9", "10", "11"]):
        for i, j in self.named_parameters():
            flag = 1
            if "esm_model" not in i:
                    flag = 0
            for k in unfix:
                if k in i:
                    flag = 0
    
            if flag == 1:
                j.requires_grad = False 
            else:
                j.requires_grad = True
     
    def forward(self, x):
        representations = self.esm_model(x, repr_layers=[self.num_layers])

        x = representations["representations"][self.num_layers][:, 0] 
        x1 = self.reverse(x)
        pre = self.cls(x) 
        pre = F.sigmoid(pre)

        y = self.dis(x1)
        y = F.sigmoid(y)

        return  pre, y
    
    def _common_training_step(self, batch):
        X1, y, X2 = batch 
        y_pre, dis_pre_x1 = self(X1) 
        _y, dis_pre_x2 = self(X2) 

        loss1 = F.binary_cross_entropy(y_pre.squeeze(), y.float()) 
        loss2 = F.binary_cross_entropy(dis_pre_x1, torch.zeros_like(dis_pre_x1)) + \
            F.binary_cross_entropy(dis_pre_x2, torch.ones_like(dis_pre_x1))
        
        if self.addadversial:
            loss = loss1+loss2*self.lamb
        else:
            loss = loss1
        
        return loss, loss1, loss2, y_pre, y


    def training_step(self, batch, batch_idx):

        loss, loss1, loss2, y_pre, y = self._common_training_step(batch)

        acc = self.acc(y_pre.squeeze(), y)

        self.log_dict({"predict loss":loss1.item(), "adversial loss":loss2.item(), "acc":acc}, prog_bar=True, on_step=True)
        self.training_step_outputs.append({"loss":loss.detach().cpu(), "y":y_pre.detach().squeeze().cpu(), "true_label":y.cpu()})
        
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
                "mean_loss":loss, 
                "train_acc": acc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
    
    def validation_step(self, batch, batch_idx):

        loss, loss1, loss2, y_pre, y = self._common_training_step(batch)

        acc = self.acc(y_pre.squeeze(), y)

        self.log_dict({"predict loss":loss1.item(), "adversial loss":loss2.item(), "acc":acc}, prog_bar=True, on_step=True)
        
        self.validation_step_outputs.append({"loss":loss.cpu(), "y":y_pre.squeeze().cpu(), "true_label":y.cpu()})
        
        return loss
    
    def on_validation_epoch_end(self):
        loss, acc = self._common_epoch_end(self.validation_step_outputs)
        # print("finish validating, loss %f, acc %f"%(loss, acc))
        self.log_dict(
            {
                "loss":loss, 
                "validate_acc": acc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
    
    def test_step(self, batch, batch_idx):
        x = batch
        y_pre, _ = self(x) 
        return y_pre 
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            if len(batch) == 3:
                X1, y, X2 = batch 
            elif len(batch) == 2:
                X1, y  = batch  
            else:
                raise ValueError   
        else:
            X1 = batch 
        pre, _ = self(X1)

        pre = pre.squeeze() 
        return pre
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, self.parameters()), lr=self.lr)

        return optimizer    

