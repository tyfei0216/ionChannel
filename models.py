import torch
import esm 
import torch.nn as nn
from esm.modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer
import torch.nn.functional as F
import pytorch_lightning as L

# from torchmetrics import Metric

import torchmetrics

from torch.autograd import Function

class GradientReversal(Function):
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
revgrad = GradientReversal.apply

class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)

class ionclf(L.LightningModule):
    def __init__(self, esm_model, unfix = None) -> None:
        super().__init__()
        self.num_layers = esm_model.num_layers 
        self.embed_dim = esm_model.embed_dim 
        self.attention_heads = esm_model.attention_heads 
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b") 
        self.alphabet_size = len(self.alphabet)

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

        # self.automatic_optimization = False

        self.reverse = GradientReversal(1)

        if unfix is None:
            self.fixParameters() 
        else:
            self.fixParameters(unfix)

        self.acc = torchmetrics.Accuracy(task="binary")
        # self.linear1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        # self.ln1 = nn.LayerNorm(self.embed_dim // 2)
        # self.linear2 = nn.Linear(self.embed_dim // 2, self.embed_dim // 4) 
        # self.ln2 = nn.LayerNorm(self.embed_dim // 4)
        # self.linear3 = nn.Linear(self.embed_dim // 4, 1)


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
        # x = self.linear1(x) 
        # x = self.ln1(x)
        # x = F.gelu(x) 
        # x = self.linear2(x)
        # x = self.ln2(x) 
        # x = F.gelu(x) 
        # x = self.linear3(x) 
        # x = F.sigmoid(x) 
        return  pre, y
    
    def training_step(self, batch, batch_idx):
        # cls_opt, dis_opt = self.optimizers() 
        X1, y, X2 = batch 
        # print(X1.shape, X2.shape)

        # real_label = torch.ones((batch_size, 1), device=self.device)
        # fake_label = torch.zeros((batch_size, 1), device=self.device)

        y_pre, dis_pre_x1 = self(X1) 
        _y, dis_pre_x2 = self(X2) 

        loss1 = F.binary_cross_entropy(y_pre.squeeze(), y.float()) 
        loss2 = F.binary_cross_entropy(dis_pre_x1, torch.zeros_like(dis_pre_x1)) + \
            F.binary_cross_entropy(dis_pre_x2, torch.ones_like(dis_pre_x1))
        loss = loss1+loss2

        acc = self.acc(y_pre.squeeze(), y)

        self.log_dict({"train predict loss":loss1.item(), "train adversial loss":loss1.item(), "train acc":acc}, prog_bar=True, on_step=True)


        # cls_opt.zero_grad()
        # self.manual_backward(loss1)
        # cls_opt.step() 

        # loss2 = F.binary_cross_entropy(dis_pre_x1, torch.ones_like(dis_pre_x1)) + F.binary_cross_entropy(dis_pre_x2, torch.zeros_like(dis_pre_x2))
        return loss#{"loss":loss1+loss2, "y":y_pre, "true_label":y} 
    

    
    def validation_step(self, batch, batch_idx):
        # this is the test loop
        X1, y, X2 = batch 
        y_pre, dis_pre_x1 = self(X1) 
        _y, dis_pre_x2 = self(X2)

        loss1 = F.binary_cross_entropy(y_pre.squeeze(), y.float()) 
        loss2 = F.binary_cross_entropy(dis_pre_x1, torch.zeros_like(dis_pre_x1)) + \
            F.binary_cross_entropy(dis_pre_x2, torch.ones_like(dis_pre_x1))

        loss = loss1+loss2
        acc = self.acc(y_pre.squeeze(), y)

        self.log_dict({"train predict loss":loss1.item(), "train adversial loss":loss2.item(), "train acc":acc}, prog_bar=True, on_step=True)


        # cls_opt.zero_grad()
        # self.manual_backward(loss1)
        # cls_opt.step() 

        # loss2 = F.binary_cross_entropy(dis_pre_x1, torch.ones_like(dis_pre_x1)) + F.binary_cross_entropy(dis_pre_x2, torch.zeros_like(dis_pre_x2))
        return loss#{"loss":loss1+loss2, "y":y_pre, "true_label":y} 

    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x = batch
        y_pre, _ = self(x) 
        return y_pre 
        
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, self.parameters()), lr=0.005)

        return optimizer    

        cls_para = []
        for i, j in self.named_parameters():
            if "dis" in i:
                continue 
            if j.requires_grad:
                cls_para.append(j)       
        cls_opt = torch.optim.Adam(cls_para, lr=1e-5)

        dis_para = []
        for i, j in self.named_parameters():
            if not "dis" in i:
                continue 
            if j.requires_grad:
                dis_para.append(j)
        dis_opt = torch.optim.Adam(dis_para, lr=1e-5)


        return cls_opt, dis_opt

