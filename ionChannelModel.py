# run by CUDA_VISIBLE_DEVICES=3,4 python ./ionChannelModel.py
import json
import os
import sys

import torch

torch.set_float32_matmul_precision("medium")  # make lightning happy
# sys.path.append("/home/tyfei/fun/utils")

import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

path = "/home/tyfei/ionChannel/ckpts/Lora32unFix3/"
# strategy = L.strategies.DeepSpeedStrategy()
k = 2
with open(os.path.join(path, "config.json"), "r") as f:
    configs = json.load(f)

checkpoint_callback = ModelCheckpoint(
    monitor="validate_acc",  # Replace with your validation metric
    mode="max",  # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)
    save_top_k=k,  # Save top k checkpoints based on the monitored metric
    save_last=True,  # Save the last checkpoint at the end of training
    dirpath=path,  # Directory where the checkpoints will be saved
    filename="{epoch}-{validate_acc:.2f}",  # Checkpoint file naming pattern
)

from torch.utils import tensorboard
# profiler = PyTorchProfiler(
#         on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/ion_test"),
#     )
logger = TensorBoardLogger("tb_logs", name="ion_test")
trainer = L.Trainer(
    logger=logger,
    accelerator="gpu",
    devices=[0],
    max_epochs=configs["train"]["epoch"],
    accumulate_grad_batches=configs["train"]["accumulate_grad_batches"],
    callbacks=[checkpoint_callback],
)


import esm

import VirusDataset

model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
batch_converter = alphabet.get_batch_converter()
# X1, y, X2 = VirusDataset.readVirusSequences(trunc=998)

import random

random.seed(configs["train"]["seed"])
torch.manual_seed(configs["train"]["seed"])
# newX1 = [(y, x[1]) for x, y in zip(X1, y)]
# _, _, X1 = batch_converter(X1)
# _, _, X2 = batch_converter(X2)
# trainset = VirusDataset.SeqDataset2(
#     X1, y, X2[random.sample(range(X2.shape[0]), X1.shape[0])]
# )
# testset = VirusDataset.TestDataset(X2)
# ds = VirusDataset.SeqdataModule(trainset=trainset, testset=testset, batch_size=5)
ds = VirusDataset.SeqdataModule(batch_size=configs["train"]["batch_size"])

import models

# model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
# batch_converter = alphabet.get_batch_converter()

model = models.fixParameters(model, unfix=configs["pretrain_model"]["unfix_layers"])
model = models.addlora(
    model,
    layers=configs["pretrain_model"]["add_lora"],
    ranks=configs["pretrain_model"]["rank"],
    alphas=configs["pretrain_model"]["alpha"],
)


clsmodel = models.ionclf(
    model,
    step_lambda=configs["model"]["lambda_adapt"],
    lamb=configs["model"]["lambda_ini"],
    max_lambda=configs["model"]["max_lambda"],
    step=configs["model"]["lambda_step"],
    p=configs["model"]["dropout"],
    thres=configs["model"]["lambda_thres"],
    lr=configs["model"]["lr"],
)
# newmodel = esm.pretrained.esm2_t12_35M_UR50D()

# from functools import reduce

# from peft import LoraConfig, TaskType, get_peft_config, get_peft_model

# esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
# add_lora = ["k_proj", "q_proj", "v_proj", "fc1", "fc2"]
# targets = []
# for i, j in esm_model.named_modules():
#     if "layers" in i:
#         test = [sub in i for sub in add_lora]
#         test = reduce(lambda x, y: x or y, test)
#         if test:
#             targets.append(i)
# peft_config = LoraConfig(
#     inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=targets
# )
# esm_model = get_peft_model(esm_model, peft_config)
# clsmodel.esm_model = esm_model

trainer.fit(clsmodel, ds)


torch.save(clsmodel.state_dict(), path + "parms.pt")
# trainer.save_checkpoint("example.ckpt")
# torch.save(clsmodel, "./train624.pt")
