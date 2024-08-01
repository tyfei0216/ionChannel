# run by CUDA_VISIBLE_DEVICES=3,4 python ./ionChannelModel.py
import json
import os
import sys

import torch

torch.set_float32_matmul_precision("medium")  # make lightning happy
sys.path.append("/home/tyfei/ion_channel")

import argparse

import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    args = parser.parse_args()
    return args


def runesm3():
    args = parseArgs()
    path = args.path

    # path = "/home/tyfei/ionChannel/ckptsesm3/unfix3/"
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
        # strategy="FSDP",
        logger=logger,
        accelerator="gpu",
        devices=[1],
        max_epochs=configs["train"]["epoch"],
        accumulate_grad_batches=configs["train"]["accumulate_grad_batches"],
        callbacks=[checkpoint_callback],
    )

    import random

    import VirusDataset

    # model = ESM3.from_pretrained("esm3_sm_open_v1")
    # model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    # batch_converter = alphabet.get_batch_converter()
    # X1, y, X2 = VirusDataset.readVirusSequences(trunc=998)

    random.seed(configs["train"]["seed"])
    torch.manual_seed(configs["train"]["seed"])

    import pickle

    data1 = []
    label = []
    data2 = []
    lens = []

    for i in configs["dataset"]["pos"]:
        with open(
            i,
            "rb",
        ) as f:
            data = pickle.load(f)
            for j in data:
                if "strcture_t" in j:
                    j["structure_t"] = j.pop("strcture_t")
        data1.extend(data)
        label.extend([1] * len(data))

    for i in configs["dataset"]["neg"]:
        with open(
            i,
            "rb",
        ) as f:
            data = pickle.load(f)
            for j in data:
                if "strcture_t" in j:
                    j["structure_t"] = j.pop("strcture_t")
        data1.extend(data)
        label.extend([0] * len(data))

    for i in configs["dataset"]["test"]:
        with open(
            i,
            "rb",
        ) as f:
            data = pickle.load(f)
            for j in data:
                if "strcture_t" in j:
                    j["structure_t"] = j.pop("strcture_t")
                lens.append(len(j["ori_seq"]))
        data2.extend(data)

    step_points = configs["augmentation"]["step_points"]
    crop = configs["augmentation"]["crop"]
    maskp = [
        (i, j)
        for i, j in zip(
            configs["augmentation"]["maskp"], configs["augmentation"]["maskpc"]
        )
    ]
    # print(crop)
    aug = VirusDataset.DataAugmentation(step_points, maskp, crop, lens)

    ds1 = VirusDataset.ESM3MultiTrackDataset(data1, data2, label, augment=aug)
    ds2 = VirusDataset.ESM3MultiTrackDatasetTEST(data2)

    ds = VirusDataset.ESM3datamodule(ds1, ds2)

    import models

    # model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    # batch_converter = alphabet.get_batch_converter()
    # model = models.fixParameters(model, unfix=configs["pretrain_model"]["unfix_layers"])
    # model = models.addlora(
    #     model,
    #     layers=configs["pretrain_model"]["add_lora"],
    #     ranks=configs["pretrain_model"]["rank"],
    #     alphas=configs["pretrain_model"]["alpha"],
    # )
    # clsmodel = models.IonclfESM3(
    #     model,
    #     step_lambda=configs["model"]["lambda_adapt"],
    #     lamb=configs["model"]["lambda_ini"],
    #     max_lambda=configs["model"]["max_lambda"],
    #     step=configs["model"]["lambda_step"],
    #     p=configs["model"]["dropout"],
    #     thres=configs["model"]["lambda_thres"],
    #     lr=configs["model"]["lr"],
    # )
    clsmodel = models.IonclfBaseline()

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


if __name__ == "__main__":
    runesm3()
