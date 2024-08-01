import pickle

import pytorch_lightning
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# import pytorch_lightning as L
from pytorch_lightning.profilers import PyTorchProfiler

import models


def loadesm2(configs):
    import esm

    model, _ = esm.pretrained.esm2_t12_35M_UR50D()
    model = models.fixParameters(model, unfix=configs["pretrain_model"]["unfix_layers"])
    model = models.addlora(
        model,
        layers=configs["pretrain_model"]["add_lora"],
        ranks=configs["pretrain_model"]["rank"],
        alphas=configs["pretrain_model"]["alpha"],
    )
    return model


def loadesm3(configs):
    from esm.models.esm3 import ESM3

    model = ESM3.from_pretrained("esm3_sm_open_v1")
    model = models.fixParameters(model, unfix=configs["pretrain_model"]["unfix_layers"])
    model = models.addlora(
        model,
        layers=configs["pretrain_model"]["add_lora"],
        ranks=configs["pretrain_model"]["rank"],
        alphas=configs["pretrain_model"]["alpha"],
    )
    return model


LOAD_PRETRAIN = {
    "esm2": loadesm2,
    "esm3": loadesm3,
}


def loadPretrainModel(configs) -> nn.Module:
    if "pretrain_model" in configs:
        model = "esm2"
        if "model" in configs["pretrain_model"]:
            model = "esm3"
    else:
        return None

    if model in LOAD_PRETRAIN:
        return LOAD_PRETRAIN[model](configs)
    else:
        raise NotImplementedError


def loadPickle(path):
    with open(
        path,
        "rb",
    ) as f:
        data = pickle.load(f)
        for j in data:
            if "strcture_t" in j:
                j["structure_t"] = j.pop("strcture_t")
    return data


def loadDatasetesm2(configs):
    import VirusDataset

    ds = VirusDataset.SeqdataModule(batch_size=configs["train"]["batch_size"])
    return ds


def loadDatasetesm3(configs):
    import VirusDataset

    data1 = []
    label = []
    data2 = []
    lens = []

    for i in configs["dataset"]["pos"]:
        data = loadPickle(i)
        data1.extend(data)
        label.extend([1] * len(data))

    for i in configs["dataset"]["neg"]:
        data = loadPickle(i)
        data1.extend(data)
        label.extend([0] * len(data))

    for i in configs["dataset"]["test"]:
        data = loadPickle(i)
        for j in data:
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
    aug = VirusDataset.DataAugmentation(step_points, maskp, crop, lens)

    ds1 = VirusDataset.ESM3MultiTrackDataset(data1, data2, label, augment=aug)
    ds2 = VirusDataset.ESM3MultiTrackDatasetTEST(data2)

    ds = VirusDataset.ESM3datamodule(ds1, ds2)
    return ds


LOAD_DATASET = {
    "esm2": loadDatasetesm2,
    "esm3": loadDatasetesm3,
}


def loadDataset(configs) -> pytorch_lightning.LightningDataModule:
    dataset = "esm2"
    if "dataset" in configs:
        dataset = configs["dataset"]["type"]

    if dataset in LOAD_DATASET:
        return LOAD_DATASET[dataset](configs)
    else:
        raise NotImplementedError


def buildSimpleModel(configs, model=None):
    clsmodel = models.IonclfBaseline(
        step_lambda=configs["model"]["lambda_adapt"],
        lamb=configs["model"]["lambda_ini"],
        max_lambda=configs["model"]["max_lambda"],
        step=configs["model"]["lambda_step"],
        p=configs["model"]["dropout"],
        thres=configs["model"]["lambda_thres"],
        lr=configs["model"]["lr"],
        clf=configs["model"]["clf"],
        dis=configs["model"]["dis"],
    )
    return clsmodel


def buildesm2Model(configs, model):
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
    return clsmodel


def buildesm3Model(configs, model):
    clsmodel = models.IonclfESM3(
        model,
        step_lambda=configs["model"]["lambda_adapt"],
        lamb=configs["model"]["lambda_ini"],
        max_lambda=configs["model"]["max_lambda"],
        step=configs["model"]["lambda_step"],
        p=configs["model"]["dropout"],
        thres=configs["model"]["lambda_thres"],
        lr=configs["model"]["lr"],
        clf=configs["model"]["clf"],
        dis=configs["model"]["dis"],
    )
    return clsmodel


BUILD_MODEL = {
    "simple": buildSimpleModel,
    "esm2": buildesm2Model,
    "esm3": buildesm3Model,
}


def buildModel(configs, basemodel=None) -> pytorch_lightning.LightningModule:
    model = "esm2"
    if "type" in configs["model"]:
        model = configs["model"]["type"]

    if model in BUILD_MODEL:
        return BUILD_MODEL[model](configs, basemodel)
    else:
        raise NotImplementedError


def buildTrainer(configs, args):
    k = 2
    if "save" in configs["train"]:
        k = configs["train"]["save"]
    checkpoint_callback = ModelCheckpoint(
        monitor="validate_acc",  # Replace with your validation metric
        mode="max",  # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)
        save_top_k=k,  # Save top k checkpoints based on the monitored metric
        save_last=True,  # Save the last checkpoint at the end of training
        dirpath=args.path,  # Directory where the checkpoints will be saved
        filename="{epoch}-{validate_acc:.2f}",  # Checkpoint file naming pattern
    )
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "tb_logs/%s" % args.name
        ),
    )
    logger = TensorBoardLogger("tb_logs", name=args.name)

    trainer = pytorch_lightning.Trainer(
        strategy=args.strategy,
        logger=logger,
        accelerator="gpu",
        profiler=profiler,
        devices=args.devices,
        max_epochs=configs["train"]["epoch"],
        accumulate_grad_batches=configs["train"]["accumulate_grad_batches"],
        callbacks=[checkpoint_callback],
    )

    return trainer
