import os
import pickle

import numpy as np
import pytorch_lightning
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger

# import pytorch_lightning as L
from pytorch_lightning.profilers import PyTorchProfiler
from torch.utils import tensorboard

import callbacks
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

    model = ESM3.from_pretrained("esm3_sm_open_v1", torch.device("cpu"))  # .cpu()
    q = [
        "transformer.blocks." + str(s) + "."
        for s in configs["pretrain_model"]["add_lora"]
    ]
    model = models.fixParameters(model, unfix=configs["pretrain_model"]["unfix_layers"])
    model = models.addlora(
        model,
        layers=q,
        ranks=configs["pretrain_model"]["rank"],
        alphas=configs["pretrain_model"]["alpha"],
    )
    return model


def loadesmc(configs):
    from esm.models.esmc import ESMC

    model = ESMC.from_pretrained(
        configs["pretrain_model"]["model"], torch.device("cpu")
    )  # .cpu()
    q = [
        "transformer.blocks." + str(s) + "."
        for s in configs["pretrain_model"]["add_lora"]
    ]
    model = models.fixParameters(model, unfix=configs["pretrain_model"]["unfix_layers"])
    model = models.addlora(
        model,
        layers=q,
        ranks=configs["pretrain_model"]["rank"],
        alphas=configs["pretrain_model"]["alpha"],
        # dtype=torch.bfloat16,
    )
    return model


LOAD_PRETRAIN = {
    "esm2": loadesm2,
    "esm3": loadesm3,
    "esmc_600m": loadesmc,
    "esmc_300m": loadesmc,
}


def loadPretrainModel(configs) -> nn.Module:
    if "pretrain_model" in configs:
        model = "esm2"
        if "model" in configs["pretrain_model"]:
            model = configs["pretrain_model"]["model"]
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

        name = os.path.basename(path)
        data = pickle.load(f)
        for j in data:
            j["origin"] = name
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
    aug = VirusDataset.DataAugmentation(
        step_points, maskp, crop, lens, tracks=configs["dataset"]["tracks"]
    )

    ds1 = VirusDataset.ESM3MultiTrackDataset(
        data1, data2, label, augment=aug, tracks=configs["dataset"]["tracks"]
    )
    ds2 = VirusDataset.ESM3MultiTrackDatasetTEST(
        data2, tracks=configs["dataset"]["tracks"]
    )

    ds = VirusDataset.ESM3datamodule(ds1, ds2)
    return ds


def loadBalancedDatasetesm3args(configs):
    import VirusDataset

    pos_datasets = []
    neg_datasets = []
    test_datasets = []
    lens = []

    sample_size = configs["dataset"]["dataset_train_sample"]
    assert len(configs["dataset"]["pos"]) == len(sample_size[0])
    assert len(configs["dataset"]["neg"]) == len(sample_size[1])

    sample_size = configs["dataset"]["dataset_val_sample"]
    assert len(configs["dataset"]["pos"]) == len(sample_size[0])
    assert len(configs["dataset"]["neg"]) == len(sample_size[1])

    for i in configs["dataset"]["pos"]:
        data = loadPickle(i)
        pos_datasets.append(data)

    for i in configs["dataset"]["neg"]:
        data = loadPickle(i)
        neg_datasets.append(data)

    for i in configs["dataset"]["test"]:
        data = loadPickle(i)
        for j in data:
            lens.append(len(j["ori_seq"]))
        test_datasets.extend(data)

    step_points = configs["augmentation"]["step_points"]
    crop = configs["augmentation"]["crop"]
    maskp = [
        (i, j)
        for i, j in zip(
            configs["augmentation"]["maskp"], configs["augmentation"]["maskpc"]
        )
    ]
    tracks = configs["augmentation"]["tracks"]

    aug = VirusDataset.DataAugmentation(step_points, maskp, crop, lens, tracks)

    if "required_labels" not in configs["dataset"]:
        configs["dataset"]["required_labels"] = []

    args = [
        pos_datasets,
        neg_datasets,
        test_datasets,
    ]

    argv = {
        "batch_size": configs["train"]["batch_size"],
        "pos_neg_train": configs["dataset"]["dataset_train_sample"],
        "pos_neg_val": configs["dataset"]["dataset_val_sample"],
        "train_test_ratio": configs["dataset"]["train_test_ratio"],
        "aug": aug,
        "tracks": configs["dataset"]["tracks"],
        "required_labels": configs["dataset"]["required_labels"],
    }

    return args, argv


def loadBalancedDatasetesm3(configs):
    import VirusDataset

    args, argv = loadBalancedDatasetesm3args(configs)
    ds = VirusDataset.ESM3BalancedDataModule(*args, **argv)
    return ds


def loadBalancedDatasetesm3activelearning(configs):
    import VirusDataset

    active_learning_list = []
    for i in configs["active_learning"]["datasets"]:
        data = loadPickle(i)
        active_learning_list.append(data)

    args, argv = loadBalancedDatasetesm3args(configs)
    ds = VirusDataset.ESM3BalancedDataModuleActiveLearning(
        active_learning_list, *args, **argv
    )
    return ds


def fixModelForActiveLearning(model):
    model = models.fixParameters(model, unfix=["clf"], fix=["additional_clf"])
    return model


def loadActiveLearningWeights(model, path):
    t = torch.load(path, map_location="cpu")
    model.load_state_dict(t["state_dict"], strict=False)
    return model


LOAD_DATASET = {
    "esm2": loadDatasetesm2,
    "esm3": loadDatasetesm3,
    "balancedesm3": loadBalancedDatasetesm3,
    "balancedesm3activelearning": loadBalancedDatasetesm3activelearning,
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
        weight_decay=configs["model"]["weight_decay"],
    )
    return clsmodel


def buildesm2Model(configs, model):
    clsmodel = models.IonclfESM2(
        model,
        step_lambda=configs["model"]["lambda_adapt"],
        lamb=configs["model"]["lambda_ini"],
        max_lambda=configs["model"]["max_lambda"],
        step=configs["model"]["lambda_step"],
        p=configs["model"]["dropout"],
        thres=configs["model"]["lambda_thres"],
        lr=configs["model"]["lr"],
        weight_decay=configs["model"]["weight_decay"],
    )
    return clsmodel


def buildesm3Model(configs, model):
    if "clf_params" not in configs["model"]:
        configs["model"]["clf_params"] = {}
    if "dis_params" not in configs["model"]:
        configs["model"]["dis_params"] = {}

    if "additional_label_weights" not in configs["model"]:
        configs["model"]["additional_label_weights"] = []

    assert len(configs["model"]["additional_label_weights"]) == len(
        configs["dataset"]["required_labels"]
    )
    if "pos_weights" not in configs["model"]:
        configs["model"]["pos_weights"] = None

    if "lr_backbone" not in configs["model"]:
        configs["model"]["lr_backbone"] = None

    if "more params" not in configs["model"]:
        configs["model"]["more params"] = {}

    clsmodel = models.IonclfESM3(
        model,
        step_lambda=configs["model"]["lambda_adapt"],
        lamb=configs["model"]["lambda_ini"],
        max_lambda=configs["model"]["max_lambda"],
        step=configs["model"]["lambda_step"],
        p=configs["model"]["dropout"],
        thres=configs["model"]["lambda_thres"],
        lr=configs["model"]["lr"],
        lr_backbone=configs["model"]["lr_backbone"],
        clf=configs["model"]["clf"],
        clf_params=configs["model"]["clf_params"],
        dis=configs["model"]["dis"],
        dis_params=configs["model"]["dis_params"],
        weight_decay=configs["model"]["weight_decay"],
        addition_label_weights=configs["model"]["additional_label_weights"],
    )
    return clsmodel


def buildesm3cModel(configs, model):
    if "clf_params" not in configs["model"]:
        configs["model"]["clf_params"] = {}
    if "dis_params" not in configs["model"]:
        configs["model"]["dis_params"] = {}

    if "addition_clf" not in configs["model"]:
        configs["model"]["addition_clf"] = None
        configs["model"]["addition_clf_params"] = None
    elif "addition_clf_params" not in configs["model"]:
        configs["model"]["addition_clf_params"] = {}

    if "additional_label_weights" not in configs["model"]:
        configs["model"]["additional_label_weights"] = []

    if "pos_weights" not in configs["model"]:
        configs["model"]["pos_weights"] = None

    assert len(configs["model"]["additional_label_weights"]) == len(
        configs["dataset"]["required_labels"]
    )

    if "lr_backbone" not in configs["model"]:
        configs["model"]["lr_backbone"] = None

    if "weight_max" not in configs["model"]:
        configs["model"]["weight_max"]

    if "weight_step" not in configs["model"]:
        configs["model"]["weight_step"] = 1

    if "more params" not in configs["model"]:
        configs["model"]["more params"] = {}

    clsmodel = models.IonclfESMC(
        model,
        step_lambda=configs["model"]["lambda_adapt"],
        lamb=configs["model"]["lambda_ini"],
        embed_dim=configs["model"]["embed_dim"],
        max_lambda=configs["model"]["max_lambda"],
        step=configs["model"]["lambda_step"],
        thres=configs["model"]["lambda_thres"],
        lr=configs["model"]["lr"],
        lr_backbone=configs["model"]["lr_backbone"],
        clf=configs["model"]["clf"],
        clf_params=configs["model"]["clf_params"],
        addition_clf=configs["model"]["addition_clf"],
        addition_clf_params=configs["model"]["addition_clf_params"],
        dis=configs["model"]["dis"],
        dis_params=configs["model"]["dis_params"],
        weight_decay=configs["model"]["weight_decay"],
        addition_label_weights=configs["model"]["additional_label_weights"],
        weight_max=configs["model"]["weight_max"],
        weight_step=configs["model"]["weight_step"],
        **configs["model"]["more params"]
    )
    return clsmodel


BUILD_MODEL = {
    "simple": buildSimpleModel,
    "esm2": buildesm2Model,
    "esm3": buildesm3Model,
    "esmc": buildesm3cModel,
}

MODEL_CLS = {
    "simple": models.IonclfBaseline,
    "esm3": models.IonclfESM3,
    "esm2": models.IonclfESM2,
}


def buildModel(
    configs, basemodel=None, checkpoint=None
) -> pytorch_lightning.LightningModule:

    if "active_learning" in configs:
        checkpoint = configs["active_learning"]["checkpoint"]

    model = "esm2"
    if "type" in configs["model"]:
        model = configs["model"]["type"]

    if model in BUILD_MODEL:
        model = BUILD_MODEL[model](configs, basemodel)
    else:
        raise NotImplementedError

    if checkpoint is not None:
        t = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(t["state_dict"], strict=False)
        gs = t["global_step"]
        if "unfreeze" in configs["pretrain_model"]:
            t = configs["pretrain_model"]["unfreeze"]["steps"]
            idx = np.argsort(t)
            idx = filter(lambda x: t[x] < gs, idx)
            model.load_freeze = [
                configs["pretrain_model"]["unfreeze"]["layers"][i] for i in idx
            ]

    if (
        "active_learning" in configs
        and "checkpoint_additional" in configs["active_learning"]
        and len(configs["active_learning"]["checkpoint_additional"]) > 0
    ):
        model = loadActiveLearningWeights(
            model, configs["active_learning"]["checkpoint_additional"]
        )

    if "active_learning" in configs:
        print("fix model for active learning")
        model = fixModelForActiveLearning(model)

    return model


def buildTrainer(configs, args):
    # k = 2
    # if "save" in configs["train"]:
    #     k = configs["train"]["save"]

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="validate_acc",  # Replace with your validation metric
    #     mode="max",  # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)
    #     save_top_k=k,  # Save top k checkpoints based on the monitored metric
    #     save_last=True,  # Save the last checkpoint at the end of training
    #     dirpath=args.path,  # Directory where the checkpoints will be saved
    #     filename="{epoch}-{validate_acc:.2f}",  # Checkpoint file naming pattern
    # )

    cbs = callbacks.getCallbacks(configs, args)

    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #         "tb_logs/%s" % args.name
    #     ),
    # )
    logger = TensorBoardLogger("tb_logs", name=args.name)

    if args.strategy == "deep":
        args.strategy = pytorch_lightning.strategies.DeepSpeedStrategy()

    pytorch_lightning.seed_everything(configs["train"]["seed"])

    if "gradient_clip_val" not in configs["train"]:
        configs["train"]["gradient_clip_val"] = None

    trainer = pytorch_lightning.Trainer(
        strategy=args.strategy,
        logger=logger,
        accelerator="gpu",
        # profiler=profiler,
        devices=args.devices,
        max_epochs=configs["train"]["epoch"],
        log_every_n_steps=1,
        gradient_clip_val=configs["train"]["gradient_clip_val"],
        accumulate_grad_batches=configs["train"]["accumulate_grad_batches"],
        callbacks=cbs,
    )

    return trainer
