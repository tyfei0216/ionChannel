import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader

import trainUtils
import VirusDataset
import pickle

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-d", "--devices", type=int, nargs="+", default=[0])
    parser.add_argument("-c", "--checkpoint", type=str, default="last.ckpt")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument(
        "-t",
        "--tracks",
        type=str,
        nargs="+",
        default=["seq_t", "structure_t", "sasa_t", "second_t"],
    )
    parser.add_argument("-i", "--input", type=str, required=True)
    args = parser.parse_args()
    return args


def run():
    args = parseArgs()
    assert os.path.isdir(args.path)
    if args.output == None:
        args.output = args.path
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(os.path.join(args.path, "config.json"), "r") as f:
        configs = json.load(f)
    print("load model")
    pretrain_model = trainUtils.loadPretrainModel(configs)
    model = trainUtils.buildModel(
        configs,
        pretrain_model,
        os.path.join(args.path, args.checkpoint),
    )
    checkpoint_basename = os.path.basename(args.checkpoint)
    checkpoint_basename = checkpoint_basename[: checkpoint_basename.find(".")]
    dataset_basename = os.path.basename(args.input)
    dataset_basename = dataset_basename[: dataset_basename.find(".")]

    print("load dataset")
    with open(args.input,"rb") as f:
        testdata = pickle.load(f)

    test_set = VirusDataset.ESM2DatasetTEST(testdata)
    dl = DataLoader(test_set, batch_size=1, shuffle=False)

    trainer = L.Trainer(accelerator="gpu", devices=args.devices)
    res = trainer.predict(model, dl)
    pre = torch.stack(res).numpy()

    plt.hist(pre)
    plt.savefig(
        os.path.join(args.output, "%s_%s.png" % (dataset_basename, checkpoint_basename))
    )

    np.savetxt(
        os.path.join(
            args.output, "%s_%s.txt" % (dataset_basename, checkpoint_basename)
        ),
        res,
    )


if __name__ == "__main__":
    run()
