import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader

import testUtils
import trainUtils
import VirusDataset


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
        default=["seq_t"],
    )
    parser.add_argument("-n", "--num", type=int, default=None)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-b", "--bayes", type=int, default=-1)
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

    # if "active_learning" in configs:
    #     model = trainUtils.buildModel(
    #         configs, pretrain_model, configs["active_learning"]["checkpoint"]
    #     )
    #     model = trainUtils.fixModelForActiveLearning(model)
    #     model = trainUtils.loadActiveLearningWeights(
    #         model, os.path.join(args.path, args.checkpoint)
    #     )
    # else:
    model = trainUtils.buildModel(
        configs,
        pretrain_model,
        os.path.join(args.path, args.checkpoint),
    )

    if args.bayes != -1:
        model.bayes_predict = args.bayes

    checkpoint_basename = os.path.basename(args.checkpoint)
    checkpoint_basename = checkpoint_basename[: checkpoint_basename.rfind(".")]
    dataset_basename = os.path.basename(args.input)
    dataset_basename = dataset_basename[: dataset_basename.rfind(".")]

    print("load dataset")
    testdata = trainUtils.loadPickle(args.input)
    test_set = VirusDataset.ESM3MultiTrackDatasetTEST(
        testdata, tracks=args.tracks, trunc=args.num
    )
    dl = DataLoader(test_set, batch_size=1, shuffle=False)

    trainer = L.Trainer(accelerator="gpu", devices=args.devices)
    res = trainer.predict(model, dl)

    # print(res)
    L.seed_everything(configs["train"]["seed"])
    df, pre = testUtils.resultDataframe(res, configs)
    # if isinstance(res[0], tuple):
    #     res = [r[0] for r in res]
    # pre = torch.stack(res).numpy()

    plt.hist(pre)
    plt.savefig(
        os.path.join(args.output, "%s_%s.png" % (dataset_basename, checkpoint_basename))
    )

    df.to_csv(
        os.path.join(
            args.output, "%s_%s.csv" % (dataset_basename, checkpoint_basename)
        ),
        index=False,
    )

    # np.savetxt(
    #     os.path.join(
    #         args.output, "%s_%s.txt" % (dataset_basename, checkpoint_basename)
    #     ),
    #     pre,
    # )


if __name__ == "__main__":
    run()
