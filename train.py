import argparse
import json
import os

import torch

import trainUtils


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-d", "--devices", type=int, nargs="+", default=[0])
    parser.add_argument("-s", "--strategy", type=str, default="auto")
    parser.add_argument("-n", "--name", type=str, default="ion_test")
    args = parser.parse_args()
    return args


def run():
    args = parseArgs()
    path = args.path
    with open(os.path.join(path, "config.json"), "r") as f:
        configs = json.load(f)
    json_formatted_str = json.dumps(configs, indent=2)
    print("fetch config from ", os.path.join(path, "config.json"))
    print("--------")
    print("config: ")
    print(json_formatted_str)
    print("--------")
    print("using devices ", args.devices)
    print("using strategy ", args.strategy)
    print("--------")
    print("build trainer")
    trainer = trainUtils.buildTrainer(configs, args)
    print("load pretrain model")
    pretrain_model = trainUtils.loadPretrainModel(configs)
    print("build finetune model")
    model = trainUtils.buildModel(configs, pretrain_model)
    print("load dataset")
    ds = trainUtils.loadDataset(configs)
    print("start training")

    trainer.fit(model, ds)
    torch.save(model.state_dict(), path + "parms.pt")


if __name__ == "__main__":
    run()