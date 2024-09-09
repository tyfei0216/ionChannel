import argparse
import os
import pickle
import random
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig
from tqdm import tqdm

sys.path.append("/home/tyfei/ionChannel")
import ioutils


def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="esm3_sm_open_v1")
    parser.add_argument("-f", "--fasta", type=str, required=True)
    parser.add_argument("-rl", "--lengths", type=int, default=30000)
    # parser.add_argument("-rh", "--rangeh", type=int, default=800)
    parser.add_argument("-s", "--save", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1509)
    parser.add_argument("--lengthThres", type=int, default=1498)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)

    args = parser.parse_args()
    return args


def main():
    args = ParseArgs()
    if args.model == "esm3_sm_open_v1":
        model = ESM3.from_pretrained("esm3_sm_open_v1")
    else:
        raise NotImplementedError

    print("runing on %s" % args.device)
    model = model.to(args.device)

    seqs = ioutils.readFasta(args.fasta)
    seqss = [i for i in seqs]
    lens = []
    for i in seqss:
        lens.append(len(i[1]))

    allres = []
    with torch.no_grad():
        for count in range(args.lengths):
            sampledlen = random.sample(lens, 1)[0]
            sampledlen = int(sampledlen * np.random.uniform(0.8, 1.2))
            if sampledlen > args.lengthThres:
                sampledlen = args.lengthThres

            torch.manual_seed(args.seed)
            protein = ESMProtein(sequence="_" * sampledlen)
            protein = model.generate(
                protein,
                GenerationConfig(track="sequence", num_steps=args.steps),
            )
            protein = model.generate(
                protein,
                GenerationConfig(track="secondary_structure", num_steps=args.steps),
            )
            protein = model.generate(
                protein, GenerationConfig(track="sasa", num_steps=args.steps)
            )
            protein = model.generate(
                protein, GenerationConfig(track="structure", num_steps=args.steps)
            )

            res = model.encode(protein)
            data = {}
            data["randomseed"] = args.seed
            data["model"] = args.model
            data["steps"] = args.steps
            data["ori_seq"] = protein.sequence
            data["seq_t"] = res.sequence.cpu().numpy()
            data["structure_t"] = res.structure.cpu().numpy()
            data["second_t"] = res.secondary_structure.cpu().numpy()
            data["sasa_t"] = res.sasa.cpu().numpy()
            data["coordinates"] = res.coordinates.cpu().numpy()
            allres.append(data)

    with open(args.save, "wb") as f:
        pickle.dump(allres, f)


if __name__ == "__main__":
    main()
