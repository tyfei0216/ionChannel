import argparse
import os
import pickle
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
    parser.add_argument("-s", "--save", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1509)
    parser.add_argument("--lengthThres", type=int, default=2000)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=16)

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

    allres = []
    with torch.no_grad():
        for i, seq in zip(tqdm(range(len(seqss))), seqss):
            try:
                if len(seq[1]) > args.lengthThres:
                    continue
                torch.manual_seed(args.seed)
                protein = ESMProtein(sequence=seq[1])
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
                data["id"] = seq[0]
                data["ori_seq"] = seq[1]
                data["seq_t"] = res.sequence.cpu().numpy()
                data["structure_t"] = res.structure.cpu().numpy()
                data["second_t"] = res.secondary_structure.cpu().numpy()
                data["sasa_t"] = res.sasa.cpu().numpy()
                data["coordinates"] = res.coordinates.cpu().numpy()
                allres.append(data)
            except:
                pass
    with open(args.save, "wb") as f:
        pickle.dump(allres, f)


if __name__ == "__main__":
    main()
