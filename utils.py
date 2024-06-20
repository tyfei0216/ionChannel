import torch
import esm
import random
from random import sample 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def readDataset(path, samplenum=None, trunc=None, seed=1234):
    random.seed(seed)
    seqs = esm.data.read_fasta(path)
    seqs = [i for i in seqs]
    if samplenum is not None:
        seqs = sample(seqs, min(samplenum, len(seqs)))
    if trunc is not None:
        trunc_seq = [] 
        for i, seq in seqs:
            if len(seq) > trunc:
                trunc_seq.append((i, seq[:trunc]))
            else:
                trunc_seq.append((i, seq))
        return trunc_seq 
    return seqs

class SeqDataset(Dataset):
    def __init__(self, seq, label):
        self.seq = torch.tensor(seq).long()
        self.label = torch.tensor(label).long()

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx):
        
        return self.seq[idx], self.label[idx]
    
def splitDataset(ds, split = [0.8, 0.2], seed = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(ds, split) 
    trainloader = DataLoader(train_set, batch_size=16, shuffle=True)
    valloader = DataLoader(val_set, batch_size=16, shuffle=True) 
    return trainloader, valloader