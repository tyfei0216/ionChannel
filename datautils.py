import os
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from datetime import date
import pytorch_lightning as L 
import re
import pytorch_lightning as L 
tqdm.pandas()

def parseline(line):
    try:
        line = line.decode("utf-8").strip()
    except:
        return None 
    if line.startswith(">"):
        return line 
    line = re.sub("[^a-z,A-Z]", "", line) 
    return line 

# inspired by esm.data.readfasta
def readFasta(path, to_upper=True, truclength=1500):
    with open(path, "rb") as f:
        line = None
        seq = []  
        for t in f:
            
            t = parseline(t)
            if t is None:
                line = None 
                continue
            if line is None:
                if not t.startswith(">"):
                    continue
                else:
                    line = t 
                    continue 
            if t.startswith(">"):
                s = "".join(seq)
                seq = [] 
                if len(s) > 0:
                    if to_upper:
                        s = s.upper() 
                    if truclength is not None:
                        s = s[:truclength]
                    yield line[1:], s
                line = t
            
            seq.append(t)


class SeqDataset2(Dataset):
    def __init__(self, seq, label, seqtest):
        self.seq = torch.tensor(seq).long()
        self.label = torch.tensor(label).long()
        self.seqtest = torch.tensor(seqtest).long() 
        self.seqlen = seq.shape[0]
        self.seqtestlen = seqtest.shape[0]

    def __len__(self):
        return max(self.seqlen, self.seqtestlen)

    def __getitem__(self, idx):
        return self.seq[idx%self.seqlen], self.label[idx%self.seqlen], self.seqtest[idx%self.seqtestlen]

class TestDataset(Dataset):
    def __init__(self, seq):
        self.seq = torch.tensor(seq).long()

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx):
        
        return self.seq[idx]
    

class SeqDataset(Dataset):
    def __init__(self, seq, label):
        self.seq = torch.tensor(seq).long()
        self.label = torch.tensor(label).long()

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx):
        
        return self.seq[idx], self.label[idx]
    
class SeqdataModule(L.LightningDataModule):
    def __init__(self, trainset, testset, batch_size = 32) -> None:
        super().__init__()
        train_set, val_set = torch.utils.data.random_split(trainset, [0.8, 0.2])
        self.train_set = train_set 
        self.test_set = testset 
        self.val_set = val_set
        self.batch_size = batch_size
    # def setup(self, ):
    #     pass 

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)