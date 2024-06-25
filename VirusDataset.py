import os

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS 
import ioutils 
import random

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import pytorch_lightning as L

def readVirusSequences(pos=None, trunc=1498, sample = 300, seed=1509):
    random.seed(seed)
    print("read positive samples")
    seqs = {}
    if pos is None:
        pos = os.listdir("/home/tyfei/datasets/ion_channel/Interprot/ion_channel/0.99")
    for i in pos:
        # print(i)
        try:
            if i.endswith(".fas"): 
                # print(i, i[:i.find(".")] in df["Accession"].values)
                gen = ioutils.readFasta("/home/tyfei/datasets/ion_channel/Interprot/ion_channel/0.99/"+i, truclength=trunc)
                seqs[i[:i.find(".")]] = [i for i in gen]
                # print(i, "success") 
        except:
            pass
            # print(i, "failed")

    sequences = [] 
    labels = [] 
    for i in seqs:
        sampled = random.sample(seqs[i], min(sample, len(seqs[i])))
        sequences.extend(sampled)
        labels.extend([1]*len(sampled))

    print("read negative samples")
    gen = ioutils.readFasta("/home/tyfei/datasets/ion_channel/Interprot/Negative_sample/decoy_1m_new.fasta", truclength=trunc)
    seqs["neg"] = [i for i in gen]
    sampled = random.sample(seqs["neg"], min(len(labels), len(seqs["neg"])))
    sequences.extend(sampled)
    labels.extend([0]*len(sampled))

    print("read virus sequences")
    allvirus = []
    for i in os.listdir("/home/tyfei/datasets/NCBI_virus/genbank_csv/"):
        allvirus.extend(ioutils.readNCBICsv("/home/tyfei/datasets/NCBI_virus/genbank_csv/"+i, truclength=trunc))

    return sequences, labels, allvirus


class SeqDataset2(Dataset):
    def __init__(self, seq, label, seqtest):
        
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq).long()
        self.seq = seq

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label).long()
        self.label = label
        
        if not isinstance(seqtest, torch.Tensor):
            seqtest = torch.tensor(seqtest).long()
        self.seqtest = seqtest

        self.seqlen = seq.shape[0]
        self.seqtestlen = seqtest.shape[0]

    def __len__(self):
        return max(self.seqlen, self.seqtestlen)

    def __getitem__(self, idx):
        return self.seq[idx%self.seqlen], self.label[idx%self.seqlen], self.seqtest[idx%self.seqtestlen]

class TestDataset(Dataset):
    def __init__(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq).long()
        self.seq = seq

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx):
        
        return self.seq[idx]
    

class SeqDataset(Dataset):
    def __init__(self, seq, label):
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq).long()
        self.seq = seq

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label).long()
        self.label = label

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx):
        
        return self.seq[idx], self.label[idx]
    
class SeqdataModule(L.LightningDataModule):
    def __init__(self, trainset=None, testset=None, path="/home/tyfei/datasets/pts/virus", batch_size = 12, train_test_split=[0.8, 0.2], seed = 1509) -> None:
        super().__init__()
        
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.path = path 
        self.seed = seed
        
        if isinstance(testset, str):
            self.test_set = torch.load(testset)
        else:
            self.test_set = testset 
        
        if isinstance(trainset, str):
            self.trainset = torch.load(trainset)
        else:
            self.trainset = trainset

        # if self.trainset is not None:
        #     train_set, val_set = torch.utils.data.random_split(trainset, train_test_split)
        
        #     self.train_set = train_set  
        #     self.val_set = val_set 

    def saveDataset(self):
        torch.save(self.trainset, os.path.join(self.path, "train.pt"))
        torch.save(self.test_set, os.path.join(self.path, "test.pt"), self.test_set)
        
    
    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            if self.trainset is None:
                if os.path.exists(os.path.join(self.path, "train.pt")):
                    self.trainset = torch.load(os.path.join(self.path, "train.pt"))
                else:
                    raise FileExistsError
            
            if not hasattr(self, "train_set"):
                torch.manual_seed(self.seed)
                train_set, val_set = torch.utils.data.random_split(self.trainset, self.train_test_split)
                self.train_set = train_set  
                self.val_set = val_set 
        
        if stage == "predict":
            if self.test_set is None:
                if os.path.exists(os.path.join(self.path, "test.pt")):
                    self.test_set = torch.load(os.path.join(self.path, "test.pt"))
                else:
                    raise FileExistsError
                
        if stage == "test":
            raise NotImplementedError


        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
    