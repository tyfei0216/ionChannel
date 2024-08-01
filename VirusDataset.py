import os
import random

import numpy as np
import pytorch_lightning as L
import torch
from esm.utils.constants import esm3 as C
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import ioutils


class MyDataLoader(DataLoader):
    def __init__(self, ds, step_ds, *args, **kwargs):
        super().__init__(ds, *args, **kwargs)
        self.ds = ds
        self.epoch = 0
        self.step_ds = step_ds

    def step(self):
        if self.step_ds:
            self.ds.step()

    def __iter__(self):
        self.epoch += 1
        if self.step_ds:
            self.ds.step()
            self.ds.ifaug = True
        else:
            self.ds.ifaug = False
        print("now epoch ", self.epoch)
        return super().__iter__()


def readVirusSequences(pos=None, trunc=1498, sample=300, seed=1509):
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
                gen = ioutils.readFasta(
                    "/home/tyfei/datasets/ion_channel/Interprot/ion_channel/0.99/" + i,
                    truclength=trunc,
                )
                seqs[i[: i.find(".")]] = [i for i in gen]
                # print(i, "success")
        except:
            pass
            # print(i, "failed")

    sequences = []
    labels = []
    for i in seqs:
        sampled = random.sample(seqs[i], min(sample, len(seqs[i])))
        sequences.extend(sampled)
        labels.extend([1] * len(sampled))

    print("read negative samples")
    gen = ioutils.readFasta(
        "/home/tyfei/datasets/ion_channel/Interprot/Negative_sample/old/decoy_1m_new_rmdup.fasta",
        truclength=trunc,
    )
    seqs["neg"] = [i for i in gen]
    sampled = random.sample(seqs["neg"], min(len(labels), len(seqs["neg"])))
    sequences.extend(sampled)
    labels.extend([0] * len(sampled))

    print("read virus sequences")
    allvirus = []
    for i in os.listdir("/home/tyfei/datasets/NCBI_virus/genbank_csv/"):
        try:
            allvirus.extend(
                ioutils.readNCBICsv(
                    "/home/tyfei/datasets/NCBI_virus/genbank_csv/" + i, truclength=trunc
                )
            )
        except Exception:
            pass

    return sequences, labels, allvirus


MIN_LENGTH = 50


class DataAugmentation:
    def __init__(
        self, step_points: list, maskp: list, crop: list, croprange: list
    ) -> None:
        assert len(step_points) == len(maskp)
        assert len(maskp) == len(crop)
        self.step_points = step_points
        self.maskp = maskp
        self.crop = crop
        self.croprange = croprange

    def _getSettings(self, step):
        maskp = (-1.0, -1.0)
        crop = -1.0
        for i in range(len(self.step_points)):
            if step > self.step_points[i]:
                maskp = self.maskp[i]
                crop = self.crop[i]
        return maskp, crop

    def getAugmentation(self, seqlen, step):
        maskp, crop = self._getSettings(step)
        if crop > 0:
            t = random.random()
            if t < crop:
                sampledlen = random.sample(self.croprange, 1)[0]
                sampledlen = int(sampledlen * np.random.uniform(0.8, 1.2))
                sampledlen = MIN_LENGTH if sampledlen < MIN_LENGTH else sampledlen
                sampledlen = min(sampledlen, seqlen - 2)
                return maskp, sampledlen
        return maskp, -1


class ESM3BaseDataset(Dataset):
    def __init__(self, tracks=["seq_t", "structure_t", "sasa_t", "second_t"]) -> None:
        assert len(tracks) > 0
        self.tracks = tracks
        self.step_cnt = 0

    def step(self):
        self.step_cnt += 1

    def resetCnt(self):
        self.step_cnt = 0

    def getToken(self, track, token):
        # assert token in ["start", "end", "mask"]
        match token:
            case "start":
                match track:
                    case "seq_t":
                        return C.SEQUENCE_BOS_TOKEN
                    case "structure_t":
                        return C.STRUCTURE_BOS_TOKEN
                    case "sasa_t":
                        return 0
                    case "second_t":
                        return 0
                    case _:
                        raise ValueError
            case "end":
                match track:
                    case "seq_t":
                        return C.SEQUENCE_EOS_TOKEN
                    case "structure_t":
                        return C.STRUCTURE_EOS_TOKEN
                    case "sasa_t":
                        return 0
                    case "second_t":
                        return 0
                    case _:
                        raise ValueError
            case "mask":
                match track:
                    case "seq_t":
                        return C.SEQUENCE_MASK_TOKEN
                    case "structure_t":
                        return C.STRUCTURE_MASK_TOKEN
                    case "sasa_t":
                        return C.SASA_UNK_TOKEN
                    case "second_t":
                        return C.SS8_UNK_TOKEN
                    case _:
                        raise ValueError
            case "pad":
                match track:
                    case "seq_t":
                        return C.SEQUENCE_PAD_TOKEN
                    case "structure_t":
                        return C.STRUCTURE_PAD_TOKEN
                    case "sasa_t":
                        return C.SASA_PAD_TOKEN
                    case "second_t":
                        return C.SS8_PAD_TOKEN
                    case _:
                        raise ValueError
            case _:
                raise ValueError

    def _maskSequence(self, sample, pos):
        for i in sample:
            sample[i][pos] = self.getToken(i, "mask")

        return sample

    def _generateMaskingPos(self, num, length, method="point"):
        assert length > num + 5
        if method == "point":
            a = np.array(random.sample(range(length - 2), num)) + 1
            return a
        elif method == "block":
            s = random.randint(1, length - num)
            a = np.array(range(s, s + num))
            return a
        else:
            raise NotImplementedError

    def _cropSequence(self, sample, start, end):
        for i in sample:
            t = torch.zeros((end - start + 2), dtype=torch.long)
            t[1:-1] = torch.tensor(sample[i][start:end])
            t[0] = self.getToken(i, "start")
            t[-1] = self.getToken(i, "end")
            sample[i] = t
        return sample

    def _augmentsample(self, sample, maskp, crop):
        samplelen = len(sample[self.tracks[0]])
        if crop > 50:
            s = random.randint(1, samplelen - crop - 1)
            sample = self._cropSequence(sample, s, s + crop)
            samplelen = crop + 2
        if maskp[0] > 0:
            num = np.random.binomial(samplelen - 2, maskp[0])
            pos = self._generateMaskingPos(num, samplelen)
            if len(pos) > 0:
                sample = self._maskSequence(sample, pos)
        if maskp[1] > 0:
            num = np.random.binomial(samplelen - 2, maskp[0])
            pos = self._generateMaskingPos(num, samplelen, "block")
            if len(pos) > 0:
                sample = self._maskSequence(sample, pos)
        return sample


class ESM3MultiTrackDataset(ESM3BaseDataset):
    def __init__(
        self,
        data1,
        data2,
        label,
        augment: DataAugmentation = None,
        tracks=["seq_t", "structure_t", "sasa_t", "second_t"],
    ) -> None:
        super().__init__(tracks=tracks)
        self.data1 = data1
        self.data2 = data2
        self.label = label
        self.aug = augment
        self.iters = 0
        self.data2order = np.arange(len(data2))
        random.shuffle(self.data2order)
        self.ifaug = False
        # self.tracks = tracks

    def __len__(self):
        return len(self.data1)

    def step(self):
        random.shuffle(self.data2order)
        super().step()

    def __getitem__(self, idx):
        x1 = {}
        x2 = {}
        for i in self.tracks:
            x1[i] = self.data1[idx][i]
            x2[i] = self.data2[self.data2order[idx % len(self.data2)]][i]
        if self.aug is not None and self.ifaug:
            maskp, crop = self.aug.getAugmentation(
                len(x1[self.tracks[0]]), self.step_cnt
            )
            x1 = self._augmentsample(x1, maskp, crop)
        return x1, torch.tensor([self.label[idx]]), x2


class ESM3MultiTrackDatasetTEST(ESM3BaseDataset):
    def __init__(
        self,
        data1,
        augment: DataAugmentation = None,
        tracks=["seq_t", "structure_t", "sasa_t", "second_t"],
    ) -> None:
        super().__init__(tracks=tracks)
        self.data1 = data1
        self.aug = augment
        # self.tracks = tracks

    def __len__(self):
        return len(self.data1)

    def step(self):
        super().step()

    def __getitem__(self, idx):
        x1 = {}
        for i in self.tracks:
            x1[i] = self.data1[idx][i]
        if self.aug is not None:
            maskp, crop = self.aug.getAugmentation(
                len(x1[self.tracks[0]]), self.step_cnt
            )
            x1 = self._augmentsample(x1, maskp, crop)
        return x1


class ESM3datamodule(L.LightningDataModule):
    def __init__(
        self,
        ds1: ESM3BaseDataset,
        ds2: ESM3BaseDataset,
        batch_size=1,
        train_test_split=[0.85, 0.15],
        seed=1509,
    ):
        super().__init__()
        self.value = 0
        # self.ds1 = ds1
        # self.ds2 = ds2
        self.batch_size = batch_size
        self.seed = seed
        torch.manual_seed(self.seed)
        # train_set, val_set = torch.utils.data.random_split(ds1, train_test_split)
        all_indices = np.arange(len(ds1))

        self.trainval_set = ds1
        self.train_indices = all_indices[: int(len(all_indices) * train_test_split[0])]
        self.val_indices = all_indices[int(len(all_indices) * train_test_split[0]) :]
        self.testset = ds2

    def train_dataloader(self):
        self.value += 1
        self.trainval_set.resetCnt()
        print("get train loader")
        return MyDataLoader(
            self.trainval_set,
            True,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.train_indices),
            num_workers=4,
        )

    def val_dataloader(self):
        self.value += 1
        print("get val loader")
        return MyDataLoader(
            self.trainval_set,
            False,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.val_indices),
            num_workers=4,
        )

    def predict_dataloader(self):
        self.value += 1
        print("get predict loader")
        return MyDataLoader(
            self.testset, False, batch_size=self.batch_size, shuffle=True, num_workers=4
        )


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
        return (
            self.seq[idx % self.seqlen],
            self.label[idx % self.seqlen],
            self.seqtest[idx % self.seqtestlen],
        )


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
    def __init__(
        self,
        trainset=None,
        testset=None,
        path="/home/tyfei/datasets/pts/virus",
        batch_size=12,
        train_test_split=[0.8, 0.2],
        seed=1509,
    ) -> None:
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

    def saveDataset(self, path=None):
        if path is not None:
            self.path = path
        torch.save(self.trainset, os.path.join(self.path, "train.pt"))
        torch.save(self.test_set, os.path.join(self.path, "test.pt"))

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            if self.trainset is None:
                if os.path.exists(os.path.join(self.path, "train.pt")):
                    self.trainset = torch.load(os.path.join(self.path, "train.pt"))
                else:
                    raise FileExistsError

            if not hasattr(self, "train_set"):
                torch.manual_seed(self.seed)
                train_set, val_set = torch.utils.data.random_split(
                    self.trainset, self.train_test_split
                )
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

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
