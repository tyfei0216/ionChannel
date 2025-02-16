import os
import random

import numpy as np
import pytorch_lightning as L
import torch
from esm.utils.constants import esm3 as C
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import ioutils

MIN_LENGTH = 50


class MyDataLoader(DataLoader):
    def __init__(self, ds, step_ds, **kwargs):
        # print(kwargs)
        super().__init__(dataset=ds, **kwargs)
        # self.ds = ds
        self.epoch = 0
        self.step_ds = step_ds

    def step(self):
        # print("step dataset")
        if self.step_ds:
            self.dataset.step()

    def __iter__(self):
        self.epoch += 1
        self.dataset.newEpoch()
        if self.step_ds:
            # self.ds.step()
            self.dataset.ifaug = True
        else:
            self.dataset.ifaug = False
        # print("now epoch ", self.epoch)
        return super().__iter__()


class DataAugmentation:
    def __init__(
        self,
        step_points: list,
        maskp: list,
        crop: list,
        croprange: list,
        maskpp: list = None,
        tracks: list = None,
    ) -> None:
        assert len(step_points) == len(maskp)
        assert len(maskp) == len(crop)
        self.tracks = tracks
        if self.tracks == None:
            self.tracks = {"seq_t": 1, "structure_t": 1, "sasa_t": 1, "second_t": 1}
        self.step_points = step_points
        self.maskp = maskp
        if maskpp is None:
            maskpp = [-1 for i in range(len(maskp))]
        self.maskpp = maskpp
        self.crop = crop
        self.croprange = croprange

    def _getSettings(self, step):
        maskp = (-1.0, -1.0)
        crop = -1.0
        maskpp = -1.0
        for i in range(len(self.step_points)):
            if step > self.step_points[i]:
                maskpp = self.maskpp[i]
                maskp = self.maskp[i]
                crop = self.crop[i]
        return maskp, crop, maskpp

    def getAugmentation(self, seqlen, step):
        maskp, crop, maskpp = self._getSettings(step)
        r = np.random.uniform(0.001, 0.999)
        if r < maskpp:
            maskp = (-1.0, -1.0)
        rettrack = {}
        flag = 0
        for i in self.tracks:

            r = np.random.uniform(0.001, 0.999)
            if r < self.tracks[i]:
                flag = 1
                rettrack[i] = True
            else:
                rettrack[i] = False

        if flag == 0:
            rettrack["seq_t"] = True
        if crop > 0:
            t = random.random()
            if t < crop:
                sampledlen = random.sample(self.croprange, 1)[0]
                sampledlen = int(sampledlen * np.random.uniform(0.8, 1.2))
                sampledlen = MIN_LENGTH if sampledlen < MIN_LENGTH else sampledlen
                sampledlen = min(sampledlen, seqlen - 2)
                return maskp, sampledlen, rettrack
        return maskp, -1, rettrack


class ESM3BaseDataset(Dataset):
    def __init__(
        self,
        tracks=["seq_t", "structure_t", "sasa_t", "second_t"],
        min_length=50,
        required_labels=[],
    ) -> None:
        assert len(tracks) > 0
        self.tracks = tracks
        self.step_cnt = 0
        self.min_length = min_length
        self.required_labels = required_labels

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
                        # return C.SASA_UNK_TOKEN
                        return 2
                    case "second_t":
                        # return C.SS8_UNK_TOKEN
                        return 2
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
                assert track == "seq_t"
                assert token in C.SEQUENCE_VOCAB
                return C.SEQUENCE_VOCAB.index(token)
                # raise ValueError

    def _maskSequence(self, sample, pos):
        for i in sample:
            if i in self.tracks:
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
            t = np.zeros((end - start + 2), dtype=int)
            t[1:-1] = sample[i][start:end]
            t[0] = self.getToken(i, "start")
            t[-1] = self.getToken(i, "end")
            sample[i] = t
        return sample

    def checkAndPickTmm(self, sample):
        if "Tmranges" not in sample:
            return None
        samplelen = len(sample[self.tracks[0]]) - 2
        tmranges = sample["Tmranges"]
        can_pick = []
        for i in tmranges:
            if i[1] < samplelen:
                can_pick.append(i)

        if len(can_pick) == 0:
            return None

        t = random.sample(can_pick, 1)[0]
        return t

    def _augmentsample(self, sample, maskp, crop, tracks=None):
        samplelen = len(sample[self.tracks[0]])
        ret = {}
        for i in self.tracks:
            ret[i] = sample[i].copy()
        if crop > self.min_length:
            t = self.checkAndPickTmm(sample)
            if t is None:
                s = random.randint(1, samplelen - crop - 1)
                ret = self._cropSequence(ret, s, s + crop)
                samplelen = len(ret[self.tracks[0]])
            else:
                if crop < (t[1] - t[0]):
                    crop = t[1] - t[0]

                s = random.randint(
                    max(1, t[0] - (crop - t[1] + t[0])), min(t[0], samplelen - crop - 1)
                )
                ret = self._cropSequence(ret, s, s + crop)
                samplelen = len(ret[self.tracks[0]])

        # assert samplelen > 45

        for i in self.tracks:
            ret["ori_" + i] = ret[i].copy()

        ret["mask"] = np.ones_like(ret[i], dtype=np.int32)

        if maskp[0] > 0 and samplelen > MIN_LENGTH:

            while True:
                num = np.random.binomial(samplelen - 2, maskp[0])
                if samplelen > num + 5:
                    break

            pos = self._generateMaskingPos(num, samplelen)
            if len(pos) > 0:
                ret = self._maskSequence(ret, pos)
                ret["mask"][pos] = 0

        if maskp[1] > 0 and samplelen > MIN_LENGTH:
            while True:
                num = np.random.binomial(samplelen - 2, maskp[0])
                if samplelen > num + 5:
                    break
            pos = self._generateMaskingPos(num, samplelen, "block")
            if len(pos) > 0:
                ret = self._maskSequence(ret, pos)
                ret["mask"][pos] = 0
        # if tracks is not None:
        #     for i in tracks:
        #         if not tracks[i]:
        #             ret.pop(i)

        # print(tracks)
        # print(sample)
        return ret

    def processSample(self, sample, _crop=None):
        if self.aug is not None and self.ifaug:
            maskp, crop, tracks = self.aug.getAugmentation(
                len(sample[self.tracks[0]]), self.step_cnt
            )
            if _crop is not None:
                crop = _crop
            x1 = self._augmentsample(sample, maskp, crop, tracks)
        else:
            x1 = {}
            for i in self.tracks:
                x1[i] = sample[i]
                x1["ori_" + i] = sample[i]
            x1["mask"] = np.ones_like(sample[i], dtype=np.int32)

        return x1

    def prepareLabels(self, sample, label):
        labels = [label]
        for i in self.required_labels:
            if "classes" in sample and i in sample["classes"]:
                labels.append(sample["classes"][i])
            else:
                labels.append(-1)
        return labels


class ESM3MultiTrackBalancedDataset(ESM3BaseDataset):

    def __init__(
        self,
        data1,
        data2,
        data3,
        augment: DataAugmentation = None,
        pos_neg_sample=None,
        tracks=["seq_t", "structure_t", "sasa_t", "second_t"],
        required_labels=[],
        shuffle=False,
        update_pnt=True,
    ) -> None:
        """_summary_

        Args:
            data1 (_type_): positive data
            data2 (_type_): negative data
            data3 (_type_): test data for domain adaptation
            augment (DataAugmentation, optional): _description_. Defaults to None.
            tracks (list, optional): _description_. Defaults to ["seq_t", "structure_t", "sasa_t", "second_t"].
        """
        super().__init__(tracks=tracks, required_labels=required_labels)
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.shuffle = shuffle
        self.update_pnt = update_pnt
        self.aug = augment
        self.iters = 0
        self.data1order = []
        self.data2order = []
        self.data3order = []  # np.arange(len(data3))
        self.pnts1 = [0 for i in data1]
        self.pnts2 = [0 for i in data2]
        self.pnt3 = [0 for i in data3]
        for i in data1:
            self.data1order.append(np.arange(len(i)))
        for i in data2:
            self.data2order.append(np.arange(len(i)))
        for i in data3:
            self.data3order.append(np.arange(len(i)))
        self.ifaug = False
        if pos_neg_sample is not None:
            self.pos_neg_sample = pos_neg_sample
            for i, j in zip(self.pos_neg_sample, [self.data1, self.data2]):
                for k in range(len(i)):
                    if i[k] < 0:
                        i[k] = len(j[k])
        else:
            self.pos_neg_sample = [[], []]
            for i in self.data1:
                self.pos_neg_sample[0].append(len(i))
            for i in self.data2:
                self.pos_neg_sample[1].append(len(i))

        self.shuffleIndex()

        self._sample = True
        # self.tracks = tracks

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, v: bool):
        assert isinstance(v, bool)
        self._sample = v

    def __len__(self):
        if self._sample:
            t = 0
            for i in self.pos_neg_sample:
                for j in i:
                    t += j
            return t
        else:
            t = 0
            for i in self.data1:
                t += len(i)
            for i in self.data2:
                t += len(i)
            return t

    def shuffleIndex(self):
        for i in [self.data1order, self.data2order]:
            for j in i:
                random.shuffle(j)
        random.shuffle(self.data3order)

    def newEpoch(self):
        print("called new epoch")
        if self.shuffle:
            self.shuffleIndex()
        if self.update_pnt:
            for i in range(len(self.pos_neg_sample[0])):
                self.pnts1[i] += self.pos_neg_sample[0][i]
                while self.pnts1[i] >= len(self.data1order[i]):
                    self.pnts1[i] -= len(self.data1order[i])
            for i in range(len(self.pos_neg_sample[1])):
                self.pnts2[i] += self.pos_neg_sample[1][i]
                while self.pnts2[i] >= len(self.data2order[i]):
                    self.pnts2[i] -= len(self.data2order[i])
            # while self.pnt3 >= len(self.data3order):
            #     self.pnt3 -= len(self.data3order)

    def getOrigin(self):
        ret = []
        for i in range(len(self)):
            t, _ = self._getitemx1(i)
            ret.append(t["origin"])
        return ret

    def _getitemx1(self, idx):
        if self.sample:
            for i in range(len(self.pos_neg_sample[0])):
                if idx - self.pos_neg_sample[0][i] < 0:
                    # print("before", self.pnts1, self.pnts1[i])
                    # self.pnts1[i] = self.pnts1[i] + 1
                    # print("after", self.pnts1, self.pnts1[i])
                    # if self.pnts1[i] > len(self.data1order[i]):
                    #     self.pnts1[i] = 1
                    return (
                        self.data1[i][
                            self.data1order[i][
                                (self.pnts1[i] + idx) % len(self.data1order[i])
                            ]
                        ],
                        1,
                    )
                else:
                    idx -= self.pos_neg_sample[0][i]
            for i in range(len(self.pos_neg_sample[1])):
                if idx - self.pos_neg_sample[1][i] < 0:
                    # self.pnts2[i] += 1
                    # if self.pnts2[i] > len(self.data2order[i]):
                    #     self.pnts2[i] = 1
                    return (
                        self.data2[i][
                            self.data2order[i][
                                (self.pnts2[i] + idx) % len(self.data2order[i])
                            ]
                        ],
                        0,
                    )
                else:
                    idx -= self.pos_neg_sample[1][i]
        else:
            for i in self.data1:
                if idx - len(i) < 0:
                    return i[idx], 1
                else:
                    idx -= len(i)
            for i in self.data2:
                if idx - len(i) < 0:
                    return i[idx], 0
                else:
                    idx -= len(i)
        raise KeyError

    def _getitemx2(self, idx):
        q = idx % len(self.data3)
        pickfrom = self.data3[q]
        ret = random.choice(pickfrom)
        return ret
        # self.pnt3 += 1
        # if self.pnt3 > len(self.data3order):
        #     self.pnt3 = 1
        return self.data3[self.data3order[(self.pnt3 + idx) % len(self.data3order)]]

    def __getitem__(self, idx):
        # x1 = {}
        x2 = {}
        t1, label = self._getitemx1(idx)
        t2 = self._getitemx2(idx)

        x1 = self.processSample(t1)

        x2 = self.processSample(t2, -1)

        # for i in self.tracks:
        #     x2[i] = t2[i]

        labels = self.prepareLabels(t1, label)

        return x1, torch.tensor(labels), x2


class ListESM3MultiTrackBalancedDataset(ESM3MultiTrackBalancedDataset):
    def __init__(
        self,
        active_learning_lists,
        list_size=50,
        list_require_mle=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.active_learning_lists = active_learning_lists
        self.list_size = list_size
        print("using list size", list_size)
        if list_require_mle is None:
            list_require_mle = [False for i in range(len(active_learning_lists))]
        self.list_require_mle = list_require_mle
        assert len(self.list_require_mle) == len(active_learning_lists)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        q = idx % len(self.active_learning_lists)
        need_mle = self.list_require_mle[q]
        t = min(self.list_size, len(self.active_learning_lists[q]))
        t = random.sample(range(len(self.active_learning_lists[q])), t)
        t = sorted(t)
        ret = [self.active_learning_lists[q][i] for i in t]

        return *res, (ret, need_mle)


class ListActiveLearningDataset(Dataset):
    def __init__(self, active_learning_lists, dataset, list_size=50):
        super().__init__()
        self.dataset = dataset
        self.active_learning_lists = active_learning_lists
        self.list_size = list_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        q = idx % len(self.active_learning_lists)
        # print(len(self.active_learning_lists[q]))
        t = min(self.list_size, len(self.active_learning_lists[q]))
        t = random.sample(range(len(self.active_learning_lists[q])), t)
        t = sorted(t)
        ret = [self.active_learning_lists[q][i] for i in t]
        return *self.dataset[idx], ret


class ESM3MultiTrackDataset(ESM3BaseDataset):
    def __init__(
        self,
        data1,
        data2,
        label,
        augment: DataAugmentation = None,
        tracks=["seq_t", "structure_t", "sasa_t", "second_t"],
        # origin = {0:""},
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

    def newEpoch(self):
        random.shuffle(self.data2order)

    # def step(self):
    # random.shuffle(self.data2order)
    # super().step()

    def __getitem__(self, idx):
        # x1 = {}
        x2 = {}

        for i in self.tracks:
            # x1[i] = self.data1[idx][i]
            x2[i] = self.data2[self.data2order[idx % len(self.data2)]][i]

        x1 = self.processSample(self.data1[idx])
        labels = self.prepareLabels(self.data1[idx], self.label[idx])
        return x1, torch.tensor(labels), x2


class ESM3MultiTrackDatasetTEST(ESM3BaseDataset):
    def __init__(
        self,
        data1,
        augment: DataAugmentation = None,
        tracks=["seq_t", "structure_t", "sasa_t", "second_t"],
        trunc=None,
    ) -> None:
        super().__init__(tracks=tracks)
        self.data1 = data1
        self.aug = augment
        self.trunc = trunc
        # self.tracks = tracks

    def __len__(self):
        if self.trunc is not None:
            return np.min(self.trunc, len(self.data1))
        return len(self.data1)

    def step(self):
        super().step()

    def __getitem__(self, idx):
        # x1 = {}
        # for i in self.tracks:
        #     x1[i] = self.data1[idx][i]
        # if self.aug is not None:
        #     maskp, crop = self.aug.getAugmentation(
        #         len(x1[self.tracks[0]]), self.step_cnt
        #     )
        #     x1 = self._augmentsample(x1, maskp, crop)
        x1 = self.processSample(self.data1[idx])
        return x1


class ESM3BalancedDataModule(L.LightningDataModule):
    def __init__(
        self,
        data1,
        data2,
        data3,
        batch_size=1,
        pos_neg_train=None,
        pos_neg_val=None,
        train_test_ratio=[0.85, 0.15],
        aug=None,
        seed=1509,
        tracks=["seq_t", "structure_t", "sasa_t", "second_t"],
        required_labels=[],
        train_lists=None,
        val_lists=None,
        train_list_require_mle=None,
        val_list_require_mle=None,
        list_size=50,
    ):
        super().__init__()
        self.value = 0
        self.batch_size = batch_size
        self.seed = seed
        self.data3 = data3
        self.required_labels = []

        from sklearn.model_selection import train_test_split

        self.traindata1 = []
        self.traindata2 = []
        self.train_indices1 = []
        self.train_indices2 = []
        self.valdata1 = []
        self.valdata2 = []
        self.val_indices1 = []
        self.val_indices2 = []
        for i in data1:
            d1, v1, i1, i2 = train_test_split(
                i,
                range(len(i)),
                train_size=train_test_ratio[0],
                random_state=self.seed,
            )
            self.traindata1.append(d1)
            self.valdata1.append(v1)
            self.train_indices1.append(i1)
            self.val_indices1.append(i2)

        for i in data2:
            d2, v2, i1, i2 = train_test_split(
                i,
                range(len(i)),
                train_size=train_test_ratio[0],
                random_state=self.seed,
            )
            self.traindata2.append(d2)
            self.valdata2.append(v2)
            self.train_indices2.append(i1)
            self.val_indices2.append(i2)

        torch.manual_seed(self.seed)
        if train_lists is None:
            self.train_set = ESM3MultiTrackBalancedDataset(
                self.traindata1,
                self.traindata2,
                self.data3,
                augment=aug,
                pos_neg_sample=pos_neg_train,
                tracks=tracks,
                required_labels=required_labels,
            )
        else:
            self.train_set = ListESM3MultiTrackBalancedDataset(
                train_lists,
                list_size,
                train_list_require_mle,
                self.traindata1,
                self.traindata2,
                self.data3,
                augment=aug,
                pos_neg_sample=pos_neg_train,
                tracks=tracks,
                required_labels=required_labels,
            )
        if val_lists is None:
            self.val_set = ESM3MultiTrackBalancedDataset(
                self.valdata1,
                self.valdata2,
                self.data3,
                augment=aug,
                pos_neg_sample=pos_neg_val,
                tracks=tracks,
                required_labels=required_labels,
            )
        else:
            self.val_set = ListESM3MultiTrackBalancedDataset(
                val_lists,
                list_size,
                val_list_require_mle,
                self.valdata1,
                self.valdata2,
                self.data3,
                augment=aug,
                pos_neg_sample=pos_neg_val,
                tracks=tracks,
                required_labels=required_labels,
            )

        self.test_set = ESM3MultiTrackDatasetTEST(self.data3, tracks=tracks)

    def train_dataloader(self):
        self.value += 1
        print("get train loader")
        return MyDataLoader(
            self.train_set,
            True,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        self.value += 1
        print("get val loader")
        return MyDataLoader(
            self.val_set,
            False,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=4,
        )

    def predict_dataloader(self):
        self.value += 1
        print("get predict loader")
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
        return MyDataLoader(
            self.test_set,
            False,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )


class ESM3BalancedDataModuleActiveLearning(ESM3BalancedDataModule):
    def __init__(self, active_learning_datasets, list_size=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.list_size = list_size

        self.active_learning_datasets = []
        cnt = 0
        for i in active_learning_datasets:
            activate_learning_dataset = []
            for j in i:
                t = {}
                for k in ["seq_t", "structure_t", "sasa_t", "second_t"]:
                    if k in j:
                        t[k] = j[k]
                t["id"] = cnt
                cnt += 1
                activate_learning_dataset.append(t)
            self.active_learning_datasets.append(activate_learning_dataset)

        self.train_set = ListActiveLearningDataset(
            self.active_learning_datasets, self.train_set, self.list_size
        )
        self.val_set = ListActiveLearningDataset(
            self.active_learning_datasets, self.val_set, self.list_size
        )


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

        train_set, val_set = torch.utils.data.random_split(ds1, train_test_split)
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
