import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def cales(pos, total):
    t = -100000
    es = 0
    for i in range(total):
        if i in pos:
            es += 1
        else:
            es -= len(pos) / (total - len(pos))
        if es > t:
            t = es
    return t


def calres(pos, total):
    t = -100000
    es = 0
    res = []
    for i in range(total):
        if i in pos:
            es += 1
        else:
            es -= len(pos) / (total - len(pos))
        if es > t:
            t = es
        res.append(es)
    return res


def calpermutation(pos, total, permutation=100000):
    res = []
    for i in tqdm(range(permutation)):
        q = random.sample(range(total), pos)
        t = cales(q, total)
        res.append(t)
    return res


def gsea(pos, total, permutation=100000, t=None):
    es = cales(pos, total)
    cnt = 0
    if t is None:
        for i in tqdm(range(permutation)):
            q = random.sample(range(total), len(pos))
            t = cales(q, total)
            if t >= es:
                cnt += 1
        return cnt / permutation
    else:
        for i in t:
            if i >= es:
                cnt += 1
        return cnt / len(t)


def yeast(path, ingore_structure=False):
    a = np.loadtxt(path)
    df = pd.read_csv("temp/res.csv", index_col=0)
    df["new"] = a
    df = df[["已合成的序列", "预测的序列", "new"]]
    df = df.sort_values("new", ascending=False)
    df["rank"] = range(len(df))
    dfyeast = pd.read_excel("temp/筛选结果.xlsx", sheet_name="Yeast screening_res")
    if ingore_structure:
        dfyeast = dfyeast[dfyeast["structure_not"] == 0]
    df["is_yeast"] = df["已合成的序列"].apply(
        lambda x: x in dfyeast["已合成的序列"].values
    )
    return df


def getEmbeddings(
    model, dl, device, trunc=30000, model_type="esm3", take_embed="first"
):
    res = []
    labels = []
    model = model.to(device)
    model.eval()
    pbar = tqdm(dl)
    cnt = 0
    with torch.no_grad():
        for i, j in enumerate(pbar):
            if isinstance(j, list):

                if j[1].dim() == 2:
                    labels.append(j[1][0][0].item())
                else:
                    labels.append(j[1][0].item())
                j = j[0]

            if cnt == trunc:
                break
            cnt += 1

            for track in ["seq_t", "structure_t", "ss8_t", "sasa_t"]:
                if track not in j:
                    j[track] = None
                else:
                    j[track] = j[track].to(device)
                    if len(j[track].size()) == 1:
                        j[track] = j[track].unsqueeze(0)

            if model_type == "esm3":
                representations = model(
                    sequence_tokens=j["seq_t"],
                    structure_tokens=j["structure_t"],
                    ss8_tokens=j["ss8_t"],
                    sasa_tokens=j["sasa_t"],
                )
            elif model_type == "esmc":
                representations = model(
                    sequence_tokens=j["seq_t"],
                )

            if take_embed == "first":
                representations = representations.embeddings[:, 0]
            else:
                representations = torch.mean(representations.embeddings, dim=1)
            res.append(representations.cpu().numpy())
    return res, labels


def tsnedf(embed, labels):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    q = np.array(embed)
    q = q.squeeze()

    pca = PCA(n_components=30)
    pca_seq = pca.fit_transform(q)
    pca_seq.shape

    X_embedded = TSNE(n_components=2).fit_transform(pca_seq)
    df = pd.DataFrame({"x": X_embedded[:, 0], "y": X_embedded[:, 1], "label": labels})
    return df


def umapdf(embed, labels, min_dist=0.3, n_neighbors=15):
    import umap

    q = np.array(embed)
    q = q.squeeze()

    reducer = umap.UMAP(min_dist=min_dist, n_neighbors=n_neighbors)
    embedding = reducer.fit_transform(q)
    df = pd.DataFrame({"x": embedding[:, 0], "y": embedding[:, 1], "label": labels})
    return df


def ERcorrelation(path):
    from scipy import stats

    a = np.loadtxt(path)
    df = pd.read_csv("temp/res.csv", index_col=0)
    df["new"] = a
    df = df[["已合成的序列", "预测的序列", "new"]]
    df = df.set_index("已合成的序列")
    # df = df.sort_values("new", ascending=False)
    # df["rank"] = range(len(df))
    dfexp = pd.read_excel("./temp/1210_对应信息.xlsx", sheet_name="Sheet1")
    dfexp["new"] = dfexp["蛋白质序列.1"].map(df["new"])
    dfexp2 = pd.read_excel("./temp/1210_对应信息.xlsx", sheet_name="Sheet2")
    dfexp2["rank"] = range(len(dfexp2))
    dfexp2["rank"] = -dfexp2["rank"]
    dfexp["rank"] = dfexp["质粒的名称"].map(dfexp2.set_index("Name")["rank"])

    print(stats.spearmanr(dfexp["new"], dfexp["rank"]))

    return dfexp