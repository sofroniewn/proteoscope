import os
from glob import glob
from os.path import join
from typing import List

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm


def _determine_load_paths(datapath):
    labels = sorted(glob(join(datapath, "Label_*.csv")))
    images = sorted(glob(join(datapath, "Image_*.npy")))
    return labels, images


def split_data(num_samples: int, splits: List[int]) -> List[int]:
    split_edges = list(np.round(num_samples * np.cumsum(splits)).astype(int))
    split_edges[-1] = num_samples
    split_edges.insert(0, 0)
    indices = np.random.permutation(num_samples)

    split_indices = []
    for i in range(len(split_edges) - 1):
        start = split_edges[i]
        stop = split_edges[i + 1]
        split_indices.append(indices[start:stop])
    return split_indices


# Get paths for labels and images
datapath = "/home/ec2-user/cytoself-data"
labels, images = _determine_load_paths(datapath)

# Concatenate and split labels
df = []
for file in labels:
    df.append(pd.read_csv(file))
df = pd.concat(df)
df = df.reset_index()

# Create protein splits
num_proteins = len(df["ensg"].unique())
splits = split_data(num_proteins, [0.8, 0.1, 0.1])
print(num_proteins, np.sum([len(s) for s in splits]))

df["split_protein"] = ""
for split_indices, split_name in zip(splits, ["train", "val", "test"]):
    full_indices = df["ensg"].isin(df["ensg"].unique()[split_indices])
    df.loc[full_indices, "split_protein"] = split_name
print((df["split_protein"] == "").sum())

# Create image splits within train proteins
num_images = (df["split_protein"] == "train").sum()
splits = split_data(num_images, [0.8, 0.1, 0.1])
print(num_images, np.sum([len(s) for s in splits]))

df["split_images"] = ""
for split_indices, split_name in zip(splits, ["train", "val", "test"]):
    full_indices = df[df["split_protein"] == "train"].index.values
    df.loc[full_indices[split_indices], "split_images"] = split_name
print((df["split_images"] != "").sum(), (df["split_protein"] == "train").sum())

df.to_csv(join(datapath, "labels.csv"))

# Convert images to zarr
shape = (len(df), 100, 100, 4)
chunks = (1, None, None, 2)
zarr_path = join(datapath, "images.zarr")

z = zarr.open(zarr_path, mode="a", shape=shape, chunks=chunks)

start = 0
for file in tqdm(images):
    data = np.load(file)
    stop = start + len(data)
    z[start:stop] = data
    os.remove(file)  # remove npy files as very large
    start = stop
