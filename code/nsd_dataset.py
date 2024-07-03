### Dataloader for NSD Dataset ###
# Assumes that data is structred exactly as in https://huggingface.co/datasets/pscotti/mindeyev2
# Code is modified from: https://github.com/MedARC-AI/MindEyeV2/blob/main/src/Train.ipynb
#
# Notes:
# - Because the subject metadata is structured as a webdataset, it is not as ergonomic to use as a traditional pytorch dataset
# - This assumes that the desired sequence length is 1, see this file for how to get past and future entries (https://github.com/MedARC-AI/MindEyeV2/blob/main/src/Train.ipynb)
#
# See the example at the bottom for how to use this dataset

# %% Imports
import webdataset as wds
from itertools import islice
import h5py
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# %% Utility Functions
def identity(x):
    return x


def my_split_by_node(urls):
    return urls


def torch_to_matplotlib(x):
    if torch.mean(x) > 10:
        x = (x.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    else:
        x = (x.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    return x.cpu().numpy()[0]


# %%
def generate_dataset(subject_no, num_sessions, base_path, voxel_dtype=torch.float16):
    base_path = Path(base_path)
    subj_id = f"subj0{subject_no}"
    sub_path = base_path / "wds" / subj_id
    h5path = base_path / ("betas_all_" + subj_id + "_fp32_renorm.hdf5")

    metadata = (
        wds.WebDataset(
            str(sub_path / "train" / f"{{0..{num_sessions}}}.tar"),
            resampled=True,
            nodesplitter=my_split_by_node,
        )
        .shuffle(750, initial=1500, rng=random.Random(42))
        .decode("torch")
        .rename(
            behav="behav.npy",
            past_behav="past_behav.npy",
            future_behav="future_behav.npy",
            olds_behav="olds_behav.npy",
        )
        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    )

    f = h5py.File(h5path, "r")
    betas = f["betas"][:]
    voxels = torch.Tensor(betas).to("cpu").to(voxel_dtype)
    return metadata, voxels


def get_batch(metadata, images, voxels):
    behav, _, _, _ = metadata

    image = images[behav[:, 0, 0].cpu().long()].float()

    voxel = voxels[behav[:, 0, 5].cpu().long()]
    voxel = torch.Tensor(voxel)

    return image, voxel


# %%
def create_NSD_dataset(
    base_path,
    fmri_transform=identity,
    image_transform=identity,
    subjects=[1],
    patch_size=16,
    include_non_avg_test=False,
    voxel_dtype=torch.float16,
    new_test=False,
    batch_size=4,
):
    assert len(subjects) == 1, "More than one subject not supported"
    subj = subjects[0]

    if (
        not new_test
    ):  # using old test set from before full dataset released (used in original MindEye paper)
        if subj == 3:
            num_test = 2113
        elif subj == 4:
            num_test = 1985
        elif subj == 6:
            num_test = 2113
        elif subj == 8:
            num_test = 1985
        else:
            num_test = 2770
    elif new_test:  # using larger test set from after full dataset released
        if subj == 3:
            num_test = 2371
        elif subj == 4:
            num_test = 2188
        elif subj == 6:
            num_test = 2371
        elif subj == 8:
            num_test = 2188
        else:
            num_test = 3000

    base_path = Path(base_path)
    f = h5py.File(base_path / "coco_images_224_float16.hdf5", "r")
    images = f["images"][:]
    images = torch.Tensor(images).to("cpu").to(torch.float16)

    num_voxels = torch.inf

    fmri_train = []
    meta_train = []
    fmri_test = []
    meta_test = []

    subj_id = f"subj0{subj}"
    sub_path = base_path / "wds" / subj_id
    h5path = base_path / ("betas_all_" + subj_id + "_fp32_renorm.hdf5")

    num_train_sessions = len(list((sub_path / "train").iterdir()))
    meta_train = (
        wds.WebDataset(
            str(sub_path / "train" / f"{{0..{num_train_sessions-1}}}.tar"),
            resampled=True,
            nodesplitter=my_split_by_node,
        )
        .shuffle(750, initial=1500, rng=random.Random(42))
        .decode("torch")
        .rename(
            behav="behav.npy",
            past_behav="past_behav.npy",
            future_behav="future_behav.npy",
            olds_behav="olds_behav.npy",
        )
        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    )

    f = h5py.File(h5path, "r")
    betas = f["betas"][:]
    fmri_train = torch.Tensor(betas).to("cpu").to(voxel_dtype)

    test_pth = sub_path / ("test_new" if new_test else "test")
    num_test_sessions = len(list(test_pth.iterdir()))
    meta_test = (
        wds.WebDataset(
            str(test_pth / f"{{0..{num_test_sessions-1}}}.tar"),
            resampled=True,
            nodesplitter=my_split_by_node,
        )
        .decode("torch")
        .rename(
            behav="behav.npy",
            past_behav="past_behav.npy",
            future_behav="future_behav.npy",
            olds_behav="olds_behav.npy",
        )
        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    )

    f = h5py.File(h5path, "r")
    betas = f["betas"][:]
    fmri_test = torch.Tensor(betas).to("cpu").to(voxel_dtype)

    num_voxels = fmri_train.shape[1]

    return (
        NSD_Dataset(meta_train, images, fmri_train, batch_size, num_voxels, (750*num_train_sessions) // batch_size),
        NSD_Dataset(meta_test, images, fmri_test, batch_size, num_voxels, num_test // batch_size),
    )


class NSD_Dataset:
    def __init__(self, meta, images, fmri, batch_size, num_voxels, num_items):
        self.meta = meta
        self.meta = self.meta.with_length(num_items)
        self.images = images
        self.fmri = fmri
        self.batch_size = batch_size
        self.num_voxels = num_voxels
        self.num_items = num_items

        self.iter = 0
    
    def __len__(self):
        return self.num_items * self.batch_size

    def get_batch(self, metadata):
        behav, _, _, _ = metadata

        image = self.images[behav[:, 0, 0].cpu().long()].float()
        image = F.resize(image, 256)

        voxel = self.fmri[behav[:, 0, 5].cpu().long()].float()
        voxel = voxel.unsqueeze(1)

        return image, voxel

# %% Examples
if __name__ == "__main__":
    # data_path = Path("/home/internkavi/kavi_tmp/datasets/nsd-mind-eye")
    # f = h5py.File(data_path / "coco_images_224_float16.hdf5", "r")
    # # if you go OOM you can remove the [:] so it isnt preloaded to cpu (will require a few edits elsewhere tho)
    # images = f["images"][:]
    # images = torch.Tensor(images).to("cpu").to(torch.float16)

    # subject_metadata, subject_voxels = generate_dataset(1, 3, data_path)

    # batch_size = 1
    # dataloader = torch.utils.data.DataLoader(
    #     subject_metadata.batched(batch_size), num_workers=4, batch_size=None
    # )
    # # %%
    # for metadata in dataloader:
    #     image, voxel = get_batch(metadata, images, subject_voxels)
    #     plt.subplot(2, 1, 1)
    #     plt.imshow(torch_to_matplotlib(image))
    #     plt.subplot(2, 1, 2)
    #     plt.plot(voxel.T)
    #     break

    train_set, test_set = create_NSD_dataset(
        "/home/internkavi/kavi_tmp/datasets/nsd-mind-eye", subjects=[1], batch_size=4
    )
    # %%
    train_loader = torch.utils.data.DataLoader(
        train_set.meta.batched(4), num_workers=4, batch_size=None
    )

    train_set.iter = 0
    for meta in train_loader:
        image, fmri = train_set.get_batch(meta)
        print(train_set.iter, fmri.shape)
        train_set.iter += 1
        if train_set.iter >= train_set.num_items:
            break
# %%
