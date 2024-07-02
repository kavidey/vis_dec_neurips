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
import h5py
from pathlib import Path
import random
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# %% Utility Functions
def identity(x):
    return x

def my_split_by_node(urls):
    return urls

def torch_to_matplotlib(x):
    if torch.mean(x)>10:
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

    f = h5py.File(h5path, 'r')
    betas = f['betas'][:]
    voxels = torch.Tensor(betas).to("cpu").to(voxel_dtype)
    return metadata, voxels

def get_batch(metadata, images, voxels):
    behav, _, _, _ = metadata
    
    image = images[behav[:,0,0].cpu().long()].float()

    voxel = voxels[behav[:,0,5].cpu().long()]
    voxel = torch.Tensor(voxel)

    return image, voxel
# %% Examples
if __name__ == "__main__":
    data_path = Path("/home/internkavi/kavi_tmp/datasets/nsd-mind-eye")
    f = h5py.File(data_path/'coco_images_224_float16.hdf5', 'r')
    images = f['images'][:]  # if you go OOM you can remove the [:] so it isnt preloaded to cpu (will require a few edits elsewhere tho)
    images = torch.Tensor(images).to("cpu").to(torch.float16)

    subject_metadata, subject_voxels = generate_dataset(1, 3, data_path)


    batch_size = 1
    dataloader = torch.utils.data.DataLoader(subject_metadata.batched(batch_size), num_workers=4, batch_size=None)
    # %%
    for metadata in dataloader:
        image, voxel = get_batch(metadata, images, subject_voxels)
        plt.subplot(2,1,1)
        plt.imshow(torch_to_matplotlib(image))
        plt.subplot(2,1,2)
        plt.plot(voxel.T)
        break
# %%
