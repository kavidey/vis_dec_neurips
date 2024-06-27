# %% IMPORTS
import os

# os.environ["HF_HOME"] = "/home/users/nus/li.rl/scratch/intern_kavi/.cache/"
import sys
import math
import time
import datetime
from omegaconf import OmegaConf
from pathlib import Path
import wandb
import numpy as np
import copy
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from einops import rearrange, repeat
from torchvision.utils import make_grid, save_image
import torchinfo

from dataset import create_Kamitani_dataset_distill, create_BOLD5000_dataset_classify
from config import Config_MBM_finetune_cross, merge_needed_cross_config
from clip_ae.utils import (
    update_config,
    wandb_logger,
    create_readme,
    add_weight_decay,
    save_model_merge_conf,
)
from clip_ae.autoencoder import fMRICLIPAutoEncoder, ConditionLDM
from sc_mbm.mae_for_image import ViTMAEConfig, ViTMAELayer

os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_DIR"] = "."
# %%
model_ckpt_path = "/home/internkavi/kavi_tmp/vis_dec_neurips/code/results/fmri_finetune_GOD_sbj_1/26-06-2024-14-27-48/checkpoints_0/checkpoint_singlesub_clip_cross_att_GOD_sbj_1_fmriw0.25_imgw1.5_fmar0.75_imar0.5_fmridl6_imgdl6_pretr1_with_checkpoints_pre_140_doublecontra.pth_epo0_mergconf.pth"
model_image_ckpt_path = "/home/internkavi/kavi_tmp/vis_dec_neurips/code/results/fmri_finetune_GOD_sbj_1/26-06-2024-14-27-48/checkpoints_0/checkpoint_singlesub_clip_cross_att_GOD_sbj_1_fmriw0.25_imgw1.5_fmar0.75_imar0.5_fmridl6_imgdl6_pretr1_with_checkpoints_pre_140_doublecontra.pth_epo0_mergconf_img.pth"
# %%
config = Config_MBM_finetune_cross()
# testing arguments
# config.pretrain_mbm_path = "/home/users/nus/li.rl/scratch/intern_kavi/vis_dec_neurips/checkpoints/checkpoints_pre_140_doublecontra.pth"
config.pretrain_mbm_path = "/home/internkavi/kavi_tmp/vis_dec_neurips/checkpoints/checkpoints_pre_140_doublecontra.pth"
config.clip_dim = 768
config.fmri_decoder_layers = 6
config.img_decoder_layers = 6
config.img_recon_weight = 1.5
config.img_mask_ratio = 0.5
config.fmri_recon_weight = 0.25
config.mask_ratio = 0.75
config.dataset = "GOD"
config.batch_size = 4
config.img_ca_weight = 1
config.guidance_scale = 1

sd = torch.load(config.pretrain_mbm_path, map_location="cpu")
config_pretrain = sd["config"]

device = (
    torch.device(f"cuda:{config.local_rank}")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
# %%
model_image_config = ViTMAEConfig.from_pretrained(config.vit_mae_model)
model_image_config.num_cross_encoder_layers = config.num_cross_encoder_layers
# model_image_config.do_cross_attention = config.do_cross_attention
model_image_config.do_cross_residual = config.do_cross_residual
model_image_config.decoder_num_hidden_layers = config.img_decoder_layers
model_image_config.hidden_size = config.clip_dim
model_image_config.num_attention_heads = 16

model = fMRICLIPAutoEncoder(config, model_image_config, config.clip_dim, device=device)
model.load_state_dict(torch.load(model_ckpt_path)["model"])
model.eval()
model.to(device)
num_voxels = model.num_voxels

model_image = ConditionLDM(
    model_image_config,
    config.clip_dim,
    config.img_ca_weight,
    config.guidance_scale,
    device=device,
)
model_image.load_state_dict(torch.load(model_image_ckpt_path))
model_image.eval()
model_image.to(device)
# %% Setup Datasets
# create dataset and dataloader
if config.dataset == "GOD":
    train_set, test_set = create_Kamitani_dataset_distill(
        path=config.kam_path,
        patch_size=config_pretrain.patch_size,
        subjects=config.kam_subs,
        fmri_transform=torch.FloatTensor,
        include_nonavg_test=config.include_nonavg_test,
        return_image_name=True,
    )
elif config.dataset == "BOLD5000":
    train_set, test_set = create_BOLD5000_dataset_classify(
        path=config.bold5000_path,
        patch_size=config_pretrain.patch_size,
        fmri_transform=torch.FloatTensor,
        subjects=config.bold5000_subs,
        include_nonavg_test=config.include_nonavg_test,
    )
else:
    raise NotImplementedError

if train_set.fmri.shape[-1] < num_voxels:
    train_set.fmri = np.pad(
        train_set.fmri, ((0, 0), (0, num_voxels - train_set.fmri.shape[-1])), "wrap"
    )
else:
    train_set.fmri = train_set.fmri[:, :num_voxels]

# print(test_set.fmri.shape)
if test_set.fmri.shape[-1] < num_voxels:
    test_set.fmri = np.pad(
        test_set.fmri, ((0, 0), (0, num_voxels - test_set.fmri.shape[-1])), "wrap"
    )
else:
    test_set.fmri = test_set.fmri[:, :num_voxels]

dataloader_hcp = DataLoader(train_set, batch_size=config.batch_size)
dataloader_hcp_test = DataLoader(test_set, batch_size=config.batch_size)
# %%
for data_dcit in dataloader_hcp_test:
    samples = data_dcit["fmri"]
    samples = samples.to(device)

    images = data_dcit["image"]
    images = images.permute(0, 3, 1, 2).float()
    images = images.to(device)

    clip_score = []

    with torch.no_grad() and torch.cuda.amp.autocast(enabled=True):
        image_input = model_image.image_feature_extractor(
            images, return_tensors="pt", do_rescale=False
        ).pixel_values
        image_input = image_input.to(device)
        image_embeddings = model_image.image_encoder(image_input)
        clip_cls = model_image.encode_clip(images)[:, 0:1]
        # clip_cls = image_embeddings.image_embeds

        fmri_embeddings = model(samples, encoder_only=True)
        fmri_cls = fmri_embeddings[:, 0:1]

        clip_score.append(
            torch.mean(torch.nn.functional.cosine_similarity(clip_cls, fmri_cls, dim=2))
            .detach()
            .cpu()
            .numpy()
        )
    
    print(f"CLIP Score: {np.mean(clip_score)}")
# %%
