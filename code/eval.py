# %% IMPORTS
import os

# os.environ["HF_HOME"] = ".cache/"
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
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from einops import rearrange, repeat
from torchvision.utils import make_grid, save_image
import torchinfo

from dataset import create_Kamitani_dataset_distill, create_BOLD5000_dataset_classify
from nsd_dataset import create_NSD_dataset
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
import torchvision.transforms.functional as F
from transformers.models.clip.modeling_clip import clip_loss

os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_DIR"] = "."
# %%
model_ckpt_path = "results/fmri_finetune_GOD_sbj_1/27-06-2024-09-50-45/checkpoints_300/checkpoint_singlesub_clip_cross_att_GOD_sbj_1_fmriw0.25_imgw1.5_fmar0.75_imar0.5_fmridl6_imgdl6_pretr1_with_checkpoints_pre_140_doublecontra.pth_epo300_mergconf.pth"
model_image_ckpt_path = "results/fmri_finetune_GOD_sbj_1/27-06-2024-09-50-45/checkpoints_300/checkpoint_singlesub_clip_cross_att_GOD_sbj_1_fmriw0.25_imgw1.5_fmar0.75_imar0.5_fmridl6_imgdl6_pretr1_with_checkpoints_pre_140_doublecontra.pth_epo300_mergconf_img.pth"
# %%
config = Config_MBM_finetune_cross()
# testing arguments
# config.pretrain_mbm_path = "/home/users/nus/li.rl/scratch/intern_kavi/vis_dec_neurips/checkpoints/checkpoints_pre_140_doublecontra.pth"
config.pretrain_mbm_path = "../checkpoints/checkpoints_pre_140_doublecontra.pth"
config.clip_dim = 768
config.fmri_decoder_layers = 6
config.img_decoder_layers = 6
config.img_recon_weight = 1.5
config.img_mask_ratio = 0.5
config.fmri_recon_weight = 0.25
config.mask_ratio = 0.75
config.dataset = "NSD"
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

# round number of voxels in NSD to multiple of 16
if config.dataset == "NSD":
    num_voxels = (15724//16)*16
else:
    num_voxels = None

model = fMRICLIPAutoEncoder(config, model_image_config, config.clip_dim, ca_weight=config.fmri_ca_weight, skip_weight=config.fmri_skip_weight, num_voxels=num_voxels, device=device)
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
elif config.dataset == "NSD":
    train_set, test_set = create_NSD_dataset(
        config.nsd_path,
        patch_size=config_pretrain.patch_size,
        fmri_transform=torch.FloatTensor,
        subjects=config.nsd_subs,
        include_non_avg_test=config.include_nonavg_test
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

if config.dataset == "NSD":
    dataloader_hcp = DataLoader(train_set.meta, batch_size=config.batch_size)
    dataloader_hcp_test = DataLoader(test_set.meta, batch_size=config.batch_size)
else:
    dataloader_hcp = DataLoader(train_set, batch_size=config.batch_size)
    dataloader_hcp_test = DataLoader(test_set, batch_size=config.batch_size)
# %%
all_clip_embeddings = []
all_fmri_embeddings = []

for data_dcit in dataloader_hcp_test:
    if config.dataset != "NSD":
        samples = data_dcit["fmri"]

        images = data_dcit["image"]
        images = images.permute(0, 3, 1, 2).float()

        image_class = data_dcit["image_class"].to(device)
    else:
        images, samples = test_set.get_batch(data_dcit)

    with torch.no_grad() and torch.cuda.amp.autocast(enabled=True):
        image_input = model_image.image_feature_extractor(
            images, return_tensors="pt", do_rescale=False
        ).pixel_values
        image_input = image_input.to(device)
        image_embeddings = model_image.image_encoder(image_input)
        # clip_cls = model_image.encode_clip(images)[:, 0]
        clip_cls = image_embeddings.image_embeds
        # clip_cls = clip_cls.div(torch.norm(clip_cls, p=2, dim=1, keepdim=True))

        fmri_embeddings = model(samples, encoder_only=True)
        fmri_embeddings = model_image.encode_clip(images) + model_image.cross_attention(model_image.encode_clip(images), fmri_embeddings)
        # fmri_embeddings = model_image.cross_attention(model_image.encode_clip(images), fmri_embeddings)
        fmri_cls = fmri_embeddings[:, 0]
        # fmri_cls = fmri_cls.div(torch.norm(fmri_cls, p=2, dim=1, keepdim=True))

        all_clip_embeddings.extend(clip_cls.detach().cpu())
        all_fmri_embeddings.extend(fmri_cls.detach().cpu())

all_clip_embeddings = torch.vstack(all_clip_embeddings)
all_fmri_embeddings = torch.vstack(all_fmri_embeddings)
# %%
# model.ca_weight = 1
# generated = model_image.generate_image(images, fmri_embeddings, 50)
# F.to_pil_image(torch.cat([*generated], dim=2))
# %%
# F.to_pil_image(torch.cat([*images], dim=2))
# %%
clip_scores = torch.nn.functional.cosine_similarity(
    all_clip_embeddings, all_fmri_embeddings, dim=1
)

print(f"CLIP Score: {torch.mean(clip_scores)}")  # %%
# %%
cf_cossim = torch.tensor(cosine_similarity(all_clip_embeddings, all_fmri_embeddings))
fc_cossim = torch.tensor(cosine_similarity(all_fmri_embeddings, all_clip_embeddings))

labels = torch.arange(len(all_clip_embeddings))
# clip retrieval from fmri
cf_retrieval = torch.sum(torch.argsort(cf_cossim, axis=1)[:, -1] == labels) / len(
    labels
)
# fmri retrieval from clip
fc_retrieval = torch.sum(torch.argsort(fc_cossim, axis=1)[:, -1] == labels) / len(
    labels
)

print(f"CLIP retrieval from fMRI: {cf_retrieval:.2%}")
print(f"fMRI retrieval from CLIP: {fc_retrieval:.2%}")
# %%
# logit_scale = model_image.image_encoder.logit_scale.exp()
logit_scale = torch.tensor(2.6592).exp()
logits_per_text = torch.matmul(all_clip_embeddings.float(), all_fmri_embeddings.t()) * logit_scale
logits_per_image = logits_per_text.t()
clip_loss(logits_per_text)
# %%
img_support = model_image(
    images,
    encoder_only=True,
)

latent, metadata = model.encode_fmri(samples, mask_ratio=model.config.mask_ratio)
x = model.encoder(latent)
cross_x = x.clone()
for blk in model.cross_blocks:
    # print(f"{cross_x.shape=}")
    cross_x_full = blk(cross_x, hidden_states_mod2=img_support)
    cross_x = cross_x_full[0]

# x = x + cross_x
# x = cross_x

x = model.norm(x)

latent = model.decoder(x)
pred = model.decode_fmri(latent, metadata)

model.recon_loss(samples, pred, metadata[0])
# %%
