# %%
import sys
import math
import time
import datetime
import os
from omegaconf import OmegaConf
from pathlib import Path
import wandb
import numpy as np
import copy
import argparse

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import create_Kamitani_dataset_distill, create_BOLD5000_dataset_classify
from config import Config_MBM_finetune_cross, merge_needed_cross_config
from clip_ae.utils import (
    wandb_logger,
    create_readme,
    fmri_transform,
    normalize,
    random_crop,
    channel_last,
)
from clip_ae.autoencoder import fMRI_CLIP_Cond_LDM
from sc_mbm.mae_for_image import ViTMAEConfig
from dataset import create_Kamitani_dataset, create_BOLD5000_dataset
from config import Config_Generative_Model

os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_DIR"] = "."


# %%
def get_args_parser():
    parser = argparse.ArgumentParser("MAE finetuning on Test fMRI", add_help=False)

    # Training Parameters
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--mask_ratio", type=float)
    parser.add_argument("--img_mask_ratio", type=float)

    # Project setting
    parser.add_argument("--root_path", type=str)
    parser.add_argument("--pretrain_mbm_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--include_nonavg_test", type=bool)

    # distributed training parameters
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--img_decoder_layers", type=int)
    parser.add_argument("--fmri_decoder_layers", type=int)
    parser.add_argument("--fmri_recon_weight", type=float)
    parser.add_argument("--img_recon_weight", type=float)
    parser.add_argument("--kam_subs", type=str)
    parser.add_argument("--bold5000_subs", type=str)
    parser.add_argument("--load_pretrain_state", type=int)

    return parser


# %%
### Setup configs ##
# args = get_args_parser()
# args = args.parse_args()
config = Config_Generative_Model()
# config = update_config(args, config)
config.clip_dim = 1024
config.checkpoint_path = Path("../checkpoints")

# Masked Autoencoder
mae_config = Config_MBM_finetune_cross()
mae_config.pretrain_mbm_path = "/home/internkavi/kavi_tmp/vis_dec_neurips/checkpoints/checkpoints_pre_140_doublecontra.pth"
mae_config.clip_dim = config.clip_dim
mae_config.fmri_decoder_layers = 6
mae_config.img_decoder_layers = 6
mae_config.img_recon_weight = 1.5 * 10
mae_config.img_mask_ratio = 0.5
mae_config.fmri_recon_weight = 0.25
mae_config.mask_ratio = 0.75
mae_config.dataset = "GOD"
mae_config.batch_size = 4
sd = torch.load(mae_config.pretrain_mbm_path, map_location="cpu")
config_pretrain = sd["config"]

# Cross Attention
cross_attention_config = ViTMAEConfig.from_pretrained(mae_config.vit_mae_model)
cross_attention_config.num_cross_encoder_layers = mae_config.num_cross_encoder_layers
# cross_attention_config.do_cross_attention = config.do_cross_attention
cross_attention_config.do_cross_residual = mae_config.do_cross_residual
cross_attention_config.decoder_num_hidden_layers = mae_config.img_decoder_layers
cross_attention_config.hidden_size = config.clip_dim
cross_attention_config.num_attention_heads = 16

# Latent Diffusion Model with Conditioning
ldm_model_path = Path("../models") / "ldm" / "cin256"
ldm_config_path = ldm_model_path / "cin256.yaml"
ldm_config = OmegaConf.load(ldm_config_path)
ldm_ckpt_path = ldm_model_path / "model.ckpt"

torch.manual_seed(config_pretrain.seed)
np.random.seed(config_pretrain.seed)

config.mae_config = mae_config.__dict__
config.ldm_config = ldm_config

wb = False
if wb:
    wandb.init(
        project="vis-dec",
        config=config,
        reinit=True,
        name="ldm_condition",
    )
    logger = pl.loggers.WandbLogger()
else:
    logger = None
config.mae_config = mae_config
config.logger = logger
# %%
output_sub = config.bold5000_subs if config.dataset == "BOLD5000" else config.kam_subs
output_path = os.path.join(
    config.checkpoint_path,
    "results",
    "fmri_finetune_{}_{}".format(config.dataset, ".".join(output_sub)),
    "%s" % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")),
)
config.output_path = output_path
Path(output_path).mkdir(parents=True, exist_ok=True)
# %%
torch.manual_seed(config_pretrain.seed)
np.random.seed(config_pretrain.seed)

crop_ratio = 0.2
img_size = 256
crop_pix = int(crop_ratio * img_size)
img_transform_train = transforms.Compose(
    [
        normalize,
        random_crop(img_size - crop_pix, p=0.5),
        transforms.Resize((256, 256)),
        channel_last,
    ]
)
img_transform_test = transforms.Compose(
    [normalize, transforms.Resize((256, 256)), channel_last]
)
if mae_config.dataset == "GOD":
    train_set, test_set = create_Kamitani_dataset_distill(
        path=mae_config.kam_path,
        patch_size=config_pretrain.patch_size,
        subjects=mae_config.kam_subs,
        fmri_transform=torch.FloatTensor,
        include_nonavg_test=mae_config.include_nonavg_test,
        return_image_name=True,
    )
elif mae_config.dataset == "BOLD5000":
    train_set, test_set = create_BOLD5000_dataset_classify(
        path=mae_config.bold5000_path,
        patch_size=config_pretrain.patch_size,
        fmri_transform=torch.FloatTensor,
        subjects=mae_config.bold5000_subs,
        include_nonavg_test=mae_config.include_nonavg_test,
    )
else:
    raise NotImplementedError

num_voxels = (sd["model"]["pos_embed"].shape[1] - 1) * config_pretrain.patch_size
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
# %%
model = fMRI_CLIP_Cond_LDM(
    mae_config,
    cross_attention_config,
    ldm_config,
    ldm_ckpt_path,
    # mae_tokens=292,
    mae_tokens=73,
    clip_dim=config.clip_dim,
    ddim_steps=250,
)
# create_readme(config, output_path)
# %%
# model.load_state_dict(
#     torch.load(
#         "/home/internkavi/kavi_tmp/vis_dec_neurips/checkpoints/results/fmri_finetune_GOD_sbj_3/12-06-2024-09-49-15/checkpoint.pth"
#     )["model_state_dict"]
# )
# model.to(torch.device("cuda"))
# # %%
# img_test = test_set[20]["image"]
# fmri_test = test_set[20]["fmri"].to(torch.device("cuda"))
# fmri_latents = model.mae(fmri_test[None], encoder_only=True)
# recon = model.generate_conditioned_image(fmri_latents)
# F.to_pil_image(recon[0].cpu())
# F.to_pil_image(torch.from_numpy(img_test).permute((2,0,1)))
# %%
# resume training if applicable
# if config.checkpoint_path is not None:
#     model_meta = torch.load(config.checkpoint_path, map_location='cpu')
#     generative_model.model.load_state_dict(model_meta['model_state_dict'])
#     print('model resumed')
# %%
# finetune the model
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=output_path,
    monitor=None,
    save_top_k=-1,
    save_last=False,
    filename="ldm_condition.e{epoch:02d}",
)
trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=config.num_epoch,
    logger=logger,
    precision=config.precision,
    accumulate_grad_batches=config.accumulate_grad,
    check_val_every_n_epoch=5,
    # strategy="ddp",
    fast_dev_run=True,
    callbacks=[checkpoint_callback],
    default_root_dir=output_path,
    devices=1,
)

dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

model.learning_rate = config.lr

trainer.fit(model, dataloader, val_dataloaders=test_loader)

model.ldm.unfreeze_whole_model()
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": config,
        "state": torch.random.get_rng_state(),
    },
    os.path.join(output_path, "checkpoint.pth"),
)

# # generate images
# # generate limited train images and generate images for subjects seperately
# generate_images(generative_model, fmri_latents_dataset_train, fmri_latents_dataset_test, config)
# %%
