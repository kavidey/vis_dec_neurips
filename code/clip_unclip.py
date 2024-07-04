# %% IMPORTS
import os
os.environ["HF_HOME"] = "/home/users/nus/li.rl/scratch/intern_kavi/.cache/"
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

from nsd_dataset import create_NSD_dataset
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


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %% PARSE ARGUMENTS
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


# %% SETUP CONFIGURATIONS
# args = get_args_parser()
# args, unknown = args.parse_known_args()
config = Config_MBM_finetune_cross()
# config = update_config(args, config)

# Setup multi-gpu processing
multi_gpu = False
# multi_gpu = torch.cuda.device_count() > 1

if multi_gpu:
    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend="nccl")

# testing arguments
config.pretrain_mbm_path = "../checkpoints/checkpoints_pre_140_doublecontra.pth"
config.finetune_path = None
# config.finetune_path = "results/fmri_finetune_GOD_sbj_1/02-07-2024-14-54-02/final/checkpoint_singlesub_clip_cross_att_GOD_sbj_1_fmriw0.25_imgw1.5_fmar0.75_imar0.5_fmridl6_imgdl6_pretr1_with_checkpoints_pre_140_doublecontra.pth_epo999_mergconf.pth"
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
config.img_skip_weight = 1
config.fmri_ca_weight = 1
config.fmri_skip_weight = 1
config.guidance_scale = 1

sd = torch.load(config.pretrain_mbm_path, map_location="cpu")
config_pretrain = sd["config"]

# Setup logging
if config.dataset == "BOLD5000":
    output_sub = config.bold5000_subs
elif config.dataset == "NSD":
    output_sub = config.nsd_subs
else:
    output_sub = config.kam_subs
output_path = os.path.join(
    config.output_path,
    "results",
    "fmri_finetune_{}_{}".format(config.dataset, output_sub),
    "%s" % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")),
)

config.output_path = output_path
if config.dataset == "GOD":
    config.wandb_name = f"clip_cross_att_{config.dataset}_{config.kam_subs}_fmriw{config.fmri_recon_weight}_imgw{config.img_recon_weight}_fmar{config.mask_ratio}_imar{config.img_mask_ratio}_fmridl{config.fmri_decoder_layers}_imgdl{config.img_decoder_layers}_pretr{config.load_pretrain_state}_with_{config.pretrain_mbm_path.split('/')[-1]}"
elif config.dataset == "NSD":
    config.wandb_name = f"clip_cross_att_{config.dataset}_{config.nsd_subs}_fmriw{config.fmri_recon_weight}_imgw{config.img_recon_weight}_fmar{config.mask_ratio}_imar{config.img_mask_ratio}_fmridl{config.fmri_decoder_layers}_imgdl{config.img_decoder_layers}_pretr{config.load_pretrain_state}_with_{config.pretrain_mbm_path.split('/')[-1]}"
else:
    config.wandb_name = f"clip_cross_att_{config.dataset}_{config.bold5000_subs}_fmriw{config.fmri_recon_weight}_imgw{config.img_recon_weight}_fmar{config.mask_ratio}_imar{config.img_mask_ratio}_fmridl{config.fmri_decoder_layers}_imgdl{config.img_decoder_layers}_pretr{config.load_pretrain_state}_with_{config.pretrain_mbm_path.split('/')[-1]}"

wandb.login(key="033a657f5ef5b2c58bc50620eef125d6f7733490")
logger = wandb_logger(config) if config.local_rank == 0 else None
# logger = None

if config.local_rank == 0:
    os.makedirs(output_path, exist_ok=True)
    create_readme(config, output_path)

device = (
    torch.device(f"cuda:{config.local_rank}")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
# device = torch.device("cpu")
torch.manual_seed(config_pretrain.seed)
np.random.seed(config_pretrain.seed)
# %% Setup Model
model_image_config = ViTMAEConfig.from_pretrained(config.vit_mae_model)
model_image_config.num_cross_encoder_layers = config.num_cross_encoder_layers
# model_image_config.do_cross_attention = config.do_cross_attention
model_image_config.do_cross_residual = config.do_cross_residual
model_image_config.decoder_num_hidden_layers = config.img_decoder_layers
model_image_config.hidden_size = config.clip_dim
model_image_config.num_attention_heads = 16
# print(model_image_config)

# round number of voxels in NSD to multiple of 16
if config.dataset == "NSD":
    num_voxels = (15724//16)*16
else:
    num_voxels = None

model = fMRICLIPAutoEncoder(config, model_image_config, config.clip_dim, ca_weight=config.fmri_ca_weight, skip_weight=config.fmri_skip_weight, num_voxels=num_voxels, device=device)
if config.finetune_path:
    model.load_state_dict(torch.load(config.finetune_path)["model"])
model.to(device)
num_voxels = model.num_voxels
model_without_ddp = model

model_image = ConditionLDM(model_image_config, config.clip_dim, ca_weight=config.img_ca_weight, skip_weight=config.img_skip_weight, guidance_scale=config.guidance_scale, device=device)
# model_image.to(device)

if multi_gpu:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(
        model,
        device_ids=[config.local_rank],
        output_device=config.local_rank,
        find_unused_parameters=config.use_nature_img_loss,
    )

    model_image = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_image)
    model_image = DistributedDataParallel(
        model_image,
        device_ids=[config.local_rank],
        output_device=config.local_rank,
        find_unused_parameters=config.use_nature_img_loss,
    )

param_groups = add_weight_decay([model, model_image], config.weight_decay)
optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))

print(optimizer)

if logger is not None:
    logger.watch_model(model, log="all", log_freq=1000)
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

print(f"Dataset size: {len(train_set)}, {len(test_set)}")
if config.dataset != "NSD":
    sampler = (
        torch.utils.data.DistributedSampler(train_set)
        if multi_gpu
        else torch.utils.data.RandomSampler(train_set)
    )
    dataloader_hcp = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler)
    # test_sampler = (
    #     torch.utils.data.DistributedSampler(test_set)
    #     if multi_gpu
    #     else torch.utils.data.RandomSampler(test_set)
    # )
    dataloader_hcp_test = DataLoader(test_set, batch_size=config.batch_size, drop_last=True)
else:
    dataloader_hcp = DataLoader(train_set.meta, batch_size=config.batch_size)
    dataloader_hcp_test = DataLoader(test_set.meta, batch_size=config.batch_size, drop_last=True)
# %%
start_time = time.time()

cor_list = []
eval_cor_list = []
eval_cor_init = 0.5
best_eval_corr_epoch = 0
saved_epoch_list = []

addition_config = {"num_voxels": num_voxels}
merged_config = merge_needed_cross_config(
    config_pretrain, config, model_image_config, addition_config
)

print("Model Summary")
print(torchinfo.summary(model, depth=1), "\n\n")

print("Model Image Summary")
print(torchinfo.summary(model_image, depth=1))

for ep in range(config.num_epoch):
    ckpt_file_name = f"checkpoint_singlesub_{config.wandb_name}_epo{ep}_mergconf.pth"
    ckpt_img_file_name = f"checkpoint_singlesub_{config.wandb_name}_epo{ep}_mergconf_img.pth"
    if multi_gpu and not config.dataset == "NSD":
        sampler.set_epoch(ep)

    # TRAIN
    model.train(True)
    model_image.train(True)
    optimizer.zero_grad()

    total_loss = []
    total_loss_image = []
    total_cor = []
    total_cor_image = []
    # total_loss_fmri = []
    accum_iter = config.accum_iter

    for data_iter_step, (data_dcit) in enumerate(dataloader_hcp):
        if config.dataset != "NSD":
            samples = data_dcit["fmri"]

            images = data_dcit["image"]
            images = images.permute(0, 3, 1, 2).float()

            image_class = data_dcit["image_class"].to(device)
        else:
            images, samples = train_set.get_batch(data_dcit)

        images = images.to(device)
        samples = samples.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            # reconstruct fmri
            img_support = model_image(
                images,
                encoder_only=True,
            )

            pred, metadata = model(
                samples,
                encoder_only=False,
                image_support=img_support,
            )

            # reconstruct image
            fmri_support = model(samples, encoder_only=True)
            # print(f"{img_support.shape=}, {fmri_support.shape=}")
            loss_img_recon = model_image(
                images,
                encoder_only=False,
                fmri_support=fmri_support,
            )

            # print(img_support, img_support.shape)
            # print(fmri_support, fmri_support.shape)

            # True outputs
            # print(img_support.last_hidden_state.shape, fmri_support.shape)
            # torch.Size([4, 197, 768]) torch.Size([4, 292, 1024])

            loss_fmri_recon = model.recon_loss(samples, pred, metadata[0])
            
        loss = (
            config.fmri_recon_weight * loss_fmri_recon
            + config.img_recon_weight * loss_img_recon
        )

        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optimizer.step()

        loss_value = loss.item()
        # if config.local_rank == 0: print(f"loss:{loss.item()}, elapsed: {(time.time()-start_time)/60:.1f} min")

        if not math.isfinite(loss_value):
            print(
                f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {ep}"
            )
            sys.exit(1)

        pred = pred.to("cpu").detach()
        samples = samples.to("cpu").detach()
        pred = model_without_ddp.unpatchify(pred)
        cor = torch.mean(
            torch.tensor(
                [
                    torch.corrcoef(torch.cat([p, s], axis=0))[0, 1]
                    for p, s in zip(pred, samples)
                ]
            )
        ).item()
        # cor_image = model_image.calc_corr(images, img_recons_output)
        optimizer.zero_grad()

        total_loss.append(loss_fmri_recon.item())
        total_loss_image.append(loss_img_recon.item())
        total_cor.append(cor)
        # total_cor_image.append(cor_image)

        if config.dataset == "NSD":
            # print(data_iter_step, train_set.num_items)
            if data_iter_step >= train_set.num_items:
                break

    if logger is not None:
        lr = optimizer.param_groups[0]["lr"]
        logger.log("train_loss_step_fmri", np.mean(total_loss), step=ep)
        logger.log("train_loss_step_image", np.mean(total_loss_image), step=ep)
        logger.log("lr", lr, step=ep)
        logger.log("cor_fmri", np.mean(total_cor), step=ep)
        # logger.log("cor_image", np.mean(cor_image), step=ep)
        if start_time is not None:
            logger.log("time (min)", (time.time() - start_time) / 60.0, step=ep)
    if config.local_rank == 0:
        print(
            f"[Epoch {ep}] train loss fmri: {np.mean(total_loss)} train loss image: {np.mean(total_loss_image)}"
        )
        print(
            f"[Epoch {ep}] train corr fmri: {np.mean(total_cor)} train corr image: {np.mean(total_cor_image)}"
        )

    # EVAL
    # save_ckpt = (ep % 5 == 0 and ep != 0)
    save_ckpt = ep % 2
    model.eval()
    model_image.eval()
    total_loss = []
    total_loss_image = []
    total_cor = []
    total_cor_image = []
    # total_loss_fmri = []
    accum_iter = config.accum_iter

    all_samples = []

    for data_iter_step, (data_dcit) in enumerate(dataloader_hcp_test):
        if config.dataset != "NSD":
            samples = data_dcit["fmri"]

            images = data_dcit["image"]
            images = images.permute(0, 3, 1, 2).float()

            image_class = data_dcit["image_class"].to(device)
        else:
            images, samples = test_set.get_batch(data_dcit)

        images = images.to(device)
        samples = samples.to(device)

        with torch.no_grad() and torch.cuda.amp.autocast(enabled=True):
            # reconstruct fmri
            img_support = model_image(
                images,
                encoder_only=True,
            )

            pred, metadata = model(
                samples,
                encoder_only=False,
                image_support=img_support.float(),
            )

            # reconstruct image
            # add mask_ratio = 0
            fmri_support = model(samples, encoder_only=True)
            # print('fmri_support ', fmri_support.shape)
            loss_img_recon = model_image(
                images,
                encoder_only=False,
                fmri_support=fmri_support,
            )

            loss_fmri_recon = model.recon_loss(samples, pred, metadata[0])

        loss = loss_fmri_recon + loss_img_recon

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(
                f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {ep}"
            )
            sys.exit(1)

        if save_ckpt and len(all_samples) == 0:
            generated_images = model_image.generate_image(images, fmri_support, steps=50)
            combined = torch.cat([torch.stack([a_row,b_row]) for a_row, b_row in zip(images.cpu(), generated_images.cpu())])
            all_samples.append(combined)

        # loss_scaler(img_recons_output.loss, optimizer, parameters=model_image.parameters(), clip_grad=config.clip_grad)

        # if (data_iter_step + 1) % accum_iter == 0:
        # cal the cor
        pred = pred.to("cpu").detach()
        samples = samples.to("cpu").detach()
        pred = model_without_ddp.unpatchify(pred)
        cor = torch.mean(
            torch.tensor(
                [
                    torch.corrcoef(torch.cat([p, s], axis=0))[0, 1]
                    for p, s in zip(pred, samples)
                ]
            )
        ).item()
        # cor_image = model_image.calc_corr(images, img_recons_output)

        total_loss.append(loss_fmri_recon.item())
        total_loss_image.append(loss_img_recon.item())
        total_cor.append(cor)
        # total_cor_image.append(cor_image)


        if config.dataset == "NSD":
            # print(data_iter_step, test_set.num_items)
            if data_iter_step >= test_set.num_items:
                break

    if logger is not None:
        logger.log("test_loss_step_fmri", np.mean(total_loss), step=ep)
        logger.log("test_loss_step_image", np.mean(total_loss_image), step=ep)
        logger.log("test_cor_fmri", np.mean(total_cor), step=ep)
        # logger.log("test_cor_image", np.mean(cor_image), step=ep)
        if start_time is not None:
            logger.log("time (min)", (time.time() - start_time) / 60.0, step=ep)
    if config.local_rank == 0:
        print(
            f"[Epoch {ep}] test loss fmri: {np.mean(total_loss)} test loss image: {np.mean(total_loss_image)}"
        )
        print(
            f"[Epoch {ep}] test corr fmri: {np.mean(total_cor)} test corr image: {np.mean(total_cor_image)}"
        )

    if save_ckpt:
        os.makedirs(os.path.join(output_path, f"checkpoints_{ep}"), exist_ok=True)
        if ep % 10 == 0:
            save_model_merge_conf(
                config_pretrain,
                ep,
                model_without_ddp,
                optimizer,
                os.path.join(output_path, f"checkpoints_{ep}"),
                merged_config,
                ckpt_file_name,
            )
            torch.save(model_image.state_dict(), os.path.join(output_path, f"checkpoints_{ep}", ckpt_img_file_name))
        
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=2)
        save_image(grid, os.path.join(output_path, f"checkpoints_{ep}", "recon_image.jpg"))

save_model_merge_conf(
    config_pretrain,
    config.num_epoch,
    model_without_ddp,
    optimizer,
    os.path.join(output_path, "final"),
    merged_config,
    ckpt_file_name,
)
torch.save(model_image.state_dict(), os.path.join(output_path, "final", ckpt_img_file_name))
# %%
