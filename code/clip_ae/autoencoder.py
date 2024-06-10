import os
from omegaconf import OmegaConf
import numpy as np
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
import sys
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder
from generative_models.sgm.models.diffusion import DiffusionEngine
from omegaconf import OmegaConf

from sc_mbm.mae_for_fmri import MAEforFMRICross, PatchEmbed1D
from sc_mbm.mae_for_image import ViTMAEConfig, ViTMAELayer
from clip_ae.utils import unclip_recon

def create_fmri_mae(config, sd, config_pretrain, model_image_config, device):
    num_voxels = (sd["model"]["pos_embed"].shape[1] - 1) * config_pretrain.patch_size
    model = MAEforFMRICross(
        num_voxels=num_voxels,
        patch_size=config_pretrain.patch_size,
        embed_dim=config_pretrain.embed_dim,
        decoder_embed_dim=config_pretrain.decoder_embed_dim,
        depth=config_pretrain.depth,
        num_heads=config_pretrain.num_heads,
        decoder_num_heads=config_pretrain.decoder_num_heads,
        mlp_ratio=config_pretrain.mlp_ratio,
        focus_range=None,
        use_nature_img_loss=False,
        # do_cross_attention=config.do_cross_attention,
        do_cross_attention=False,
        cross_encoder_config=model_image_config,
        decoder_depth=config.fmri_decoder_layers,
    )
    model.load_state_dict(sd["model"], strict=False)
    model.to(device)

    patch_embed = PatchEmbed1D(num_voxels, config_pretrain.patch_size, in_chans=1, embed_dim=config_pretrain.embed_dim)

    return model, num_voxels, patch_embed

def create_unclip_diffusion_engine():
    # prep unCLIP
    config = OmegaConf.load("generative_models/configs/unclip6.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    network_config = unclip_params["network_config"]
    denoiser_config = unclip_params["denoiser_config"]
    first_stage_config = unclip_params["first_stage_config"]
    conditioner_config = unclip_params["conditioner_config"]
    sampler_config = unclip_params["sampler_config"]
    scale_factor = unclip_params["scale_factor"]
    disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]

    first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
    sampler_config['params']['num_steps'] = 38

    diffusion_engine = DiffusionEngine(network_config=network_config,
                        denoiser_config=denoiser_config,
                        first_stage_config=first_stage_config,
                        conditioner_config=conditioner_config,
                        sampler_config=sampler_config,
                        scale_factor=scale_factor,
                        disable_first_stage_autocast=disable_first_stage_autocast)

    return diffusion_engine

class unCLIP(nn.Module):
    def __init__(self, model_image_config, clip_dim=1024, device=torch.device("cpu")):
        super().__init__()

        self.device = device

        # self.clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        #     arch="ViT-bigG-14",
        #     version="laion2b_s39b_b160k",
        #     output_tokens=True,
        #     only_tokens=True,
        #     device=device
        # )
        # self.clip_img_embedder.to(device)

        self.clip_seq_dim = 256
        self.clip_emb_dim = 1664

        self.diffusion_engine = create_unclip_diffusion_engine()
        self.diffusion_engine.eval()
        self.diffusion_engine.to(device)

        self.vector_suffix = self.generate_vector_suffix()

        self.cross_blocks = nn.ModuleList(
            [
                ViTMAELayer(model_image_config, True)
                for _ in range(model_image_config.num_cross_encoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(clip_dim)

    def generate_vector_suffix(self):
        batch={"jpg": torch.randn(1,3,1,1).to(self.device), # jpg doesnt get used, it's just a placeholder
            "original_size_as_tuple": torch.ones(1, 2).to(self.device) * 768,
            "crop_coords_top_left": torch.zeros(1, 2).to(self.device)}
        out = self.diffusion_engine.conditioner(batch)
        vector_suffix = out["vector"].to(self.device)
        return vector_suffix

    def encode_clip(self, img):
        # return self.clip_img_embedder(img)
        return self.diffusion_engine.conditioner.embedders[0](img)

    def decode_clip(self, embeddings):
        samples = unclip_recon(embeddings, self.diffusion_engine, self.vector_suffix, num_samples=embeddings.shape[0])
        return transforms.Resize(256)(samples)

    def forward(self, img, encoder_only=False, fmri_support=None):
        embeddings = self.encode_clip(img)
        if encoder_only:
            return embeddings
        else:
            x = embeddings
            cross_x = x.clone()

            for blk in self.cross_blocks:
                cross_x_full = blk(cross_x, hidden_states_mod2=fmri_support)
                cross_x = cross_x_full[0]

            x = x + cross_x

            x = self.norm(x)
            # print(f"{x.shape=}")
            pred = self.decode_clip(x)
            return pred
    
    def recon_loss(self, imgs, pred, mask=None):
        """
        imgs: [N, 3, H, W] pred: [N, L, p*p*3] mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = self.patchify(imgs)
        # if self.config.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.0e-6) ** 0.5

        # loss = (pred - target) ** 2
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss = ((imgs - pred) ** 2).mean()
        return loss

    def calc_corr(self, imgs, pred):
        f_imgs = imgs.reshape(len(imgs), -1).cpu()
        f_pred = pred.reshape(len(pred), -1).cpu()

        corrs = [np.corrcoef(f_imgs[i], f_pred[i])[0][1] for i in range(len(imgs))]

        return np.mean(corrs)

class fMRICLIPAutoEncoder(nn.Module):
    def __init__(
        self, config, model_image_config, clip_dim=1024, device=torch.device("cpu")
    ):
        super().__init__()

        self.config = config
        sd = torch.load(config.pretrain_mbm_path, map_location="cpu")
        config_pretrain = sd["config"]

        self.mae, self.num_voxels, self.patch_embed = create_fmri_mae(
            config, sd, config_pretrain, model_image_config, device
        )

        # Freeze Model so we're only learning linear layers
        for param in self.mae.parameters():
            param.requires_grad = False

        # self.map_dims = nn.Conv1d(292, 1, 1)
        # self.unmap_dims = nn.ConvTranspose1d(1, 292, 1)
        self.encoder = nn.Linear(config_pretrain.embed_dim, clip_dim)
        self.decoder = nn.Linear(clip_dim, config_pretrain.embed_dim)
        self.cross_blocks = nn.ModuleList(
            [
                ViTMAELayer(model_image_config, True)
                for _ in range(model_image_config.num_cross_encoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(clip_dim)

    def encode_fmri(self, sample, mask_ratio=0.0):
        x, mask, ids_restore = self.mae.forward_encoder(sample, mask_ratio=mask_ratio)
        return x, (mask, ids_restore)

    def decode_fmri(self, x, metadata):
        _, ids_restore = metadata
        return self.mae.forward_decoder(x, ids_restore)

    def reconstruction_loss(self, target, pred, metadata):
        mask, _ = metadata
        return self.mae.forward_loss(target, pred, mask)

    def forward(self, sample, encoder_only=False, image_support=None):
        latent, metadata = self.encode_fmri(sample, mask_ratio=self.config.mask_ratio)
        # print(latent.shape, self.map_dims(latent).shape)
        # print(f"{latent.shape=}")
        # latent = self.map_dims(latent)
        x = self.encoder(latent)
        # print(f"{x.shape=}")
        if encoder_only:
            x = self.norm(x)
            return x
        else:
            image_support_2d = image_support.unsqueeze(1)
            # print(f"{image_support.shape=}, {image_support_2d.shape=}")
            cross_x = x.clone()

            for blk in self.cross_blocks:
                # print(f"{cross_x.shape=}")
                cross_x_full = blk(cross_x, hidden_states_mod2=image_support)
                cross_x = cross_x_full[0]

            x = x + cross_x

            x = self.norm(x)
            # print(f"{x.shape=}")
            
            # print(f"{x.shape=}, {self.unmap_dims(x).shape=}")
            # x = self.unmap_dims(x)
            latent = self.decoder(x)
            pred = self.decode_fmri(latent, metadata)

            return pred, metadata
    
    def patchify(self, imgs):
        """
        imgs: (N, 1, num_voxels)
        x: (N, L, patch_size)
        """
        p = self.patch_embed.patch_size
        assert imgs.ndim == 3 and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_embed.patch_size
        h = x.shape[1]
        
        imgs = x.reshape(shape=(x.shape[0], 1, h * p))
        return imgs
    
    def recon_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, num_voxels]
        pred: [N, L, p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  if mask.sum() != 0 else (loss * mask).sum() # mean loss on removed patches
        return loss
