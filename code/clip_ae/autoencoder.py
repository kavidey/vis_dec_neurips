import os
from omegaconf import OmegaConf
import numpy as np
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPVisionModel, CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import (
    UNet2DConditionModel,
    PNDMScheduler,
    AutoencoderKL,
    DDIMScheduler,
)

from sc_mbm.mae_for_fmri import MAEforFMRICross, PatchEmbed1D
from sc_mbm.mae_for_image import ViTMAEConfig, ViTMAELayer


def create_fmri_mae(
    config, sd, config_pretrain, model_image_config, clip_dim, device=None, num_voxels=None
):
    if not num_voxels:
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
        do_cross_attention=config.do_cross_attention,
        # do_cross_attention=False,
        cross_encoder_config=model_image_config,
        decoder_depth=config.fmri_decoder_layers,
        cross_map_dim=clip_dim,
    )
    model.load_checkpoint(sd["model"])
    if device:
        model.to(device)

    patch_embed = PatchEmbed1D(
        num_voxels,
        config_pretrain.patch_size,
        in_chans=1,
        embed_dim=config_pretrain.embed_dim,
    )

    return model, num_voxels, patch_embed


def normalize_embeddings(encoder_output, image_encoder):
    embeds = image_encoder.vision_model.post_layernorm(encoder_output.last_hidden_state)
    embeds = image_encoder.visual_projection(embeds)
    embeds_pooled = embeds[:, 0:1]
    embeds = embeds / torch.norm(embeds_pooled, dim=-1, keepdim=True)
    return embeds

class ConditionLDM(nn.Module):
    def __init__(self, model_image_config, clip_dim=1024, ca_weight=1, guidance_scale=0, skip_weight=1, device=torch.device("cpu")):
        super().__init__()

        self.device = device
        self.ca_weight = ca_weight
        self.skip_weight = skip_weight

        self.guidance_scale = 0
        self.do_unconditional_guidance = guidance_scale != 1

        self.diffusion_model_id = "shi-labs/versatile-diffusion"
        self.unet = UNet2DConditionModel.from_pretrained(
            self.diffusion_model_id, subfolder="image_unet"
        )
        # for param in [p for n, p in self.unet.named_parameters() if "attn2" not in n]:
        #     param.requires_grad = False
        # self.unet.train()
        self.unet.requires_grad_(False)
        self.unet.to(device)

        self.image_feature_extractor = CLIPImageProcessor.from_pretrained(
            self.diffusion_model_id, subfolder="image_feature_extractor"
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.diffusion_model_id, subfolder="image_encoder"
        )
        self.image_encoder.requires_grad_(False)
        self.image_encoder.to(device)

        self.vae = AutoencoderKL.from_pretrained(
            self.diffusion_model_id, subfolder="vae"
        )
        self.vae.requires_grad_(False)
        self.vae.to(device)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.diffusion_model_id, subfolder="scheduler"
        )

        self.cross_blocks = nn.ModuleList(
            [
                ViTMAELayer(model_image_config, True)
                for _ in range(model_image_config.num_cross_encoder_layers)
            ]
        )
        for block in self.cross_blocks:
            block.to(device)

        self.norm = nn.Softmax(dim=2)
        self.norm.to(device)

    def encode_clip(self, img):
        # inputs = self.clip_processor(images=img, return_tensors="pt", do_rescale=False)
        # pixel_values = inputs["pixel_values"].to(self.device)
        # clip_embeddings = self.clip_model(pixel_values).last_hidden_state

        image_input = self.image_feature_extractor(img, return_tensors="pt", do_rescale=False).pixel_values
        image_input = image_input.to(self.device)
        image_embeddings = self.image_encoder(image_input)
        image_embeddings = normalize_embeddings(image_embeddings, self.image_encoder)

        return image_embeddings

    def diffusion_step(self, image, condition):
        """
        condition needs to have shape (batch, X, 768)
        """
        img = self.image_feature_extractor(image, do_rescale=False)
        img = torch.vstack([torch.from_numpy(t)[None] for t in img["pixel_values"]])
        img = img.to(image.device)

        latents = (
            self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor
        )
        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.size(0),),
            device=latents.device,
        ).long()

        if self.do_unconditional_guidance:
            uncond_img = [np.zeros((512, 512, 3)) + 0.5] * image.shape[0]
            uncond_img = self.image_feature_extractor(uncond_img, do_rescale=False)
            uncond_img = torch.vstack([torch.from_numpy(t)[None] for t in uncond_img["pixel_values"]])
            uncond_embeddings = self.encode_clip(uncond_img)

            condition = torch.cat([uncond_embeddings, condition])
            latent_model_inputs = torch.cat([latents]*2)
            noise_inputs = torch.cat([noise]*2)
            timesteps = torch.cat([timesteps]*2)
        else:
            latent_model_inputs = latents
            noise_inputs = noise

        noisy_latents = self.scheduler.add_noise(latent_model_inputs, noise_inputs, timesteps)

        noise_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states=condition
        ).sample

        if self.do_unconditional_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        loss = F.mse_loss(noise_pred, noise)

        return loss

    def forward(self, img, encoder_only=False, fmri_support=None):
        embeddings = self.encode_clip(img)
        if encoder_only:
            return embeddings
        else:
            x = embeddings
            cross_x = self.cross_attention(x, fmri_support)
            x = x * self.skip_weight + cross_x * self.ca_weight

            return self.diffusion_step(img, x)
    
    def cross_attention(self, embeddings, fmri_support):
        cross_x = embeddings.clone()

        for blk in self.cross_blocks:
            cross_x_full = blk(cross_x, hidden_states_mod2=fmri_support)
            cross_x = cross_x_full[0]
        
        cross_x = self.norm(cross_x)

        return cross_x
        

    def recon_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W] pred: [N, L, p*p*3] mask: [N, L], 0 is keep, 1 is remove,
        """
        loss = ((imgs - pred) ** 2).mean()
        return loss

    def calc_corr(self, imgs, pred):
        f_imgs = imgs.reshape(len(imgs), -1).cpu()
        f_pred = pred.reshape(len(pred), -1).cpu()

        corrs = [np.corrcoef(f_imgs[i], f_pred[i])[0][1] for i in range(len(imgs))]

        return np.mean(corrs)

    @torch.no_grad()
    def generate_image(self, image, fmri_support, steps):
        embeddings = self.encode_clip(image)
        x = embeddings
        cross_x = self.cross_attention(x, fmri_support)
        x = x * self.skip_weight + cross_x * self.ca_weight

        latents = (
            torch.randn((x.shape[0], 4, 64, 64), device=self.device)
            * self.scheduler.init_noise_sigma
        )

        if self.do_unconditional_guidance:
            uncond_img = [np.zeros((512, 512, 3)) + 0.5] * x.shape[0]
            uncond_img = self.image_feature_extractor(uncond_img, do_rescale=False)
            uncond_img = torch.vstack([torch.from_numpy(t)[None] for t in uncond_img["pixel_values"]])
            uncond_embeddings = self.encode_clip(uncond_img)

            x = torch.cat([uncond_embeddings, x])

        self.scheduler.set_timesteps(steps)
        scheduler_steps = self.scheduler.timesteps
        
        for t in scheduler_steps:
            latent_model_input = torch.cat([latents] * 2) if self.do_unconditional_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=x,
                return_dict=False,
            )[0]

            if self.do_unconditional_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents_scaled = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents_scaled).sample.detach()

        image = (image / 2 + 0.5).clamp(0, 1)

        return transforms.Resize(256)(image)


class fMRICLIPAutoEncoder(nn.Module):
    def __init__(
        self, config, model_image_config, clip_dim=1024, num_voxels=None, ca_weight=1, skip_weight=1, device=torch.device("cpu")
    ):
        super().__init__()

        self.config = config
        sd = torch.load(config.pretrain_mbm_path, map_location="cpu")
        config_pretrain = sd["config"]

        self.mae, self.num_voxels, self.patch_embed = create_fmri_mae(
            config, sd, config_pretrain, model_image_config, clip_dim, device, num_voxels=num_voxels
        )

        self.ca_weight = ca_weight
        self.skip_weight = skip_weight

        # Freeze model so we're only learning linear layers
        # for param in self.mae.parameters():
        #     param.requires_grad = False

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

        x = self.encoder(latent)
        # print(f"{x.shape=}")
        if encoder_only:
            x = self.norm(x)
            return x
        else:
            # print(f"{image_support.shape=}, {image_support_2d.shape=}")
            cross_x = x.clone()

            for blk in self.cross_blocks:
                # print(f"{cross_x.shape=}")
                cross_x_full = blk(cross_x, hidden_states_mod2=image_support)
                cross_x = cross_x_full[0]

            x = x*self.skip_weight + cross_x*self.ca_weight

            x = self.norm(x)
            # print(f"{x.shape=}")

            # print(f"{x.shape=}, {self.unmap_dims(x).shape=}")
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

        loss = (
            (loss * mask).sum() / mask.sum() if mask.sum() != 0 else (loss * mask).sum()
        )  # mean loss on removed patches
        return loss
