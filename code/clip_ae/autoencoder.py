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

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
import sys

sys.path.append("generative_models/")
import sgm
from generative_models.sgm.modules.encoders.modules import (
    FrozenOpenCLIPImageEmbedder,
)  # bigG embedder
from generative_models.sgm.models.diffusion import DiffusionEngine
from omegaconf import OmegaConf

from sc_mbm.mae_for_fmri import MAEforFMRICross, PatchEmbed1D
from sc_mbm.mae_for_image import ViTMAEConfig, ViTMAELayer
from clip_ae.utils import unclip_recon, instantiate_from_config
from clip_ae.conditioned_ldm import cond_stage_model

# from dc_ldm.models.diffusion.plms import PLMSSampler


def create_fmri_mae(
    config, sd, config_pretrain, model_image_config, clip_dim, device=None
):
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
    model.load_state_dict(sd["model"], strict=False)
    if device:
        model.to(device)

    patch_embed = PatchEmbed1D(
        num_voxels,
        config_pretrain.patch_size,
        in_chans=1,
        embed_dim=config_pretrain.embed_dim,
    )

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

    first_stage_config["target"] = "sgm.models.autoencoder.AutoencoderKL"
    sampler_config["params"]["num_steps"] = 38

    diffusion_engine = DiffusionEngine(
        network_config=network_config,
        denoiser_config=denoiser_config,
        first_stage_config=first_stage_config,
        conditioner_config=conditioner_config,
        sampler_config=sampler_config,
        scale_factor=scale_factor,
        disable_first_stage_autocast=disable_first_stage_autocast,
    )

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
        batch = {
            "jpg": torch.randn(1, 3, 1, 1).to(
                self.device
            ),  # jpg doesnt get used, it's just a placeholder
            "original_size_as_tuple": torch.ones(1, 2).to(self.device) * 768,
            "crop_coords_top_left": torch.zeros(1, 2).to(self.device),
        }
        out = self.diffusion_engine.conditioner(batch)
        vector_suffix = out["vector"].to(self.device)
        return vector_suffix

    def encode_clip(self, img):
        # return self.clip_img_embedder(img)
        return self.diffusion_engine.conditioner.embedders[0](img)

    def decode_clip(self, embeddings):
        samples = unclip_recon(
            embeddings,
            self.diffusion_engine,
            self.vector_suffix,
            num_samples=embeddings.shape[0],
        )
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


def normalize_embeddings(encoder_output, image_encoder):
    embeds = image_encoder.vision_model.post_layernorm(encoder_output.last_hidden_state)
    embeds = image_encoder.visual_projection(embeds)
    embeds_pooled = embeds[:, 0:1]
    embeds = embeds / torch.norm(embeds_pooled, dim=-1, keepdim=True)
    return embeds


class ConditionLDM(nn.Module):
    def __init__(self, model_image_config, clip_dim=1024, ca_weight=1, guidance_scale=0, device=torch.device("cpu")):
        super().__init__()

        self.device = device
        self.ca_weight = ca_weight

        self.guidance_scale = 0
        self.do_unconditional_guidance = guidance_scale > 0

        self.diffusion_model_id = "shi-labs/versatile-diffusion"
        self.unet = UNet2DConditionModel.from_pretrained(
            self.diffusion_model_id, subfolder="image_unet"
        )
        for param in [p for n, p in self.unet.named_parameters() if "attn2" not in n]:
            param.requires_grad = False
        self.unet.train()
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
            block.cuda()

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

        if self.do_unconditional_guidance:
            uncond_img = [np.zeros((512, 512, 3)) + 0.5] * image.shape[0]
            uncond_img = self.image_feature_extractor(uncond_img, do_rescale=False)
            uncond_img = torch.vstack([torch.from_numpy(t)[None] for t in uncond_img["pixel_values"]])
            uncond_embeddings = self.encode_clip(uncond_img)

            condition = torch.cat([uncond_embeddings, condition])

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
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

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
            x = x + cross_x

            return self.diffusion_step(img, x)
    
    def cross_attention(self, embeddings, fmri_support):
        cross_x = embeddings.clone()

        for blk in self.cross_blocks:
            cross_x_full = blk(cross_x, hidden_states_mod2=fmri_support)
            cross_x = cross_x_full[0]

        return cross_x * self.ca_weight
        

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
        x = x + cross_x

        if self.do_unconditional_guidance:
            uncond_img = [np.zeros((512, 512, 3)) + 0.5] * x.shape[0]
            uncond_img = self.image_feature_extractor(uncond_img, do_rescale=False)
            uncond_img = torch.vstack([torch.from_numpy(t)[None] for t in uncond_img["pixel_values"]])
            uncond_embeddings = self.encode_clip(uncond_img)

            x = torch.cat([uncond_embeddings, x])

        self.scheduler.set_timesteps(steps)
        scheduler_steps = self.scheduler.timesteps
        latents = (
            torch.randn((x.shape[0], 4, 64, 64), device=self.device)
            * self.scheduler.init_noise_sigma
        )

        for t in scheduler_steps:
            latent_model_input = self.scheduler.scale_model_input(latents, t)
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
        self, config, model_image_config, clip_dim=1024, device=torch.device("cpu")
    ):
        super().__init__()

        self.config = config
        sd = torch.load(config.pretrain_mbm_path, map_location="cpu")
        config_pretrain = sd["config"]

        self.mae, self.num_voxels, self.patch_embed = create_fmri_mae(
            config, sd, config_pretrain, model_image_config, clip_dim, device
        )

        # Freeze Model so we're only learning linear layers
        for param in self.mae.parameters():
            param.requires_grad = False

        self.map_dims = nn.Conv1d(73, 257, 1)
        self.unmap_dims = nn.ConvTranspose1d(257, 73, 1)
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
        latent = self.map_dims(latent)
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

            x = x + cross_x

            x = self.norm(x)
            # print(f"{x.shape=}")

            # print(f"{x.shape=}, {self.unmap_dims(x).shape=}")
            latent = self.decoder(x)
            latent = self.unmap_dims(latent)
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


class fMRI_CLIP_Cond_LDM(pl.LightningModule):
    def __init__(
        self,
        mae_config,
        cross_attention_config,
        ldm_config,
        ldm_ckpt_path,
        mae_tokens=292,
        clip_dim=1024,
        ddim_steps=250,
    ):
        super().__init__()

        self.mae_config = mae_config
        self.cross_attention_config = cross_attention_config
        self.ldm_config = ldm_config

        ### Masked Autoencoder ##
        sd = torch.load(mae_config.pretrain_mbm_path, map_location="cpu")
        mae_config_pretrain = sd["config"]
        self.mae, self.num_voxels, self.patch_embed = create_fmri_mae(
            self.mae_config,
            sd,
            mae_config_pretrain,
            self.cross_attention_config,
            clip_dim,
        )
        # for param in self.mae.parameters():
        #     param.requires_grad = False
        self.freeze_mae()  # don't want to freeze cross attention layers

        ### Cross Attention Layers ##
        # self.map_dims = nn.Conv1d(292, 1, 1)
        # self.unmap_dims = nn.ConvTranspose1d(1, 292, 1)
        # self.encoder = nn.Linear(mae_config_pretrain.embed_dim, clip_dim)
        # self.decoder = nn.Linear(clip_dim, mae_config_pretrain.embed_dim)
        # self.cross_blocks = nn.ModuleList(
        #     [
        #         ViTMAELayer(cross_attention_config, True)
        #         for _ in range(cross_attention_config.num_cross_encoder_layers)
        #     ]
        # )
        # self.norm = nn.LayerNorm(clip_dim)

        ### LDM ###
        self.ldm = instantiate_from_config(ldm_config.model)
        pl_sd = torch.load(ldm_ckpt_path, map_location="cpu")["state_dict"]

        m, u = self.ldm.load_state_dict(pl_sd, strict=False)
        self.ldm.cond_stage_trainable = True
        self.ldm.cond_stage_model = cond_stage_model(
            mae_tokens, clip_dim, cond_dim=512, global_pool=False
        )
        self.ldm.ddim_steps = ddim_steps
        self.ldm.re_init_ema()
        self.ldm.p_channels = ldm_config.model.params.channels
        self.ldm.p_image_size = ldm_config.model.params.image_size
        self.ldm.ch_mult = (
            ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult
        )
        self.ldm.freeze_whole_model()
        self.ldm.unfreeze_cond_stage()
        self.ldm.train_cond_stage_only = True

        ### CLIP ###
        self.clip_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        # self.clip_model.requires_grad_(False)

    def on_fit_start(self):
        self.ldm.cond_stage_model.to(self.device)

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        fmri = batch["fmri"]
        # print(image.shape, fmri.shape)
        # torch.Size([2, 256, 256, 3]) torch.Size([2, 1, 4656])
        # Generate CLIP Embeddings
        clip_embeddings = self.get_clip_embeddings(image)
        # image_support = self.apply_cross_attention(clip_embeddings)

        # Reconstruct fMRI
        loss_fmri_recon, pred, mask = self.mae(
            fmri,
            encoder_only=False,
            # image_support=image_support,
            image_support=clip_embeddings,
        )

        # Generate image with fMRI conditioning
        fmri_latents = self.mae(fmri, encoder_only=True)
        # print(f"{fmri_latents.shape=}")
        x, _ = self.ldm.get_input(batch, self.ldm.first_stage_key)
        loss_img_recon, cc = self.ldm(x, fmri_latents)

        # Loss Calculations
        loss = (
            self.mae_config.fmri_recon_weight * loss_fmri_recon
            + self.mae_config.img_recon_weight * loss_img_recon
        )

        self.log("train/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/fmri_loss", loss_fmri_recon, on_epoch=True, logger=True)
        self.log("train/img_loss", loss_img_recon, on_epoch=True, logger=True)

        return loss

    # def validation_step(self, batch, batch_idx):
    #     image = batch["image"]
    #     fmri = batch["fmri"]
    #     # print(image.shape, fmri.shape)
    #     # torch.Size([2, 256, 256, 3]) torch.Size([2, 1, 4656])
    #     # Generate CLIP Embeddings
    #     clip_embeddings = self.get_clip_embeddings(image)
    #     # image_support = self.apply_cross_attention(clip_embeddings)

    #     # Reconstruct fMRI
    #     loss_fmri_recon, pred, mask = self.mae(
    #         fmri,
    #         encoder_only=False,
    #         # image_support=image_support,
    #         image_support=clip_embeddings,
    #     )

    #     # Generate image with fMRI conditioning
    #     fmri_latents = self.mae(fmri, mask_ratio=0, encoder_only=True)
    #     # print(f"{fmri_latents.shape=}")
    #     x, _ = self.ldm.get_input(batch, self.ldm.first_stage_key)
    #     loss_img_recon, cc = self.ldm(x, fmri_latents)

    #     # Loss Calculations
    #     loss = (
    #         self.mae_config.fmri_recon_weight * loss_fmri_recon
    #         + self.mae_config.img_recon_weight * loss_img_recon
    #     )

    #     self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("val/fmri_loss", loss_fmri_recon, on_epoch=True, logger=True)
    #     self.log("val/img_loss", loss_img_recon, on_epoch=True, logger=True)
    #     return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def get_clip_embeddings(self, image):
        inputs = self.clip_processor(
            images=image, do_rescale=False, return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        clip_embeddings = self.clip_model(pixel_values).last_hidden_state
        return clip_embeddings

    def apply_cross_attention(self, clip_embeddings):
        cross_x = x.clone()

        for blk in self.cross_blocks:
            # print(f"{cross_x.shape=}")
            cross_x_full = blk(cross_x, hidden_states_mod2=clip_embeddings)
            cross_x = cross_x_full[0]

        x = x + cross_x

        x = self.norm(x)

        return x

    def freeze_mae(self):
        self.mae.patch_embed.requires_grad_(False)
        self.mae.cls_token.requires_grad_(False)
        self.mae.pos_embed.requires_grad_(False)
        self.mae.blocks.requires_grad_(False)

        self.mae.decoder_embed.requires_grad_(False)
        self.mae.mask_token.requires_grad_(False)
        self.mae.decoder_blocks.requires_grad_(False)
        self.mae.decoder_norm.requires_grad_(False)
        self.mae.decoder_pred.requires_grad_(False)

    def generate_conditioned_image(self, condition):
        # image_recon = self.generate_conditioned_image(fmri_latents)
        # print(f"{image.shape=}, {image_recon.shape=}")
        # loss_img_recon = (image-image_recon.permute(0, 2, 3, 1)) ** 2

        sampler = PLMSSampler(self.ldm)

        shape = (
            self.ldm_config.model.params.channels,
            self.ldm_config.model.params.image_size,
            self.ldm_config.model.params.image_size,
        )

        c = self.ldm.cond_stage_model(condition)
        # c = condition

        samples_ddim, _ = sampler.sample(
            S=250,  # ddim_steps,
            conditioning=c,
            batch_size=c.shape[0],
            shape=shape,
            verbose=False,
        )

        x_samples_ddim = self.ldm.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        return x_samples_ddim
