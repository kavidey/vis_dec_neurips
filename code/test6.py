# %%
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torchvision.transforms.functional as F
from einops import rearrange, repeat
from dc_ldm.models.diffusion.plms import PLMSSampler

from clip_ae.utils import instantiate_from_config
from clip_ae.conditioned_ldm import cond_stage_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
model_path = Path("../models") / "ldm" / "cin256"
config_path = model_path / "cin256.yaml"
ckp_path = model_path / "model.ckpt"

config = OmegaConf.load(config_path)
# config.model.params.unet_config.params.use_time_cond = use_time_cond
# config.model.params.unet_config.params.global_pool = global_pool

cond_dim = config.model.params.unet_config.params.context_dim

model = instantiate_from_config(config.model)
pl_sd = torch.load(ckp_path, map_location="cpu")["state_dict"]

m, u = model.load_state_dict(pl_sd, strict=False)
model.cond_stage_trainable = True
model.cond_stage_model = cond_stage_model(292, 1024, cond_dim=512, global_pool=False)
# model.cond_stage_model = cond_stage_model(292, 1024, cond_dim=1280, global_pool=True)
# %%
model.ddim_steps = 250
model.re_init_ema()
# if logger is not None:
#     logger.watch(model, log="all", log_graph=False)

model.p_channels = config.model.params.channels
model.p_image_size = config.model.params.image_size
model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

fmri_latent_dim = model.cond_stage_model.fmri_latent_dim

model.to(device)

sampler = PLMSSampler(model)
# %%
num_samples = 1

with model.ema_scope():
    model.eval()
    latent = torch.rand((3, 292, 1024))
    assert latent.shape[-1] == fmri_latent_dim, 'dim error'

    shape = (
        config.model.params.channels,
        config.model.params.image_size,
        config.model.params.image_size,
    )

    c = model.cond_stage_model(latent.to(device))
    # c = torch.zeros((2, 256, 512)).to(device).float()
    samples_ddim, _ = sampler.sample(
        S=1,  # ddim_steps,
        conditioning=c,
        batch_size=c.shape[0],
        shape=shape,
        verbose=False,
    )

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

# img = 255. * rearrange(x_samples_ddim, 'b c h w -> b h w c').cpu().numpy()
# Image.fromarray(img.astype(np.uint8))
for i,sample in enumerate(x_samples_ddim):
    img = F.to_pil_image(sample.cpu())
    img.save(f"test6-{i}.png")
# %%
