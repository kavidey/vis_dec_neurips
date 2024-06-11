import wandb
import torch
import copy
import numpy as np
import os
import importlib
import torchvision.transforms as transforms
from einops import rearrange, repeat
import pytorch_lightning as pl


def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, "README.md"), "w+") as f:
        print(config.__dict__, file=f)


def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0] * sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)


def add_weight_decay(models, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for model in models:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config


class wandb_logger:
    def __init__(self, config):
        wandb.init(
            project="vis-dec",
            group="stepA_sc-mbm_crossatt_singlesub",
            anonymous="allow",
            config=config,
            reinit=True,
            name=config.wandb_name,
        )

        self.config = config
        self.step = None

    def log(self, name, data, step=None):
        if step is None:
            wandb.log({name: data})
        else:
            wandb.log({name: data}, step=step)
            self.step = step

    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def log_image(self, name, fig):
        if self.step is None:
            wandb.log({name: wandb.Image(fig)})
        else:
            wandb.log({name: wandb.Image(fig)}, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)


def create_trainer(
    num_epoch,
    precision=32,
    accumulate_grad_batches=2,
    logger=None,
    check_val_every_n_epoch=0,
):
    acc = "gpu" if torch.cuda.is_available() else "cpu"
    return pl.Trainer(
        accelerator=acc,
        max_epochs=num_epoch,
        logger=logger,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_checkpointing=False,
        enable_model_summary=False,
        gradient_clip_val=0.5,
        check_val_every_n_epoch=check_val_every_n_epoch,
        # strategy="ddp",
        # fast_dev_run=True,
    )


from generative_models.sgm.util import append_dims


def unclip_recon(
    x,
    diffusion_engine,
    vector_suffix,
    num_samples=1,
    offset_noise_level=0.04,
    device=torch.device("cuda"),
):
    assert x.ndim == 3
    if x.shape[0] == 1:
        x = x[[0]]
    with torch.no_grad(), torch.cuda.amp.autocast(
        dtype=torch.float16
    ), diffusion_engine.ema_scope():
        z = torch.randn(num_samples, 4, 96, 96).to(
            device
        )  # starting noise, can change to VAE outputs of initial image for img2img

        # clip_img_tokenized = clip_img_embedder(image)
        # tokens = clip_img_tokenized
        token_shape = x.shape
        tokens = x
        c = {
            "crossattn": tokens.repeat(num_samples, 1, 1),
            "vector": vector_suffix.repeat(num_samples, 1),
        }

        tokens = torch.randn_like(x)
        uc = {
            "crossattn": tokens.repeat(num_samples, 1, 1),
            "vector": vector_suffix.repeat(num_samples, 1),
        }

        for k in c:
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

        noise = torch.randn_like(z)
        sigmas = diffusion_engine.sampler.discretization(
            diffusion_engine.sampler.num_steps
        )
        sigma = sigmas[0].to(z.device)

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(
            1.0 + sigmas[0] ** 2.0
        )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x * 0.8 + 0.2), min=0.0, max=1.0)
        # samples = torch.clamp((samples_x + .5) / 2.0, min=0.0, max=1.0)
        return samples


def save_model_merge_conf(
    config,
    epoch,
    model,
    optimizer,
    checkpoint_paths,
    config_merge=None,
    checkpoint_file_name=None,
):
    os.makedirs(checkpoint_paths, exist_ok=True)
    to_save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "config_merge": config_merge,
    }
    if checkpoint_file_name is None:
        torch.save(to_save, os.path.join(checkpoint_paths, "checkpoint_mergconf.pth"))
    else:
        torch.save(to_save, os.path.join(checkpoint_paths, checkpoint_file_name))


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, "h w c -> c h w")
    img = torch.tensor(img)
    img = img * 2.0 - 1.0  # to -1 ~ 1
    return img


class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img


def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, "c h w -> h w c")
