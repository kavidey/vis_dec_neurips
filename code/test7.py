# %%
import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    VersatileDiffusionPipeline,
    UNet2DConditionModel,
    PNDMScheduler,
)
from transformers import CLIPProcessor, CLIPVisionModel

# %%
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)
# %%
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=250).images[0]
image
# %%
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
for param in [p for n, p in unet.named_parameters() if "attn2" not in n]:
    param.requires_grad = False
unet.train()

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
vae.requires_grad_(False)

scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
# %%
clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# %%
inputs = clip_processor(
    images=torch.rand((256, 256, 3)), do_rescale=False, return_tensors="pt"
)
pixel_values = inputs["pixel_values"]
clip_embeddings = clip_model(pixel_values).last_hidden_state
# %%
l = vae.encode(torch.rand((1, 3, 256, 256))).latent_dist.sample() * vae.config.scaling_factor
unet(l, scheduler.config.num_train_timesteps, encoder_hidden_states=torch.rand((1,100,768)))
# %%
# vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)
# init_latents = pipe.vae.encode(image)
# init_latents = init_latents.latent_dist.sample()
# init_latents = pipe.vae.config.scaling_factor * init_latents