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
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel
import torchvision.transforms.functional as F

# %%
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda:6"
# device = "cpu"

# %%
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)
# %%
# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt, num_inference_steps=250).images[0]
# image
# %%
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
# for param in [p for n, p in unet.named_parameters() if "attn2" not in n]:
#     param.requires_grad = False
# unet.train()
unet.to(device)

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
vae.requires_grad_(False)
vae.to(device)

scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
# %%
clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
clip_model.to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# %%
# inputs = clip_processor(
#     images=torch.rand((256, 256, 3)), do_rescale=False, return_tensors="pt"
# )
# pixel_values = inputs["pixel_values"].to(device)
# clip_embeddings = clip_model(pixel_values).last_hidden_state
# %%
# l = (
#     vae.encode(torch.rand((1, 3, 256, 256), device=device)).latent_dist.sample()
#     * vae.config.scaling_factor
# )
# unet(
#     l,
#     scheduler.config.num_train_timesteps,
#     encoder_hidden_states=torch.rand((1, 100, 768), device=device),
# )
# %%
# vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)
# init_latents = pipe.vae.encode(image)
# init_latents = init_latents.latent_dist.sample()
# init_latents = pipe.vae.config.scaling_factor * init_latents
# %%
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
pipe = StableDiffusionPipeline(
    vae=vae,
    unet=unet,
    scheduler=scheduler,
    feature_extractor=None,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    safety_checker=None,
)
pipe.to(device)
# %%
steps = 50
# %%
with torch.no_grad():
    condition = torch.rand((1, 77, 768), device=device)
    real_condition = pipe.encode_prompt(
        "photo of a cat",
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )[0]

    scheduler.set_timesteps(steps)
    scheduler_steps = scheduler.timesteps
    latents = torch.randn((1, 4, 64, 64), device=device) * scheduler.init_noise_sigma

    for t in scheduler_steps:
        latent_model_input = scheduler.scale_model_input(latents, t)
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=condition,
            return_dict=False,
        )[0]
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
# %%
with torch.no_grad():
    latents_scaled = latents / vae.config.scaling_factor
    image = vae.decode(latents_scaled).sample.detach()
# image = pipe.image_processor.postprocess(image)[0]
# image
image = (image / 2 + 0.5).clamp(0, 1)
F.to_pil_image(image[0])
# %%
pipe(num_inference_steps=steps, prompt_embeds=condition).images[0]
# %%
