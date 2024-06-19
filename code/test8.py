# %%
import os
from diffusers import (
    VersatileDiffusionImageVariationPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
)
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
import requests
from io import BytesIO
from PIL import Image
import torchvision.transforms.functional as F

# %%
model_id = "shi-labs/versatile-diffusion"
device = "cuda"
# pipe = VersatileDiffusionImageVariationPipeline.from_pretrained(
#     model_id, torch_dtype=torch.float16
# )
# pipe = pipe.to("cuda")
# %%
# download an initial image
url = "https://huggingface.co/datasets/diffusers/images/resolve/main/benz.jpg"
response = requests.get(url)
ref_image = Image.open(BytesIO(response.content)).convert("RGB")

# %%
# generator = torch.Generator(device="cuda").manual_seed(0)
# image = pipe(image, generator=generator).images[0]
# # image.save("./car_variation.png")
# image
# %%
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="image_unet")
unet.to(device)

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
vae.requires_grad_(False)
vae.to(device)

scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    model_id, subfolder="image_encoder"
)
image_encoder.to(device)

image_feature_extractor = CLIPImageProcessor.from_pretrained(
    model_id, subfolder="image_feature_extractor"
)
# %%
steps = 50
# %%
def normalize_embeddings(encoder_output):
    embeds = image_encoder.vision_model.post_layernorm(encoder_output.last_hidden_state)
    embeds = image_encoder.visual_projection(embeds)
    embeds_pooled = embeds[:, 0:1]
    embeds = embeds / torch.norm(embeds_pooled, dim=-1, keepdim=True)
    return embeds
# %%
with torch.no_grad():
    # image_input = image_feature_extractor(F.to_tensor(F.resize(ref_image, 256)), return_tensors="pt").pixel_values
    image_input = image_feature_extractor(F.to_tensor(F.resize(ref_image, 256)), return_tensors="pt", do_rescale=False).pixel_values
    image_input = image_input.to(device)
    image_embeddings = image_encoder(image_input)
    image_embeddings = normalize_embeddings(image_embeddings)

    # condition = torch.rand((1, 77, 768), device=device)
    # real_condition = pipe.encode_prompt(
    #     "photo of a cat",
    #     device=device,
    #     num_images_per_prompt=1,
    #     do_classifier_free_guidance=False,
    # )[0]

    # latent.shape, image_embeddings.shape, timesteps
    # torch.Size([1, 4, 64, 64]) torch.Size([2, 257, 768]) tensor([981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741, 721,
    #     701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461, 441,
    #     421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181, 161,
    #     141, 121, 101,  81,  61,  41,  21,   1], device='cuda:0')

    scheduler.set_timesteps(steps)
    scheduler_steps = scheduler.timesteps
    latents = torch.randn((1, 4, 64, 64), device=device) * scheduler.init_noise_sigma

    for t in scheduler_steps:
        latent_model_input = scheduler.scale_model_input(latents, t)
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=image_embeddings,
            return_dict=False,
        )[0]
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
# %%
image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
image = (image / 2 + 0.5).clamp(0, 1)
F.to_pil_image(image[0])
# %%