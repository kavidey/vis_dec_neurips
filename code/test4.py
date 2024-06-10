# %%
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
# %%
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image
import torch

pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip",
    torch_dtype=torch.float16,
    variation="fp16",
)
pipe = pipe.to("cuda")

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
init_image = load_image(url)

pipe(init_image).images[0]
# %%
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPVisionModel

model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

inputs = processor(images=init_image, return_tensors="pt")
image_embeddings = model(**inputs).pooler_output
# %%
# image_embeddings = image_embeddings.to("cuda").half()
# pipe.to("cuda")
# pipe(image_embeds=image_embeddings).images[0]
# %%
# if not isinstance(image, torch.Tensor):
#     image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

# image = image.to(device=device, dtype=dtype)
# image_embeds = self.image_encoder(image).image_embeds
image_embeddings = pipe.image_encoder(
    pipe.feature_extractor(init_image, return_tensors="pt").pixel_values.to("cuda")
)
# %%
pipe(image_embeds=image_embeddings.image_embeds).images[0]
# %%
pipe(image_embeds=image_embeddings.last_hidden_state).images[0]
# %%
#pip install git+https://github.com/huggingface/diffusers.git transformers accelerate
import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableUnCLIPImg2ImgPipeline

#Start the StableUnCLIP Image variations pipeline
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
)
pipe = pipe.to("cuda")

#Get image from URL
url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/tarsila_do_amaral.png"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")

#Pipe to make the variation
images = pipe(init_image).images
# images[0].save("tarsila_variation.png")
# https://github.com/Stability-AI/stablediffusion/blob/main/doc/UNCLIP.MD
# We finetuned SD 2.1 to accept a CLIP ViT-L/14 image embedding in addition to the text encodings. This means that the model can be used to produce image variations, but can also be combined with a text-to-image embedding prior to yield a full text-to-image model at 768x768 resolution.
# %%
model = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

inputs = processor(images=init_image, return_tensors="pt")
image_embeddings = model(**inputs).pooler_output
# %%
from diffusers.utils import load_image
import torchvision.transforms as transforms

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
import sys
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder

device = "cuda"

clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device)

clip_seq_dim = 256
clip_emb_dim = 1664

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
init_image = load_image(url)
tensor_img = transforms.Resize(256)(transforms.PILToTensor()(init_image))[None].float().to(device)
clip_embeddings = clip_img_embedder(tensor_img)
# %%
del clip_img_embedder
import torch
torch.cuda.empty_cache()
# %%
from generative_models.sgm.models.diffusion import DiffusionEngine
from omegaconf import OmegaConf
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
offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
sampler_config['params']['num_steps'] = 38

diffusion_engine = DiffusionEngine(network_config=network_config,
                       denoiser_config=denoiser_config,
                       first_stage_config=first_stage_config,
                       conditioner_config=conditioner_config,
                       sampler_config=sampler_config,
                       scale_factor=scale_factor,
                       disable_first_stage_autocast=disable_first_stage_autocast)
# set to inference
diffusion_engine.eval()
diffusion_engine.to(device)

batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
      "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
      "crop_coords_top_left": torch.zeros(1, 2).to(device)}
out = diffusion_engine.conditioner(batch)
vector_suffix = out["vector"].to(device)
print("vector_suffix", vector_suffix.shape)
# %%
from clip_ae.utils import unclip_recon
samples = unclip_recon(clip_embeddings, diffusion_engine, vector_suffix)
# %%
