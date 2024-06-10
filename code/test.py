# %%
import torch
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline

# Step 1: Install required libraries
# !pip install transformers diffusers
# %%
# Step 2: Load the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
unclip_pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# %%
# Example input
text = ["A photo of a cat"]
image = ["test.jpg"]  # You can use images instead of text if preferred
# %%
# Step 3: Generate CLIP embeddings
inputs = clip_processor(text=text, images=image, return_tensors="pt", padding=True)
outputs = clip_model(**inputs)
clip_embeddings = outputs.text_embeds if text else outputs.image_embeds


# Ensure the embeddings are on the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_embeddings = clip_embeddings.to(device)
unclip_pipeline = unclip_pipeline.to(device)

# Step 5: Pass the embeddings to the UnCLIP model
generated_image = unclip_pipeline(clip_embeddings=clip_embeddings)