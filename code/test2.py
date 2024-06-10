import torch
from diffusers import StableDiffusionPipeline, SDXLAutoencoderPipeline

# Load the pre-trained SDXL model and unclip pipeline
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
unclip_pipe = StableDiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16
)

# Load the SDXL autoencoder for CLIP embedding processing
autoencoder = SDXLAutoencoderPipeline.from_pretrained(model_id)


def generate_from_clip_embeddings(clip_embedding, num_inference_steps=50):
    """
    Generates an image from a provided CLIP embedding.

    Args:
        clip_embedding: A tensor of shape (batch_size, 768) representing the CLIP embedding.
        num_inference_steps: Number of diffusion steps for image generation (default: 50).

    Returns:
        A list of generated images as PIL Images.
    """
    # Preprocess CLIP embedding with the autoencoder
    latent_denoise = autoencoder(clip_embedding).latent_denoise

    # Generate image using unclip pipeline with latent as input
    images = unclip_pipe(
        latent=latent_denoise, num_inference_steps=num_inference_steps
    ).images

    return images


# Example usage
clip_embedding = torch.randn(1, 768)  # Replace with your actual CLIP embedding
generated_images = generate_from_clip_embeddings(clip_embedding)
