import torch
from diffusers import StableDiffusionImg2ImgPipeline
import os
import zipfile
from PIL import Image
from dataset_utils import prepare_image

def setup_pipeline(model_id="runwayml/stable-diffusion-v1-5", device=None):
    """Initialize and setup the StableDiffusionImg2ImgPipeline"""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16
    )
    return pipe.to(device)

def generate_image(pipe, init_image, prompt, strength=0.75, guidance_scale=7.5):
    """Generate a new image using the pipeline"""
    return pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale
    ).images[0]

def process_dataset_images(pipe, dataset, prompt, output_dir="generated_images"):
    """Process all images in the dataset and generate new ones"""
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, item in enumerate(dataset):
        init_image = prepare_image(item['image'])
        if init_image is None:
            continue
            
        generated_image = generate_image(pipe, init_image, prompt)
        filename = f"{output_dir}/generated_{idx}.png"
        generated_image.save(filename)
        print(f"Saved {filename}")

def create_zip_archive(source_dir, zip_filename):
    """Create a ZIP archive of the generated images"""
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                zipf.write(
                    os.path.join(root, file),
                    arcname=file
                )
    return zip_filename