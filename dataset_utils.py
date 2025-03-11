from datasets import load_dataset
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import requests

# Cache for storing loaded images
image_cache = {}

def load_and_prepare_dataset(dataset_name, split="train"):
    """Load dataset from HuggingFace and prepare it for use"""
    return load_dataset(dataset_name, split=split)

def display_sample_images(dataset, num_images=5):
    """Display sample images from the dataset"""
    for i, item in enumerate(dataset):
        if 'image' in item:
            image_data = item['image']
            try:
                if isinstance(image_data, Image.Image):
                    init_image = image_data
                elif isinstance(image_data, bytes):
                    init_image = Image.open(BytesIO(image_data))
                elif isinstance(image_data, dict) and 'path' in image_data:
                    init_image = Image.open(image_data['path'])
                elif isinstance(image_data, str):
                    init_image = Image.open(image_data)
                else:
                    print(f"Unsupported image data format at index {i}: {type(image_data)}")
                    continue

                plt.figure()
                plt.imshow(init_image)
                plt.title(f'Image {i}')
                plt.axis('off')
                plt.show()

                if i >= num_images - 1:
                    break
            except Exception as e:
                print(f"Error processing image at index {i}: {str(e)}")
                continue

def prepare_image(image_data):
    """Prepare image for model input with caching"""
    # Generate a cache key based on the image data
    cache_key = str(hash(str(image_data)))
    
    # Check if image is already in cache
    if cache_key in image_cache:
        return image_cache[cache_key]
    
    # Load and process the image if not in cache
    if isinstance(image_data, str):
        response = requests.get(image_data)
        init_image = Image.open(BytesIO(response.content))
    elif isinstance(image_data, Image.Image):
        init_image = image_data
    else:
        return None

    if init_image.mode != 'RGB':
        init_image = init_image.convert('RGB')
    
    processed_image = init_image.resize((512, 512))
    
    # Store in cache
    image_cache[cache_key] = processed_image
    
    return processed_image