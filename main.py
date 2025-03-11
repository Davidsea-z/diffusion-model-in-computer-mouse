from config import config
from dataset_utils import load_and_prepare_dataset, display_sample_images, prepare_image
from model_utils import setup_pipeline, process_dataset_images, create_zip_archive

def main():
    # Load dataset
    dataset = load_and_prepare_dataset(config.dataset_name)
    
    # Display sample images (optional)
    display_sample_images(dataset)
    
    # Setup the pipeline
    pipe = setup_pipeline()
    
    # Define the prompt for image generation
    prompt = "stylish, computer mouse, front view, no logo, high quality, flat design"
    
    # Process all images in the dataset
    output_dir = "generated_images"
    process_dataset_images(pipe, dataset, prompt, output_dir)
    
    # Create ZIP archive of generated images
    zip_filename = "generated_images.zip"
    create_zip_archive(output_dir, zip_filename)
    print(f"Generated images are zipped in {zip_filename}")

if __name__ == "__main__":
    main()