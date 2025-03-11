from model_config import ModelConfig

# Select your model
model_name = 'stable-diffusion-v1-5'  # Change this to switch models
#- 'stable-diffusion-v1-5'
#- 'stable-diffusion-2-1'
#- 'stable-diffusion-xl'
model_config = ModelConfig.get_model_config(model_name)

# Use the configuration
print(f"Using model: {model_config['name']}")
print(f"Default resolution: {model_config['default_resolution']}")