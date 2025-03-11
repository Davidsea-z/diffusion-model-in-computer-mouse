class ModelConfig:
    """
    Configuration class for different diffusion models
    """
    MODELS = {
        'stable-diffusion-v1-5': {
            'name': 'runwayml/stable-diffusion-v1-5',
            'torch_dtype': 'float16',
            'default_resolution': (512, 512)
        },
        'stable-diffusion-2-1': {
            'name': 'stabilityai/stable-diffusion-2-1',
            'torch_dtype': 'float16',
            'default_resolution': (768, 768)
        },
        'stable-diffusion-xl': {
            'name': 'stabilityai/stable-diffusion-xl-base-1.0',
            'torch_dtype': 'float16',
            'default_resolution': (1024, 1024)
        }
    }

    @classmethod
    def get_model_config(cls, model_name='stable-diffusion-v1-5'):
        """
        Get configuration for a specific model
        """
        if model_name not in cls.MODELS:
            raise ValueError(f"Model {model_name} not found. Available models: {list(cls.MODELS.keys())}")
        return cls.MODELS[model_name]