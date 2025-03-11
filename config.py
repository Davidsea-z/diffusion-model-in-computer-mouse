from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128
    train_batch_size = 16
    eval_batch_size = 16
    num_epochs = 120
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"
    output_dir = "computer-mouse"
    push_to_hub = True
    hub_private_repo = False
    overwrite_output_dir = True
    seed = 0
    dataset_name = "tokakimiku/computer_mouse"

config = TrainingConfig()