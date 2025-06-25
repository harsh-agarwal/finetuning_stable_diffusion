from peft import LoraConfig, get_peft_model

# Returns a PEFT LoraConfig for the given rank

def get_peft_lora_config(rank):
    return LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_k", "to_q", "to_v"],
        lora_dropout=0.0,
        bias="none",
        task_type="UNCONDITIONAL_IMAGE_GENERATION"  # Changed to match the task type for Stable Diffusion
    )

# Applies PEFT LoRA to the UNet

def apply_peft_lora(unet, lora_config):
    return get_peft_model(unet, lora_config)

def save_lora_weights(unet, output_dir):
    unet.save_attn_procs(output_dir) 