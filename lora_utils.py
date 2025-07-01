from diffusers.models.attention_processor import LoRAAttnProcessor
import torch

def get_lora_config(rank):
    """Returns LoRA configuration for diffusers UNet"""
    return {
        "r": rank,
        "lora_alpha": rank,
        "target_modules": ["to_k", "to_q", "to_v"],
        "lora_dropout": 0.0,
    }

def apply_lora_to_unet(unet, lora_config):
    """Applies LoRA to UNet using diffusers' LoRAAttnProcessor"""
    # Get the rank from config
    rank = lora_config["r"]
    lora_alpha = lora_config["lora_alpha"]
    target_modules = lora_config["target_modules"]
    lora_dropout = lora_config["lora_dropout"]
    
    # Apply LoRA to attention layers
    for name, module in unet.named_modules():
        if any(target in name for target in target_modules):
            if hasattr(module, 'weight'):
                # Create LoRA attention processor
                lora_processor = LoRAAttnProcessor(
                    hidden_size=module.weight.shape[0],
                    cross_attention_dim=None,
                    rank=rank,
                    network_alpha=lora_alpha,
                    dropout=lora_dropout,
                )
                
                # Replace the module with LoRA processor
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = unet.get_submodule(parent_name)
                setattr(parent_module, child_name, lora_processor)
    
    return unet

def save_lora_weights(unet, output_dir):
    """Saves LoRA weights from UNet"""
    unet.save_attn_procs(output_dir) 