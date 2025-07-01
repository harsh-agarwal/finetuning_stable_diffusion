import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

class LoRALinearLayer(nn.Module):
    """Custom LoRA linear layer for KQV attention"""
    def __init__(self, in_features, out_features, rank=16, alpha=16, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)
        
        # Original weights (frozen)
        self.original_weight = None
        self.original_bias = None
        
    def forward(self, x):
        if self.original_weight is None:
            # First call - store original weights
            if hasattr(self, 'weight'):
                self.original_weight = self.weight.data.clone()
                self.original_bias = self.bias.data.clone() if hasattr(self, 'bias') else None
                # Freeze original weights
                self.weight.requires_grad = False
                if hasattr(self, 'bias'):
                    self.bias.requires_grad = False
        
        # Original forward pass
        original_output = nn.functional.linear(x, self.original_weight, self.original_bias)
        
        # LoRA forward pass
        lora_output = self.lora_dropout(x)
        lora_output = nn.functional.linear(lora_output, self.lora_A.T, None)
        lora_output = nn.functional.linear(lora_output, self.lora_B.T, None)
        
        return original_output + lora_output * self.scale

def get_lora_config(rank):
    """Returns LoRA configuration for diffusers UNet"""
    return {
        "r": rank,
        "lora_alpha": rank,
        "target_modules": ["to_k", "to_q", "to_v"],
        "lora_dropout": 0.0,
    }

def apply_lora_to_unet(unet, lora_config):
    """Applies LoRA to UNet using custom LoRA layers for KQV attention"""
    rank = lora_config["r"]
    lora_alpha = lora_config["lora_alpha"]
    target_modules = lora_config["target_modules"]
    lora_dropout = lora_config["lora_dropout"]
    
    # Apply LoRA to attention layers
    for name, module in unet.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Get the dimensions
                in_features = module.in_features
                out_features = module.out_features
                
                # Create LoRA layer
                lora_layer = LoRALinearLayer(
                    in_features=in_features,
                    out_features=out_features,
                    rank=rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout
                )
                
                # Copy original weights to LoRA layer
                lora_layer.original_weight = module.weight.data.clone()
                if module.bias is not None:
                    lora_layer.original_bias = module.bias.data.clone()
                
                # Replace the module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = unet.get_submodule(parent_name)
                setattr(parent_module, child_name, lora_layer)
    
    return unet

def save_lora_weights(unet, output_dir):
    """Saves LoRA weights from UNet"""
    lora_weights = {}
    
    # Collect all LoRA weights
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinearLayer):
            lora_weights[f"{name}.lora_A"] = module.lora_A
            lora_weights[f"{name}.lora_B"] = module.lora_B
    
    # Save weights
    torch.save(lora_weights, f"{output_dir}/lora_weights.pt")
    print(f"LoRA weights saved to {output_dir}/lora_weights.pt") 