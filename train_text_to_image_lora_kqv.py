# coding=utf-8
"""
Main training script for Stable Diffusion LoRA (KQV only), using modular utilities.
"""
import argparse
import logging
import math
import os
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import DiffusionPipeline
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from dataset_utils import filter_valid_caption, get_image_transforms, preprocess_example
from lora_utils import get_peft_lora_config, apply_peft_lora, save_lora_weights
from model_utils import load_models
from validation_utils import run_validation

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA training script for Stable Diffusion, KQV only.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default = "stabilityai/stable-diffusion-2-1")
    parser.add_argument("--data_dir", type=str, default="./celeba")
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--caption_column", type=str, default="Smiling")
    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned-lora-kqv")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_epochs", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    vae, unet, tokenizer, text_encoder, noise_scheduler = load_models(args.pretrained_model_name_or_path)
    
    # Set up LoRA for KQV layers only using PEFT
    lora_config = get_peft_lora_config(args.lora_rank)
    unet = apply_peft_lora(unet, lora_config)

    # Load dataset using torchvision
    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.CelebA(
        root=args.data_dir,
        split='train',
        target_type='attr',
        transform=transform,
        download=True
    )

    # Create a custom dataset class to handle the text encoding
    class CelebADataset(torch.utils.data.Dataset):
        def __init__(self, celeba_dataset, tokenizer, caption_column):
            self.dataset = celeba_dataset
            self.tokenizer = tokenizer
            self.caption_column = caption_column
            self.attr_names = celeba_dataset.attr_names

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, attr = self.dataset[idx]
            # Convert attribute to text description
            attr_idx = self.attr_names.index(self.caption_column)
            caption = "a person" + (" smiling" if attr[attr_idx] else "")
            
            # Tokenize the caption
            input_ids = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

            return {
                "pixel_values": image,
                "input_ids": input_ids
            }

    # Create the dataset with text encoding
    dataset = CelebADataset(dataset, tokenizer, args.caption_column)

    # Create dataloader
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Optimizer and scheduler
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = torch.optim.AdamW(lora_layers, lr=args.learning_rate)
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare for accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # Training loop
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    for epoch in range(999):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                input_ids = batch["input_ids"].to(accelerator.device)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(input_ids)[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item(), "step": global_step})
                if global_step % args.checkpointing_steps == 0:
                    accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                if args.validation_prompt and global_step % (args.validation_epochs * len(train_dataloader)) == 0:
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        torch_dtype=vae.dtype,
                    )
                    pipeline = pipeline.to(accelerator.device)
                    run_validation(pipeline, args.validation_prompt, args.num_validation_images, args.output_dir, global_step)
                    torch.cuda.empty_cache()
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    # Save LoRA weights only
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora_weights(unet, args.output_dir)
        print(f"LoRA weights saved to {args.output_dir}")

if __name__ == "__main__":
    main() 