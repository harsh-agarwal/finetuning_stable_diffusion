import torch
from torchvision import transforms

def filter_valid_caption(example, caption_column):
    caption = example[caption_column]
    if isinstance(caption, list):
        caption = caption[0] if caption else ""
    return caption is not None and isinstance(caption, str) and caption.strip() != ""

def get_image_transforms(resolution, center_crop, random_flip):
    transform_list = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
    ]
    if random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    return transforms.Compose(transform_list)

def preprocess_example(example, image_column, caption_column, image_transforms, tokenizer):
    image = example[image_column]
    if not isinstance(image, torch.Tensor):
        image = transforms.functional.pil_to_tensor(image)
    image = image_transforms(image)
    caption = example[caption_column]
    if isinstance(caption, list):
        caption = caption[0] if caption else ""
    example["pixel_values"] = image
    example["input_ids"] = tokenizer(
        caption,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[0]
    return example 