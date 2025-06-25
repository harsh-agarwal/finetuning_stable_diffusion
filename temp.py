from datasets import load_dataset
dataset = load_dataset("guangyil/laion-coco-aesthetic", split="train[:500]")
print(len(dataset))
print(dataset[0].keys())