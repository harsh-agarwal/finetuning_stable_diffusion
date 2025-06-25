import os

def run_validation(pipeline, prompt, num_images, output_dir, global_step):
    pipeline.set_progress_bar_config(disable=True)
    images = []
    for _ in range(num_images):
        image = pipeline(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        images.append(image)
    for i, image in enumerate(images):
        image.save(os.path.join(output_dir, f"validation_{global_step}_{i}.png"))
    del pipeline 