# Author: Ayush Thakur, Github - @ayush-thakur02
# Link: bio.link/ayush_thakur02

import gradio as gr

from diffusers import DiffusionPipeline
import torch

import base64
from io import BytesIO
import os
import gc

from PIL import Image

from torch.nn import DataParallel

class UNetDataParallel(DataParallel):
    def forward(self, *inputs, **kwargs):     
        inputs = inputs[0], inputs[1].item()
        return super().forward(*inputs, **kwargs)


model_dir = os.getenv("SDXL_MODEL_DIR")

if model_dir:
    model_key_base = os.path.join(model_dir, "stable-diffusion-xl-base-1.0")
    model_key_refiner = os.path.join(model_dir, "stable-diffusion-xl-refiner-1.0")
else:
    model_key_base = "stabilityai/stable-diffusion-xl-base-1.0"
    model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Process environment variables

# Use refiner (enabled by default)
enable_refiner = os.getenv("ENABLE_REFINER", "true").lower() == "true"
# Output images before the refiner and after the refiner
output_images_before_refiner = os.getenv("OUTPUT_IMAGES_BEFORE_REFINER", "false").lower() == "true"

offload_base = os.getenv("OFFLOAD_BASE", "true").lower() == "true"
offload_refiner = os.getenv("OFFLOAD_REFINER", "true").lower() == "true"

default_num_images = int(os.getenv("DEFAULT_NUM_IMAGES", "4"))
if default_num_images < 1:
    default_num_images = 1

print("Loading model", model_key_base)
pipe = DiffusionPipeline.from_pretrained(model_key_base, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

multi_gpu = os.getenv("MULTI_GPU", "false").lower() == "true"

if multi_gpu:
    pipe.unet = UNetDataParallel(pipe.unet)
    pipe.unet.config, pipe.unet.dtype, pipe.unet.add_embedding = pipe.unet.module.config, pipe.unet.module.dtype, pipe.unet.module.add_embedding
    pipe.to("cuda")
else:
    if offload_base:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")


if enable_refiner:
    print("Loading model", model_key_refiner)
    pipe_refiner = DiffusionPipeline.from_pretrained(model_key_refiner, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    if multi_gpu:
        pipe_refiner.unet = UNetDataParallel(pipe_refiner.unet)
        pipe_refiner.unet.config, pipe_refiner.unet.dtype, pipe_refiner.unet.add_embedding = pipe_refiner.unet.module.config, pipe_refiner.unet.module.dtype, pipe_refiner.unet.module.add_embedding
        pipe_refiner.to("cuda")
    else:
        if offload_refiner:
            pipe_refiner.enable_model_cpu_offload()
        else:
            pipe_refiner.to("cuda")

is_gpu_busy = False
def infer(prompt, negative, scale, samples=4, steps=25, refiner_strength=0.3, seed=-1):
    prompt, negative = [prompt] * samples, [negative] * samples
    g = torch.Generator(device="cuda")
    if seed != -1:
        g.manual_seed(seed)
    else:
        g.seed()

    images_b64_list = []

    if not enable_refiner or output_images_before_refiner:
        images = pipe(prompt=prompt, negative_prompt=negative, guidance_scale=scale, num_inference_steps=steps, generator=g).images
    else:
        images = pipe(prompt=prompt, negative_prompt=negative, guidance_scale=scale, num_inference_steps=steps, output_type="latent", generator=g).images

    gc.collect()
    torch.cuda.empty_cache()

    if enable_refiner:
        if output_images_before_refiner:
            for image in images:
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                image_b64 = f"data:image/jpeg;base64,{img_str}"
                images_b64_list.append(image_b64)

        images = pipe_refiner(prompt=prompt, negative_prompt=negative, image=images, num_inference_steps=steps, strength=refiner_strength, generator=g).images

        gc.collect()
        torch.cuda.empty_cache()

    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_b64 = (f"data:image/jpeg;base64,{img_str}")
        images_b64_list.append(image_b64)

    return images_b64_list

block = gr.Blocks()

with block:
    gr.Markdown("""
<div align="center" markdown=1>

# Stable Diffuion XL Web

### Github - @ayush-thakur02

[![BioLink](https://img.shields.io/badge/bio.link-000000%7D?style=for-the-badge&logo=biolink&logoColor=white)](https://bio.link/ayush_thakur02)

</div>
""")
    with gr.Group():
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(
                        label="Enter your prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt"
                    )
                    negative = gr.Textbox(
                        label="Enter your negative prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter a negative prompt"
                    )
                btn = gr.Button("Generate image")
        gallery = gr.Gallery(
            label="Generated images", elem_id="gallery").style(grid=[2], height="auto")
        
        with gr.Accordion("Advanced settings", open=False):
            samples = gr.Slider(label="Images", minimum=1, maximum=max(4, default_num_images), value=default_num_images, step=1)
            steps = gr.Slider(label="Steps", minimum=1, maximum=150, value=25, step=1)
            if enable_refiner:
                refiner_strength = gr.Slider(label="Refiner Strength", minimum=0, maximum=1.0, value=0.3, step=0.1)
            else:
                refiner_strength = gr.Slider(label="Refiner Strength (refiner not enabled)", minimum=0, maximum=0, value=0, step=0)
            guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=50, value=7, step=0.1
            )
            seed = gr.Slider(
                label="Seed",
                minimum=-1,
                maximum=2147483647,
                value=-1,
                step=1
            )
        negative.submit(infer, inputs=[text, negative, guidance_scale, samples, steps, refiner_strength, seed], outputs=[gallery], postprocess=False)
        text.submit(infer, inputs=[text, negative, guidance_scale, samples, steps, refiner_strength, seed], outputs=[gallery], postprocess=False)
        btn.click(infer, inputs=[text, negative, guidance_scale, samples, steps, refiner_strength, seed], outputs=[gallery], postprocess=False)

block.queue().launch(share=True)