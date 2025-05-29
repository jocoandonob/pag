import torch
from diffusers import AutoPipelineForText2Image
import os
from PIL import Image

def generate_images(prompt, output_dir="output", pag_scales=[0.0, 3.0]):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA is not available. Running on CPU will be very slow.")
        print("Please install PyTorch with CUDA support for better performance.")
        print("You can install it using: pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html")
    
    # Initialize the base pipeline
    print("Loading base pipeline...")
    pipeline_sdxl = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    
    # Create PAG-enabled pipeline
    print("Creating PAG-enabled pipeline...")
    pipeline = AutoPipelineForText2Image.from_pipe(
        pipeline_sdxl,
        enable_pag=True,
        pag_applied_layers=["mid"]
    )
    
    # Enable model CPU offload only if CUDA is available
    if device.type == "cuda":
        pipeline.enable_model_cpu_offload()
    
    # Generate images for each PAG scale
    for pag_scale in pag_scales:
        print(f"Generating image with PAG scale: {pag_scale}")
        generator = torch.Generator(device=device).manual_seed(0)
        
        images = pipeline(
            prompt=prompt,
            num_inference_steps=25,
            guidance_scale=7.0,
            generator=generator,
            pag_scale=pag_scale,
        ).images
        
        # Save the generated image
        for idx, image in enumerate(images):
            output_path = os.path.join(output_dir, f"image_pag_{pag_scale}_{idx}.png")
            image.save(output_path)
            print(f"Saved image to: {output_path}")

if __name__ == "__main__":
    prompt = "an insect robot preparing a delicious meal, anime style"
    generate_images(prompt) 