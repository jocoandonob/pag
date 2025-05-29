import torch
from diffusers import AutoPipelineForText2Image
import gradio as gr
import os

# Initialize the pipeline
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the base pipeline
    pipeline_sdxl = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    
    # Create PAG-enabled pipeline
    pipeline = AutoPipelineForText2Image.from_pipe(
        pipeline_sdxl,
        enable_pag=True,
        pag_applied_layers=["mid"]
    )
    
    if device.type == "cuda":
        pipeline.enable_model_cpu_offload()
    
    return pipeline

# Load the model
pipeline = load_model()

def generate_image(prompt, pag_scale, guidance_scale, num_inference_steps, seed):
    # Set the seed for reproducibility
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # Generate the image
    images = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        pag_scale=pag_scale,
    ).images
    
    return images[0]

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="Enter your prompt here...",
            value="an insect robot preparing a delicious meal, anime style"
        ),
        gr.Slider(
            minimum=0.0,
            maximum=5.0,
            value=3.0,
            step=0.1,
            label="PAG Scale"
        ),
        gr.Slider(
            minimum=1.0,
            maximum=20.0,
            value=7.0,
            step=0.1,
            label="Guidance Scale"
        ),
        gr.Slider(
            minimum=1,
            maximum=100,
            value=25,
            step=1,
            label="Number of Inference Steps"
        ),
        gr.Number(
            value=0,
            label="Seed",
            precision=0
        )
    ],
    outputs=gr.Image(label="Generated Image"),
    title="PAG Image Generator",
    description="Generate images using Stable Diffusion XL with Prompt-Aware Guidance (PAG)",
    examples=[
        ["an insect robot preparing a delicious meal, anime style", 3.0, 7.0, 25, 0],
        ["a futuristic cityscape with flying cars, cyberpunk style", 2.0, 7.5, 30, 42],
        ["a magical forest with glowing mushrooms and fairies", 4.0, 8.0, 35, 123],
    ]
)

if __name__ == "__main__":
    demo.launch(share=True) 