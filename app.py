import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image
import gradio as gr
import os

# Initialize the pipelines
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize text2image pipeline
    pipeline_sdxl_text2img = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    
    # Create PAG-enabled text2image pipeline
    pipeline_text2img = AutoPipelineForText2Image.from_pipe(
        pipeline_sdxl_text2img,
        enable_pag=True,
        pag_applied_layers=["mid"]
    )
    
    # Initialize image2image pipeline
    pipeline_sdxl_img2img = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    
    # Create PAG-enabled image2image pipeline
    pipeline_img2img = AutoPipelineForImage2Image.from_pipe(
        pipeline_sdxl_img2img,
        enable_pag=True,
        pag_applied_layers=["mid"]
    )
    
    if device.type == "cuda":
        pipeline_text2img.enable_model_cpu_offload()
        pipeline_img2img.enable_model_cpu_offload()
    
    return pipeline_text2img, pipeline_img2img

# Load the models
pipeline_text2img, pipeline_img2img = load_models()

def generate_text2image(prompt, pag_scale, guidance_scale, num_inference_steps, seed):
    # Set the seed for reproducibility
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # Generate the image
    images = pipeline_text2img(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        pag_scale=pag_scale,
    ).images
    
    return images[0]

def generate_image2image(prompt, init_image, strength, pag_scale, guidance_scale, seed):
    # Set the seed for reproducibility
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # Generate the image
    images = pipeline_img2img(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        pag_scale=pag_scale,
        generator=generator,
    ).images
    
    return images[0]

# Create the Gradio interface
with gr.Blocks(title="PAG Image Generator") as demo:
    gr.Markdown("# PAG Image Generator")
    gr.Markdown("Generate images using Stable Diffusion XL with Prompt-Aware Guidance (PAG)")
    
    with gr.Tabs():
        with gr.TabItem("Text to Image"):
            with gr.Row():
                with gr.Column():
                    prompt_text2img = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        value="an insect robot preparing a delicious meal, anime style"
                    )
                    pag_scale_text2img = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        value=3.0,
                        step=0.1,
                        label="PAG Scale"
                    )
                    guidance_scale_text2img = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.0,
                        step=0.1,
                        label="Guidance Scale"
                    )
                    num_inference_steps = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=25,
                        step=1,
                        label="Number of Inference Steps"
                    )
                    seed_text2img = gr.Number(
                        value=0,
                        label="Seed",
                        precision=0
                    )
                    generate_text2img_btn = gr.Button("Generate Image")
                
                with gr.Column():
                    output_image_text2img = gr.Image(label="Generated Image")
            
            # Set up the generation function
            generate_text2img_btn.click(
                fn=generate_text2image,
                inputs=[prompt_text2img, pag_scale_text2img, guidance_scale_text2img, num_inference_steps, seed_text2img],
                outputs=output_image_text2img
            )
            
            # Add examples
            examples_text2img = [
                ["an insect robot preparing a delicious meal, anime style", 3.0, 7.0, 25, 0],
                ["a futuristic cityscape with flying cars, cyberpunk style", 2.0, 7.5, 30, 42],
                ["a magical forest with glowing mushrooms and fairies", 4.0, 8.0, 35, 123],
            ]
            gr.Examples(examples=examples_text2img, inputs=[prompt_text2img, pag_scale_text2img, guidance_scale_text2img, num_inference_steps, seed_text2img])
        
        with gr.TabItem("Image to Image"):
            with gr.Row():
                with gr.Column():
                    prompt_img2img = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        value="a dog catching a frisbee in the jungle"
                    )
                    init_image = gr.Image(
                        label="Input Image",
                        type="pil"
                    )
                    strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.8,
                        step=0.1,
                        label="Strength"
                    )
                    pag_scale_img2img = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        value=4.0,
                        step=0.1,
                        label="PAG Scale"
                    )
                    guidance_scale_img2img = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.0,
                        step=0.1,
                        label="Guidance Scale"
                    )
                    seed_img2img = gr.Number(
                        value=0,
                        label="Seed",
                        precision=0
                    )
                    generate_img2img_btn = gr.Button("Generate Image")
                
                with gr.Column():
                    output_image_img2img = gr.Image(label="Generated Image")
            
            # Set up the generation function
            generate_img2img_btn.click(
                fn=generate_image2image,
                inputs=[prompt_img2img, init_image, strength, pag_scale_img2img, guidance_scale_img2img, seed_img2img],
                outputs=output_image_img2img
            )
            
            # Add example
            gr.Examples(
                examples=[[
                    "a dog catching a frisbee in the jungle",
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png",
                    0.8,
                    4.0,
                    7.0,
                    0
                ]],
                inputs=[prompt_img2img, init_image, strength, pag_scale_img2img, guidance_scale_img2img, seed_img2img]
            )

if __name__ == "__main__":
    demo.launch(share=True) 