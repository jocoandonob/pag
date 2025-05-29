import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting, ControlNetModel
from diffusers.utils import load_image
import gradio as gr
import os

# Global pipeline caches
pipeline_text2img = None
pipeline_img2img = None
pipeline_inpaint = None
pipeline_controlnet = None

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dtype(device):
    return torch.float16 if device.type == "cuda" else torch.float32

def generate_text2image(prompt, pag_scale, guidance_scale, num_inference_steps, seed):
    global pipeline_text2img
    if pipeline_text2img is None:
        device = get_device()
        pipeline_sdxl_text2img = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=get_dtype(device)
        )
        pipeline_text2img = AutoPipelineForText2Image.from_pipe(
            pipeline_sdxl_text2img,
            enable_pag=True,
            pag_applied_layers=["mid"]
        )
        if device.type == "cuda":
            pipeline_text2img.enable_model_cpu_offload()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = pipeline_text2img(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        pag_scale=pag_scale,
    ).images
    return images[0]

def generate_image2image(prompt, init_image, strength, pag_scale, guidance_scale, seed):
    global pipeline_img2img
    if pipeline_img2img is None:
        device = get_device()
        pipeline_sdxl_img2img = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=get_dtype(device)
        )
        pipeline_img2img = AutoPipelineForImage2Image.from_pipe(
            pipeline_sdxl_img2img,
            enable_pag=True,
            pag_applied_layers=["mid"]
        )
        if device.type == "cuda":
            pipeline_img2img.enable_model_cpu_offload()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = pipeline_img2img(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        pag_scale=pag_scale,
        generator=generator,
    ).images
    return images[0]

def generate_inpainting(prompt, init_image, mask_image, strength, pag_scale, guidance_scale, num_inference_steps, seed):
    global pipeline_inpaint
    if pipeline_inpaint is None:
        device = get_device()
        pipeline_sdxl_inpaint = AutoPipelineForInpainting.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=get_dtype(device)
        )
        pipeline_inpaint = AutoPipelineForInpainting.from_pipe(
            pipeline_sdxl_inpaint,
            enable_pag=True,
            pag_applied_layers=["mid"]
        )
        if device.type == "cuda":
            pipeline_inpaint.enable_model_cpu_offload()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = pipeline_inpaint(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        pag_scale=pag_scale,
        generator=generator,
    ).images
    return images[0]

def generate_controlnet(prompt, control_image, controlnet_scale, pag_scale, guidance_scale, num_inference_steps, seed):
    global pipeline_controlnet
    if pipeline_controlnet is None:
        device = get_device()
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=get_dtype(device)
        )
        pipeline_sdxl_controlnet = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=get_dtype(device)
        )
        pipeline_controlnet = AutoPipelineForText2Image.from_pipe(
            pipeline_sdxl_controlnet,
            enable_pag=True,
            pag_applied_layers=["mid"]
        )
        if device.type == "cuda":
            pipeline_controlnet.enable_model_cpu_offload()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = pipeline_controlnet(
        prompt=prompt,
        image=control_image,
        controlnet_conditioning_scale=controlnet_scale,
        num_inference_steps=num_inference_steps,
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
        
        with gr.TabItem("Inpainting"):
            with gr.Row():
                with gr.Column():
                    prompt_inpaint = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        value="A majestic tiger sitting on a bench"
                    )
                    init_image_inpaint = gr.Image(
                        label="Input Image",
                        type="pil"
                    )
                    mask_image = gr.Image(
                        label="Mask Image",
                        type="pil"
                    )
                    strength_inpaint = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.8,
                        step=0.1,
                        label="Strength"
                    )
                    pag_scale_inpaint = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        value=3.0,
                        step=0.1,
                        label="PAG Scale"
                    )
                    guidance_scale_inpaint = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.1,
                        label="Guidance Scale"
                    )
                    num_inference_steps_inpaint = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Number of Inference Steps"
                    )
                    seed_inpaint = gr.Number(
                        value=1,
                        label="Seed",
                        precision=0
                    )
                    generate_inpaint_btn = gr.Button("Generate Image")
                
                with gr.Column():
                    output_image_inpaint = gr.Image(label="Generated Image")
            
            # Set up the generation function
            generate_inpaint_btn.click(
                fn=generate_inpainting,
                inputs=[
                    prompt_inpaint, init_image_inpaint, mask_image, strength_inpaint,
                    pag_scale_inpaint, guidance_scale_inpaint, num_inference_steps_inpaint, seed_inpaint
                ],
                outputs=output_image_inpaint
            )
            
            # Add example
            gr.Examples(
                examples=[[
                    "A majestic tiger sitting on a bench",
                    "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png",
                    "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png",
                    0.8,
                    3.0,
                    7.5,
                    50,
                    1
                ]],
                inputs=[
                    prompt_inpaint, init_image_inpaint, mask_image, strength_inpaint,
                    pag_scale_inpaint, guidance_scale_inpaint, num_inference_steps_inpaint, seed_inpaint
                ]
            )
        
        with gr.TabItem("ControlNet"):
            with gr.Row():
                with gr.Column():
                    prompt_controlnet = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        value=""
                    )
                    control_image = gr.Image(
                        label="Control Image (Canny Edge)",
                        type="pil"
                    )
                    controlnet_scale = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="ControlNet Scale"
                    )
                    pag_scale_controlnet = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        value=3.0,
                        step=0.1,
                        label="PAG Scale"
                    )
                    guidance_scale_controlnet = gr.Slider(
                        minimum=0.0,
                        maximum=20.0,
                        value=0.0,
                        step=0.1,
                        label="Guidance Scale"
                    )
                    num_inference_steps_controlnet = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Number of Inference Steps"
                    )
                    seed_controlnet = gr.Number(
                        value=1,
                        label="Seed",
                        precision=0
                    )
                    generate_controlnet_btn = gr.Button("Generate Image")
                with gr.Column():
                    output_image_controlnet = gr.Image(label="Generated Image")
            generate_controlnet_btn.click(
                fn=generate_controlnet,
                inputs=[
                    prompt_controlnet, control_image, controlnet_scale,
                    pag_scale_controlnet, guidance_scale_controlnet,
                    num_inference_steps_controlnet, seed_controlnet
                ],
                outputs=output_image_controlnet
            )
            gr.Examples(
                examples=[[
                    "",
                    "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/pag_control_input.png",
                    0.5,
                    3.0,
                    0.0,
                    50,
                    1
                ]],
                inputs=[
                    prompt_controlnet, control_image, controlnet_scale,
                    pag_scale_controlnet, guidance_scale_controlnet,
                    num_inference_steps_controlnet, seed_controlnet
                ]
            )

if __name__ == "__main__":
    demo.launch(share=True) 