# Stable Diffusion XL with PAG Image Generator

This application uses Stable Diffusion XL with Prompt-Aware Guidance (PAG) to generate images. It demonstrates the effect of different PAG scales on image generation.

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- At least 8GB of VRAM

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script with the default prompt:
```bash
python generate_images.py
```

The script will:
1. Download the Stable Diffusion XL model (if not already downloaded)
2. Generate images with different PAG scales (0.0 and 3.0)
3. Save the generated images in the `output` directory

## Customization

You can modify the `generate_images.py` script to:
- Change the prompt
- Adjust PAG scales
- Modify generation parameters (steps, guidance scale, etc.)
- Change the output directory

## Output

The generated images will be saved in the `output` directory with filenames indicating their PAG scale:
- `image_pag_0.0_0.png`: Image generated with PAG scale 0.0
- `image_pag_3.0_0.png`: Image generated with PAG scale 3.0 