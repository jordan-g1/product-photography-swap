# Product Photography Swap - ComfyUI Workflow

An advanced ComfyUI workflow for seamlessly swapping products in photography using AI-powered segmentation and style transfer. This workflow leverages FLUX models, GroundingDINO, and SAM for automated object detection and replacement.

## Features

- **Automated Object Detection**: Uses GroundingDINO + SAM for precise object segmentation
- **Style-Aware Replacement**: Employs FLUX Redux for contextually appropriate product swapping
- **High-Quality Inpainting**: Utilizes FLUX Fill models for seamless integration
- **Optimized Performance**: Nunchaku models for faster inference

## How It Works

1. **Upload Base Image**: Load your original product photograph
2. **Upload Reference Image**: Provide the new product you want to swap in
3. **Define Target Object**: Specify what product you want to replace (e.g., "bag", "shoes", "bottle")

The workflow automatically:

- Detects and segments the target object using GroundingDINO
- Creates precise masks with SAM (Segment Anything Model)
- Applies style transfer using FLUX Redux
- Generates the final swapped image with FLUX Fill inpainting

## Installation

Download and place these into the appropriate models folders:

**Flux Models:**

- `flux1-redux-dev.safetensors` (style model)
- `FLUX1/ae.safetensors` (VAE)
- `svdq-int4-flux.1-fill-dev` (Nunchaku model)

**CLIP Models:**

- `ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors`
- `t5xxl_fp16.safetensors`
- `sigclip_vision_patch14_384.safetensors`

**LoRA:**

- `FLUX.1-Turbo-Alpha.safetensors`
- `comfyui_subject_lora16.safetensors` (Subject LoRA)

**Segmentation Models:**

- `GroundingDINO_SwinT_OGC (694MB)`
- `sam_vit_h_4b8939.pth`

## Required Custom Nodes

Most can be installed through ComfyUI node manager. Included in this repo is the ResizeImageToNearest64AspectRatio node. Add it to your `custom_nodes` folder.

## Improving Quality

- Try adjusting the mask settings (expanding or contracting depending on use case)
- Try adjusting flux redux settings
- Test out number of inference steps

Feel free to dm me with any questions u/Original_Caramel2510 on reddit.
