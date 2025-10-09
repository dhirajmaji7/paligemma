# PaliGemma

This repository contains the implementation of the PaliGemma Vision-Language Model (VLM) entirely from scratch, integrating the SigLIP vision encoder and Gemma language model into a unified multimodal transformer with prefix-LM masking, grouped/multi-query attention, rotary embeddings, and contrastive pretraining objectives for cross-modal alignment.
It has a modular multimodal inference framework featuring KV cache optimization, top-p sampling, and weight tying between embedding and LM head layers — enabling efficient autoregressive text generation from image inputs and prompts.

## Key Topics

- Vision Transformer Models (SigLip)
- Language Models (Gemma)
- Constrastive Loss (CLIP, SigLip)
- KV Cache (Prefilling and token generation)
- Grouped-Query and Multi-Query Attention
- Rotary Positional Embeddings
- Weight Tying for Embedding layers
- Top-P Sampling for generation

## Directory Structure

- `model_siglip.py` — SigLIP model implementation.
- `model_gemma.py` — PaliGemma and Gemma model implementation.
- `processing_paligemma.py` — Image and data processing utilities.
- `load_model.py` — Utilities for loading models and weights.
- `inference.py` — Main script for running inference with the model.
- `launch_inference.sh` — Shell script to launch inference with arguments.
- `paligemma-3b-pt-224/` — Model weights and configuration files.
- `test_images/` — Example images for testing.

## Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:dhirajmaji7/paligemma.git
   cd paligemma
   ```

2. **Docker**

    You can use the dev-containers to run the model inference, or create your own image using the dockerfile - `.devcontainer/Dockerfile.inference` and mount this repository inside your container.



## Usage

### Model Files

You can get the weights for Paligemma model from Huggingface (Git LFS needs to be installed)
```
sudo apt install git-lfs -y
git lfs install
git clone https://huggingface.co/google/paligemma-3b-pt-224
```

The model files should be in the `paligemma-3b-pt-224/` directory. Ensure all required files (e.g., `.safetensors`, and `config.json`) are present.

### Run Inference

You can run inference using the provided shell script. Modify the arguments in the bash script and run:

```bash
bash launch_inference.sh
```

## Results

<img src="test_images/image1.jpeg" alt="Sample test image 1" width="300" />

Describe the image: Taj Mahal


<img src="test_images/image2.jpeg" alt="Sample test image 2" width="300" />

Describe the image: dog playing with a ball


## References

1. https://arxiv.org/abs/2407.07726 - Paligemma paper

2. https://huggingface.co/blog/paligemma - Blog about Paligemma

3. https://github.com/google-research/big_vision - Contains more supporting materials and links to more resources and papers in the field. Paligemma also supports object detection and segmentation.


