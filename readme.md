# Bajirao Mastani: Multi-Stage Vision-Language Captioning Pipeline

This repository contains a complete pipeline for generating descriptive captions from still frames of the movie *Bajirao Mastani*. The project includes object-aware captioning using models like LLaMA 3.2, Qwen 2.5, and Grounded Segment Anything + Grounding DINO.

---

## 1. Dataset Preparation

Download and organize the dataset by running:

```bash
python get_dataset.py
```

This script processes the movie and stores frame metadata in the `BajiraoMastani/` directory including:

* `Frames.json`, `Clips.json`: Scene and frame annotations
* `Transcripts.json`: Dialogue data (optional)

---

## 2. Captioning with LLaMA 3.2

LLaMA is used for vision-language caption generation with and without object descriptors.

### Environment Setup:

```bash
conda create -n llama_inference python=3.12 -y
conda activate llama_inference
pip install -r llama_inf_reqs.txt
```

### Run LLaMA captioning:

* Basic captioning:

  ```bash
  python llama_inference.py
  ```

* Captioning with GSAM descriptors:

  ```bash
  python llama_inference_GSAM.py
  ```

---

## 3. Captioning with Qwen 2.5

Qwen 2.5 is another vision-language model used for captioning frames.

### Environment Setup:

```bash
conda create -n qwen python=3.12 -y
conda activate qwen
pip install -r qwen_reqs.txt
```

Make sure your `.env` file contains your `HUGGINGFACE_TOKEN`.

### Run Qwen captioning:

```bash
python qwen2.5_inference.py --input-dir input_images --output-csv output/output_captions/Qwen2.5_captions.csv
```

---

## 4. Object Grounding with GSAM + Grounding DINO

This stage extracts object labels, bounding boxes, and segmentation masks for every image.

### Environment Setup:

```bash
conda create -n sam_env python=3.12 -y
conda activate sam_env
pip install -r sam_env_reqs.txt
```

### Download Weights:

```bash
wget https://huggingface.co/IDEA-Research/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth -P models/
wget https://huggingface.co/facebook/segment-anything/resolve/main/sam_vit_h_4b8939.pth -P models/
```

### Run grounding + segmentation:

```bash
python Grounded_SAM_final.py
```

This generates `batch_descriptors.tsv` containing object descriptions for each image.

---

## Outputs

All final outputs are stored in `output/output_captions/`:

* `LLaMa3.2_Captions.csv`: LLaMA captions without objects
* `LLaMa3.2_captions_GSAM.csv`: LLaMA with grounded object descriptors
* `Qwen2.5_captions.csv`: Captions from Qwen model
* `Ensemble_Captions.tsv`: Optional merged captions
* `*_contains_objects_cleaned.csv`: Refined GSAM+caption outputs

Segmentation and object annotations are in:

* `descriptor_outputs/`
* `output/outputs_SAM_GroundingDINO/`

---

## Citation

This repository is part of a project to explore character and background consistency in cinematic storytelling.
