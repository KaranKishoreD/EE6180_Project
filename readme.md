# EE6180 Advanced Topics in Artificial Intelligence - Image Captioning Project on Movie Frames

This repository contains a complete pipeline for generating descriptive captions from still frames of the movie *Bajirao Mastani*. The project includes object-aware captioning using models like LLaMA 3.2, Qwen 2.5, and Grounded Segment Anything + Grounding DINO.

---

## 1. Dataset Preparation

Download and organize the dataset by running:

```bash
python ./BajiraoMastani/BajiraoMastani/get_dataset.py
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

Make sure you create a `.env` file contains your `HUGGINGFACE_TOKEN`.

* Basic captioning:

  ```bash
  export HUGGINGFACE_TOKEN=$(grep HUGGINGFACE_TOKEN .env | cut -d '=' -f2)
  python LLaMa_inference.py --hf_token $HUGGINGFACE_TOKEN --input_path INPUT_FOLDER_PATH (./input_images/data_subset) --output_path OUTPUT_FOLDER_PATH --num_gpus NUMBER_OF_GPUS
  ```

* Captioning with GSAM descriptors:



  ```bash
  python llama_inference_GSAM.py --hf_token $HUGGINGFACE_TOKEN --input_path INPUT_FOLDER_PATH (./input_images/data_subset) --output_path OUTPUT_FOLDER_PATH --num_gpus NUMBER_OF_GPUS --descriptor_path PATH_TO_TSV/CSV_FILE_FOR_OBJECT_BOUNDING_BOX
  ```

---

## 3. Captioning with Qwen 2.5

Qwen 2.5 Omni is another vision-language model used for getting image captions.

### Environment Setup:

```bash
conda create -n qwen python=3.12 -y
conda activate qwen
pip install -r qwen_reqs.txt
```


### Run Qwen captioning:

```bash
python qwen2.5_inference.py --input-dir input_images --output-csv output/output_captions/Qwen2.5_captions.csv --gpu gpu_id
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

* `LLaMa3.2_Captions.csv`: LLaMA captions without getting object boundaries from SAM
* `LLaMa3.2_captions_GSAM.csv`: LLaMA captions with grounded object descriptors
* `Qwen2.5_captions.csv`: Captions using Qwen 2.5 model
* `Ensemble_Captions.tsv`: Merged captions using ensemble of following models: LLaMa + Qwen 2.5 + Phi4 + BLIP 
* `*_contains_objects_cleaned.csv`: Refined GSAM + caption outputs

Segmentation and object annotations are in:

* `descriptor_outputs/` : Annotations has rectangular bounding boxes around detected objects, polygon contours are generated using SAM and approximation of those contours is shown in polygon_approximation for the subset of 50 images listed in ./input_images/data_subset
* `output/outputs_SAM_GroundingDINO/`: Polygon boundaries, annotations, inverted mask and mask for different objects for a single image (illustration purpose)

---