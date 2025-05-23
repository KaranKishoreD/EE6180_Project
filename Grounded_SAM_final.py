import os
import sys
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv

from segment_anything import build_sam, SamPredictor
from huggingface_hub import hf_hub_download

# Local imports
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import load_image, predict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load GroundingDINO model
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filename = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
args = SLConfig.fromfile(hf_hub_download(repo_id=ckpt_repo_id, filename=ckpt_config_filename))
args.device = device
model = build_model(args)
checkpoint = torch.load(hf_hub_download(repo_id=ckpt_repo_id, filename=ckpt_filename), map_location=device)
model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
model.eval()

# Load SAM
SAM_CHECKPOINT = "/data0/karan/models/sam/sam_vit_h_4b8939.pth"
sam_predictor = SamPredictor(build_sam(checkpoint=SAM_CHECKPOINT).to(device))

# Paths
input_images_dir = "/data0/karan/input_images/data_subset"
captions_tsv_path = "/data0/karan/input_captions/Ensemble_Captions.tsv"
output_descriptor_dir = "descriptor_outputs"
os.makedirs(output_descriptor_dir, exist_ok=True)

poly_approx_dir = os.path.join(output_descriptor_dir, "polygon_approximation")
poly_contour_dir = os.path.join(output_descriptor_dir, "polygon_contours")
annotations_dir = os.path.join(output_descriptor_dir, "annotations")

os.makedirs(poly_approx_dir, exist_ok=True)
os.makedirs(poly_contour_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

output_descriptor_path = os.path.join(output_descriptor_dir, "batch_descriptors.tsv")

# Read captions TSV
image_captions = {}
with open(captions_tsv_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split('\t', 1)
        if len(parts) != 2:
            continue
        fname, caption_text = parts
        caption_text = caption_text.replace('\\n', '\n').strip()
        image_captions[fname] = caption_text

print(f"Loaded captions for {len(image_captions)} images.")

output_rows = []

for img_filename, caption_text in image_captions.items():
    print(f"\nProcessing image: {img_filename}")

    image_path = os.path.join(input_images_dir, img_filename)
    if not os.path.exists(image_path):
        print(f"Image file {image_path} not found, skipping.")
        continue

    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=caption_text,
        box_threshold=0.3,
        text_threshold=0.25
    )
    print(f"Detected {boxes.shape[0]} objects in {img_filename}.")

    if boxes.shape[0] == 0:
        print(f"No objects detected for {img_filename}, skipping descriptor.")
        output_rows.append({
            "image": img_filename,
            "descriptor": "[]"
        })
        continue

    sam_predictor.set_image(image_source)
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(device), image_source.shape[:2])
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    masks = masks.cpu()

    descriptors = []

    for i in range(masks.shape[0]):
        label = phrases[i]
        rect_box = boxes_xyxy[i].tolist()
        mask = masks[i, 0].numpy().astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Original contour polygon points
        orig_polygon = [pt[0].tolist() for pt in contours[0]]

        # Polygon approx with 1% epsilon of contour perimeter
        epsilon = 0.01 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)
        approx_polygon = [pt[0].tolist() for pt in approx]

        descriptor = {
            "label": label,
            "rect_box": [round(x, 2) for x in rect_box],
            "polygon": approx_polygon
        }
        descriptors.append(descriptor)

        # --- Save visualizations ---

        # 1) Polygon approximation overlay (poly_appr_<img>_<label>.png)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_source)
        patch = plt.Polygon(approx_polygon, closed=True, edgecolor='lime', facecolor='none', linewidth=2)
        ax.add_patch(patch)
        ax.set_title(f"Polygon Approximation: {label}")
        ax.axis('off')
        out_path_approx = os.path.join(poly_approx_dir, f"poly_appr_{img_filename}_{label.replace(' ', '_')}.png")
        fig.savefig(out_path_approx)
        plt.close(fig)

        # 2) Polygon contour overlay (poly_<img>_<label>.png)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_source)
        patch_orig = plt.Polygon(orig_polygon, closed=True, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(patch_orig)
        ax.set_title(f"Original Polygon Contour: {label}")
        ax.axis('off')
        out_path_contour = os.path.join(poly_contour_dir, f"poly_{img_filename}_{label.replace(' ', '_')}.png")
        fig.savefig(out_path_contour)
        plt.close(fig)

        # 3) Annotated image with mask overlay (annotated_<img>_<label>_<masknum>.png)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_source)
        # mask overlay in Reds colormap with alpha blending
        ax.imshow(mask * 255, alpha=0.4, cmap='Reds')
        # Draw bbox
        rect = plt.Rectangle(
            (rect_box[0], rect_box[1]),
            rect_box[2] - rect_box[0],
            rect_box[3] - rect_box[1],
            linewidth=2,
            edgecolor='yellow',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.set_title(f"Annotated: {label}")
        ax.axis('off')
        out_path_annot = os.path.join(annotations_dir, f"annotated_{img_filename}_{label.replace(' ', '_')}_{i}.png")
        fig.savefig(out_path_annot)
        plt.close(fig)

        print(f"Saved visualizations for object '{label}' in image '{img_filename}'.")

    # Save descriptors for this image
    descriptor_json_str = json.dumps(descriptors)
    output_rows.append({
        "image": img_filename,
        "descriptor": descriptor_json_str
    })

# Write batch descriptor TSV
with open(output_descriptor_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["image", "descriptor"], delimiter='\t')
    writer.writeheader()
    for row in output_rows:
        writer.writerow(row)

print(f"\nAll descriptors and images saved to '{output_descriptor_dir}' and '{output_descriptor_path}'.")
