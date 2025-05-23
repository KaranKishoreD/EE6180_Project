
from groundingdino.util.inference import load_model, load_image, predict, annotate
import os
import supervision as sv
import torch
from datumaro import Polygon, Bbox, RleMask, Dataset, DatasetItem, Image
import cv2

HOME = "/data0/karan"
DINO_CONFIG = "/data0/karan/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS = "/data0/karan/GroundingDINO/weights/groundingdino_swint_ogc.pth"
model = load_model(DINO_CONFIG, DINO_WEIGHTS)

images = ["/data0/karan/input_images/data_subset/283.jpg"]
results = []

for image_name in images:
    image_path = os.path.join(HOME, "data", image_name)

    TEXT_PROMPT = "A bald man with a white mark on his forehead is wearing a white outfit. He is standing in a dimly lit environment with a soft light source behind him, creating a halo effect. The background is blurred with circular bokeh lights, suggesting a festive or ceremonial setting."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(image_path)

    # Generate predictions
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    # Annotate the image
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    # Save the annotated image to disk
    save_path = os.path.join(HOME, "output", f"annotated_{image_name}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, annotated_frame[:, :, ::-1])  # BGR to RGB conversion
    print(f"Saved annotated image to {save_path}")

    # Assemble annotations for Datumaro
    h, w, _ = image_source.shape
    annotations = []
    for box in boxes:
        result = box * torch.Tensor([w, h, w, h])
        x_min, y_min, x_max, y_max = result.tolist()
        annotations.append(Bbox(x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min, label=1))

    results.append(
        DatasetItem(
            id=image_name.split(".")[0],
            media=Image.from_file(path=image_path),
            annotations=annotations
        )
    )

# Create Datumaro dataset
dataset = Dataset.from_iterable(results, categories=[TEXT_PROMPT])