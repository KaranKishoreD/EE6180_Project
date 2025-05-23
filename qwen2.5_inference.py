import os
import csv
import torch
import argparse
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

USER_TEXT = """
You are a highly capable vision-language model. Your task is to write a clear and descriptive caption for the given image.

Follow these rules:
1. Focus on describing what is visible in the image.
2. Be descriptive about the image.
3. Do not make assumptions about context not visible in the image.
5. Do not include special characters or quotes around your caption.

Example:
The image shows a determined female warrior in armor, drawing a bowstring with intense focus. She wears a helmet with protective chainmail covering the sides of her face, and a metallic armguard on her wrist. The background is dimly lit with warm tones, likely from torches or firelight, enhancing the dramatic atmosphere. Her gaze is fixed forward, suggesting she is targeting an enemy or preparing for battle in a historical or epic setting.
"""
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
print(hf_token)
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5 Image Captioning")
    parser.add_argument("--input-dir", required=True, help="Folder containing images")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use (default: 0)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Initialize model and processor
    model_name = "Qwen/Qwen2.5-Omni-7B"
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={"":device},  # assign to specific GPU
	# token = hf_token
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name, token = hf_token)

    # System instruction
    SYSTEM_TEXT = "You are a helpful assistant that generates concise captions for images. Focus only on what is visible."

    # Get image files
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    # Write header to CSV
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["Filename", "Caption"])

        for img_file in tqdm(image_files, desc="Captioning images"):
            img_path = os.path.join(args.input_dir, img_file)

            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_TEXT}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": USER_TEXT}
                    ]
                }
            ]

            text = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
                return_tensors="pt"
            )
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model.generate(**inputs, return_audio=False)
            decoded = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            print(f"🖼️{img_file}: {decoded}")
            writer.writerow([img_file, decoded])

if __name__ == "__main__":
    main()
