import os
import argparse
import torch
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from tqdm.auto import tqdm
import csv
from PIL import Image
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import shutil
import time

USER_TEXT = """You are a highly capable vision-language model. Your task is to write a clear and concise caption for the given image, which is a scene from an Indian movie.

Follow these rules:

1. Focus only on what is visually present in the image.
2. Be concise, using 2–3 sentences.
3. Do not infer or assume any context beyond what is visible.
4. Do not mention the movie name, actor names, or specific character names.
5. Do not include special characters or quotation marks around the caption.

Example:
In this vibrant scene from an Indian period drama, characters admire the splendor of a mirrored palace and discuss the upcoming festivities. The ruler’s brother is praised for his recent accomplishments, while the queen's presence adds warmth to the gathering. Traditional dance and celebration fill the space, highlighting the cultural richness and grandeur of the setting.
"""
#USER_TEXT = """
#You are a highly capable vision-language model. Your task is to write a clear and concise caption for the given image.

#Follow these rules:
#1. Focus on describing what is visible in the image.
#2. Be concise, around 1–2 sentences.
#3. Do not make assumptions about context not visible in the image.
#5. Do not include special characters or quotes around your caption.

#Example:
#A man riding a bicycle down a mountain trail.
#"""

def is_image_corrupt(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except (IOError, SyntaxError, Image.UnidentifiedImageError):
        return True

def find_and_move_corrupt_images(folder_path, corrupt_folder):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    num_cores = mp.cpu_count()
    with tqdm(total=len(image_files), desc="Checking for corrupt images", unit="file",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(is_image_corrupt, image_files))
            pbar.update(len(image_files))

    corrupt_images = [img for img, is_corrupt in zip(image_files, results) if is_corrupt]

    os.makedirs(corrupt_folder, exist_ok=True)
    for img in tqdm(corrupt_images, desc="Moving corrupt images", unit="file",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
        shutil.move(img, os.path.join(corrupt_folder, os.path.basename(img)))

    print(f"Moved {len(corrupt_images)} corrupt images to {corrupt_folder}")

def get_image(image_path):
    return Image.open(image_path).convert('RGB')

def llama_progress_bar(total, desc, position=0):
    """Custom progress bar with llama emojis."""
    bar_format = "{desc}: |{bar}| {percentage:3.0f}% [{n_fmt}/{total_fmt}, {rate_fmt}{postfix}]"
    return tqdm(total=total, desc=desc, position=position, bar_format=bar_format, ascii="🦙·")

def process_images(rank, world_size, args, model_name, input_files, output_csv):
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        device_map="cuda:1",  # Use only cuda:1 (first GPU)
        torch_dtype=torch.bfloat16,
        token=args.hf_token
    )
    processor = MllamaProcessor.from_pretrained(model_name, token=args.hf_token)

    chunk_size = len(input_files) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - 1 else len(input_files)

    # Create CSV file and write header if it doesn't exist
    file_exists = os.path.isfile(output_csv)
    if not file_exists:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Caption'])

    pbar = llama_progress_bar(total=end_idx - start_idx, desc=f"GPU {rank}", position=rank)

    for filename in input_files[start_idx:end_idx]:
        image_path = os.path.join(args.input_path, filename)
        image = get_image(image_path)

        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_TEXT}]}
        ]

        prompt = processor.apply_chat_template(
            conversation,
            add_special_tokens=False,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = processor(image, prompt, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, temperature=1, top_p=0.9, max_new_tokens=512)
        decoded_output = processor.decode(output[0])[len(prompt):]

        print(f"🖼️ {filename}: {decoded_output.strip()}")

        # Append each result to CSV file
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([filename, decoded_output])

        pbar.update(1)
        pbar.set_postfix({"Last File": filename})

    pbar.close()

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Image Captioning")
    parser.add_argument("--hf_token", required=True, help="Hugging Face API token")
    parser.add_argument("--input_path", required=True, help="Path to input image folder")
    parser.add_argument("--output_path", required=True, help="Path to output CSV folder")
    parser.add_argument("--num_gpus", type=int, required=True, help="Number of GPUs to use")
    parser.add_argument("--corrupt_folder", default="corrupt_images", help="Folder to move corrupt images")
    args = parser.parse_args()

    model_name = "meta-llama/Llama-3.2-11b-Vision-Instruct"

    print("🦙 Starting image processing pipeline...")
    start_time = time.time()

    # Find and move corrupt images
    corrupt_folder = os.path.join(args.input_path, args.corrupt_folder)
    find_and_move_corrupt_images(args.input_path, corrupt_folder)

    print("HEY THERE!!")

    # Get list of remaining (non-corrupt) image files, sorted
    input_files = sorted([f for f in os.listdir(args.input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"[DEBUG] Remaining images after removing corrupt ones: {len(input_files)}")
    print(f"[DEBUG] Filenames: {input_files}")


    print(f"\n🦙 Processing {len(input_files)} images using {args.num_gpus} GPUs...")

    mp.set_start_method('spawn', force=True)
    processes = []

    for rank in range(args.num_gpus):
        output_csv = os.path.join(args.output_path, f"captions_gpu_{rank}.csv")
        p = mp.Process(target=process_images, args=(rank, args.num_gpus, args, model_name, input_files, output_csv))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n🦙 Total processing time: {total_time:.2f} seconds")
    print("🦙 Image captioning completed successfully!")

if __name__ == "__main__":
    main()
