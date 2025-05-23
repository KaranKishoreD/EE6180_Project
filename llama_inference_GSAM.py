'''
import os
import argparse
import torch
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from tqdm.auto import tqdm
import csv
from PIL import Image, UnidentifiedImageError
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import shutil
import time

# Revised USER_TEXT prompt (prompt engineering reflected)
USER_TEXT = """You are a highly capable vision-language model. Write a clear, concise caption (2–3 sentences) describing only what is visible in the image, which is a scene from an Indian movie.

Rules:
1. Describe only visible content.
2. Do not infer beyond the image.
3. Avoid mentioning movie names, actors, or characters.
4. Do not include special characters or quotation marks.

Example:
In this vibrant scene from an Indian period drama, characters admire the splendor of a mirrored palace and discuss the upcoming festivities. The ruler’s brother is praised for his recent accomplishments, while the queen's presence adds warmth to the gathering. Traditional dance and celebration fill the space, highlighting the cultural richness and grandeur of the setting.
"""

def is_image_corrupt(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except (IOError, SyntaxError, UnidentifiedImageError):
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

    if not corrupt_images:
        print("No corrupt images found.")
        return

    os.makedirs(corrupt_folder, exist_ok=True)
    for img in tqdm(corrupt_images, desc="Moving corrupt images", unit="file",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
        shutil.move(img, os.path.join(corrupt_folder, os.path.basename(img)))

    print(f"Moved {len(corrupt_images)} corrupt images to {corrupt_folder}")

def get_image(image_path):
    return Image.open(image_path).convert('RGB')

def load_descriptor(descriptor_path, image_filename):
    """
    Load descriptor text for an image. Matches base name of image file to descriptor txt file.
    E.g., 'image1.jpg' -> 'descriptor_path/image1.txt'
    """
    base_name = os.path.splitext(image_filename)[0]
    descriptor_file = os.path.join(descriptor_path, base_name + ".txt")
    if os.path.isfile(descriptor_file):
        try:
            with open(descriptor_file, "r", encoding="utf-8") as f:
                descriptor_text = f.read().strip()
            return descriptor_text
        except Exception as e:
            print(f"Warning: Could not read descriptor file {descriptor_file}: {e}")
            return None
    else:
        return None

def llama_progress_bar(total, desc, position=0):
    bar_format = "{desc}: |{bar}| {percentage:3.0f}% [{n_fmt}/{total_fmt}, {rate_fmt}{postfix}]"
    return tqdm(total=total, desc=desc, position=position, bar_format=bar_format, ascii="🦙·")

def process_images(rank, world_size, args, model_name, input_files, output_csv):
    device = f"cuda:{rank}"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        device_map={0: device},
        torch_dtype=torch.bfloat16,
        use_auth_token=args.hf_token
    )
    model.to(device)
    model.eval()

    processor = MllamaProcessor.from_pretrained(model_name, use_auth_token=args.hf_token)

    chunk_size = len(input_files) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - 1 else len(input_files)

    if not os.path.isfile(output_csv):
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Caption'])

    pbar = llama_progress_bar(total=end_idx - start_idx, desc=f"GPU {rank}", position=rank)

    for filename in input_files[start_idx:end_idx]:
        image_path = os.path.join(args.input_path, filename)
        image = get_image(image_path)

        # Load descriptor text if descriptor_path given and exists
        descriptor_text = None
        if args.descriptor_path:
            descriptor_text = load_descriptor(args.descriptor_path, filename)

        # Build prompt with explicit descriptor explanation
        if descriptor_text:
            prompt_text = (
                f"{USER_TEXT}"
                "You can additionally use the object descriptors from the image (object labels, rectangular bounding boxes with 4 coordinates, polygon approximation of bounding box):\n"
                f"{descriptor_text}\n\n"
                "Write a clear and detailed caption describing only what is visible.\n\n"
            )
        else:
            prompt_text = USER_TEXT

        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
        ]

        prompt = processor.apply_chat_template(
            conversation,
            add_special_tokens=False,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = processor(image, prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, temperature=1, top_p=0.9, max_new_tokens=512)

        decoded_output = processor.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        print(f"prompt: {prompt_text}")
        print(f"File: {filename}: {decoded_output}")

        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([filename, decoded_output])

        pbar.update(1)
        pbar.set_postfix({"Last File": filename})

    pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Image Captioning with Descriptors")
    parser.add_argument("--hf_token", required=True, help="Hugging Face API token")
    parser.add_argument("--input_path", required=True, help="Path to input image folder")
    parser.add_argument("--output_path", required=True, help="Path to output CSV folder")
    parser.add_argument("--num_gpus", type=int, required=True, help="Number of GPUs to use")
    parser.add_argument("--corrupt_folder", default="corrupt_images", help="Folder to move corrupt images")
    parser.add_argument("--descriptor_path", default=None, help="Folder containing descriptor text files for images")
    args = parser.parse_args()

    model_name = "meta-llama/Llama-3.2-11b-Vision-Instruct"

    print("🦙 Starting image processing pipeline...")
    start_time = time.time()

    # Move corrupt images to separate folder
    corrupt_folder = os.path.join(args.input_path, args.corrupt_folder)
    find_and_move_corrupt_images(args.input_path, corrupt_folder)

    input_files = sorted([
        f for f in os.listdir(args.input_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and
        os.path.isfile(os.path.join(args.input_path, f))
    ])

    print(f"[DEBUG] Remaining images after removing corrupt ones: {len(input_files)}")
    print(f"[DEBUG] Filenames: {input_files}")

    print(f"\n🦙 Processing {len(input_files)} images using {args.num_gpus} GPUs...")

    mp.set_start_method('spawn', force=True)
    processes = []

    os.makedirs(args.output_path, exist_ok=True)

    for rank in range(args.num_gpus):
        output_csv = os.path.join(args.output_path, f"captions_gpu_{rank}.csv")
        p = mp.Process(target=process_images,
                       args=(rank, args.num_gpus, args, model_name, input_files, output_csv))
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
'''
import os
import argparse
import torch
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from tqdm.auto import tqdm
import csv
from PIL import Image, UnidentifiedImageError
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import shutil
import time
import pandas as pd

USER_TEXT = """You are a highly capable vision-language model. Your task is to write a clear and concise caption. 
Assume that the image is a scene from an Indian movie. This does not need to be stated in the caption.

Follow these rules:

1. Focus only on what is visually present in the image.
2. Be concise, using 2–3 sentences.
3. Do not infer or assume any context beyond what is visible.
4. Do not mention the movie name, actor names, or specific character names.
5. Do not include special characters or quotation marks around the caption.
"""

def is_image_corrupt(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except (IOError, SyntaxError, UnidentifiedImageError):
        return True

def find_and_move_corrupt_images(folder_path, corrupt_folder):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    num_cores = mp.cpu_count()
    with tqdm(total=len(image_files), desc="Checking for corrupt images", unit="file") as pbar:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(is_image_corrupt, image_files))
            pbar.update(len(image_files))

    corrupt_images = [img for img, is_corrupt in zip(image_files, results) if is_corrupt]

    if not corrupt_images:
        print("No corrupt images found.")
        return

    os.makedirs(corrupt_folder, exist_ok=True)
    for img in tqdm(corrupt_images, desc="Moving corrupt images", unit="file"):
        shutil.move(img, os.path.join(corrupt_folder, os.path.basename(img)))

    print(f"Moved {len(corrupt_images)} corrupt images to {corrupt_folder}")

def get_image(image_path):
    return Image.open(image_path).convert('RGB')

def load_descriptor_map(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    return {row['image']: row['descriptor'] for _, row in df.iterrows()}

def llama_progress_bar(total, desc, position=0):
    return tqdm(total=total, desc=desc, position=position, ascii="🦙·")

def process_images(rank, world_size, args, model_name, input_files, descriptor_map):
    device = f"cuda:{rank}"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        device_map={"":0},
        torch_dtype=torch.bfloat16,
        use_auth_token=args.hf_token
    )
    model.to(device)
    model.eval()

    processor = MllamaProcessor.from_pretrained(model_name, use_auth_token=args.hf_token)

    chunk_size = len(input_files) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - 1 else len(input_files)

    output_csv = os.path.join("/data0/karan/output_capts_llama3", "captions_GSAM_ojpspec.csv")
    if rank == 0 and not os.path.isfile(output_csv):
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Caption'])

    pbar = llama_progress_bar(total=end_idx - start_idx, desc=f"GPU {rank}", position=rank)

    for filename in input_files[start_idx:end_idx]:
        image_path = os.path.join(args.input_path, filename)
        image = get_image(image_path)

        descriptor_text = descriptor_map.get(filename, "None")

        prompt_text = (
            f"{USER_TEXT}"
            "You can use the following object descriptor extracted from image to inform your caption\n"
            "Each descriptor contains an object label, a rectangular bounding box [x_min, y_min, x_max, y_max] and a polygon outline as a set of points"
            "where x_min and y_min represent the top-left corner, and x_max and y_max represent the bottom-right corner of the box"
            f"{descriptor_text}\n\n"
            "Write a clear and detailed caption describing only what is visible. Ensure that you describe all the objects present in the descriptor in addition to other visible details accurately. Use full sentences. Avoid saying that the frame is from a movie.\n\n"
        )

        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
        ]

        prompt = processor.apply_chat_template(
            conversation,
            add_special_tokens=False,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = processor(image, prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, temperature=1, top_p=0.9, max_new_tokens=512)

        decoded_output = processor.decode(output[0], skip_special_tokens=False)[len(prompt):]

        print(f"prompt: {prompt_text}")
        print(f"File: {filename}: {decoded_output}")

        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([filename, decoded_output])

        pbar.update(1)
        pbar.set_postfix({"Last File": filename})

    pbar.close()

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Image Captioning with TSV Descriptors")
    parser.add_argument("--hf_token", required=True, help="Hugging Face API token")
    parser.add_argument("--input_path", required=True, help="Path to input image folder")
    parser.add_argument("--output_path", required=True, help="Folder to save output CSVs")
    parser.add_argument("--num_gpus", type=int, required=True, help="Number of GPUs to use")
    parser.add_argument("--corrupt_folder", default="corrupt_images", help="Folder to move corrupt images")
    parser.add_argument("--descriptor_path", required=True, help="Path to TSV file with image descriptors")
    args = parser.parse_args()

    model_name = "meta-llama/Llama-3.2-11b-Vision-Instruct"

    print("🦙 Starting image processing pipeline...")
    start_time = time.time()

    corrupt_folder = os.path.join(args.input_path, args.corrupt_folder)
    find_and_move_corrupt_images(args.input_path, corrupt_folder)

    input_files = sorted([
        f for f in os.listdir(args.input_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and
        os.path.isfile(os.path.join(args.input_path, f))
    ])

    print(f"[DEBUG] Remaining images after removing corrupt ones: {len(input_files)}")

    descriptor_map = load_descriptor_map(args.descriptor_path)

    mp.set_start_method('spawn', force=True)
    processes = []

    os.makedirs(args.output_path, exist_ok=True)

    for rank in range(args.num_gpus):
        p = mp.Process(target=process_images,
                       args=(rank, args.num_gpus, args, model_name, input_files, descriptor_map))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_time = time.time() - start_time
    print(f"\n🦙 Total processing time: {total_time:.2f} seconds")
    print("🦙 Image captioning completed successfully!")

if __name__ == "__main__":
    main()
