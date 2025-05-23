import os
import json
import wget
import concurrent.futures

# Function to download the image using wget
def download_image(url, output_dir):
    try:
        # Get the image file name from the URL
        file_name = os.path.basename(url)
        
        # Create the output path
        output_path = os.path.join(output_dir, file_name)
        
        # Download the image
        print(f"Downloading {file_name}...")
        wget.download(url, output_path)
        print(f"Downloaded {file_name} to {output_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

# Function to process the JSON and extract URLs
def extract_urls_from_json(json_data):
    urls = []
    for item in json_data:
        file_url = item.get("file", "")
        if file_url:
            urls.append(file_url)
    return urls

# Function to handle multithreaded downloading
def download_images_concurrently(json_file, output_dir, num_threads=4):
    # Read JSON data
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Extract image URLs from the JSON
    urls = extract_urls_from_json(json_data)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use ThreadPoolExecutor for multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit download tasks to the executor
        futures = [executor.submit(download_image, url, output_dir) for url in urls]
        
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()  # This will raise any exceptions caught during execution

# Main entry point
if __name__ == "__main__":
    # Path to the JSON file containing image URLs
    json_file = 'mastani/Frames.json'  # Change to your JSON file path

    # Directory to save the downloaded images
    output_dir = 'downloaded_images'  # Change to your desired output directory

    # Start the image downloading process
    download_images_concurrently(json_file, output_dir)
