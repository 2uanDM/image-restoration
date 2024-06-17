from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import requests
import os
from typing import Optional
from datasets import load_dataset

dataset = load_dataset("YangQiee/HQ-50K")

urls = [dataset["train"][i]["text"] for i in range(dataset["train"].num_rows)]

download_dir = "hq50k"
os.makedirs(download_dir, exist_ok=True)

futures = []

def download_image(idx, url, download_dir) -> Optional[str]:
    max_retries = 5
    file_name = idx
    
    while max_retries > 0:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            }
            response = requests.get(url, timeout=10, headers=headers)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            max_retries -= 1
            continue
        else:
            # Check if the image is valid
            if response.status_code != 200:
                print(f"Failed to download {url}")
                max_retries -= 1
                continue
            
            # Get the file extension from response headers
            extension = response.headers.get("content-type", "jpg").split("/")[-1]
            
            extensions = ["jpg", "jpeg", "png", "webp"]
            
            for ext in extensions:
                if ext in extension.lower():
                    extension = ext
                    break
            
            extension = 'jpg'
            
            # Save the image
            with open(f"{download_dir}/{file_name}.{extension}", "wb") as f:
                f.write(response.content)
            
            return os.path.join(download_dir, f"{file_name}.{extension}")
    
    if max_retries == 0:
        print(f"Failed to download {url}")
        return None
            

with ThreadPoolExecutor(max_workers=10) as executor:
    for download_idx, url in enumerate(urls):
        futures.append(executor.submit(download_image, download_idx, url, download_dir))
    
    for idx, future in enumerate(tqdm.tqdm(as_completed(futures), total=len(futures))):
        future.result()