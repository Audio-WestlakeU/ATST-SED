import os.path
import os
from tqdm import tqdm
import requests
from pathlib import Path

def download_from_url(url, destination):
    if os.path.exists(destination):
        print("Skipping download as file in {} exists already".format(destination))
        return
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    os.makedirs(Path(destination).parent, exist_ok=True)
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")