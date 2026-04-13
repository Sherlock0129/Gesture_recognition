import urllib.request
import zipfile
import os
import ssl

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

def download_and_extract(url, zip_path, extract_dir):
    if not os.path.exists(extract_dir):
        print(f"Downloading dataset from {url} ... (This may take a few minutes)")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        print("Cleaning up zip file...")
        os.remove(zip_path)
        print(f"Dataset successfully extracted to: {extract_dir}")
    else:
        print(f"Dataset directory '{extract_dir}' already exists. Skipping download.")

if __name__ == "__main__":
    # 使用著名的 Laurence Moroney 提供的小型“石头剪刀布”合成数据集 (约 200 MB)
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip"
    
    # 目标目录
    target_dir = "dataset"
    zip_file = "rps.zip"
    
    # Check if rps directory exists inside dataset
    if not os.path.exists(os.path.join(target_dir, 'rps')):
        os.makedirs(target_dir, exist_ok=True)
        print(f"Downloading dataset from {dataset_url} ... (This may take a few minutes)")
        urllib.request.urlretrieve(dataset_url, zip_file)
        
        print("Extracting zip file...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        print("Cleaning up zip file...")
        os.remove(zip_file)
        print(f"Dataset successfully extracted to: {target_dir}")
    else:
        print(f"Dataset directory '{target_dir}/rps' already exists. Skipping download.")
    
    print("\nDataset structure:")
    for root, dirs, files in os.walk(os.path.join(target_dir, 'rps')):
        if not dirs: # leaf directories
            print(f"- {root}: {len(files)} images")
