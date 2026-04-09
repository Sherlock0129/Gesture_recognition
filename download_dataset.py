import urllib.request
import zipfile
import os

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
    dataset_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip"
    
    # 目标目录
    target_dir = "dataset"
    zip_file = "rps.zip"
    
    os.makedirs(target_dir, exist_ok=True)
    download_and_extract(dataset_url, zip_file, target_dir)
    
    print("\nDataset structure:")
    for root, dirs, files in os.walk(os.path.join(target_dir, 'rps')):
        if not dirs: # leaf directories
            print(f"- {root}: {len(files)} images")
