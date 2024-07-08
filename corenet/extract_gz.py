import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def extract_tar_file(filename, source_directory, target_base_directory):
    try:
        tar_path = os.path.join(source_directory, filename)
        target_directory = os.path.join(target_base_directory, os.path.splitext(os.path.splitext(filename)[0])[0])
        
        # 检查目标目录是否已经存在
        if os.path.exists(target_directory):
            return filename, "Skipped: Directory already exists"

        with tarfile.open(tar_path, 'r:gz') as tar:
            # 提取所有文件到目标基目录
            tar.extractall(path=target_base_directory)
        return filename, None
    except tarfile.ReadError as e:
        return filename, str(e)

def find_tar_files(directory):
    """递归查找目录中所有的tar.gz文件。"""
    tar_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tar.gz'):
                tar_files.append(os.path.relpath(os.path.join(root, file), directory))
    return tar_files

# 使用多线程进行解压缩
if __name__ == '__main__':
    source_directory = '/ML-A100/team/mm/models/laion2b/shard3_tar_gz'  # 当前目录
    target_base_directory = "/ML-A100/team/mm/models/laion2b/shard3"  # 替换为你的目标目录

    tar_files = find_tar_files(source_directory)
    
    if not os.path.exists(target_base_directory):
        os.makedirs(target_base_directory)

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 使用tqdm显示进度条
        future_to_file = {executor.submit(extract_tar_file, tar_file, source_directory, target_base_directory): tar_file for tar_file in tar_files}
        
        for future in tqdm(as_completed(future_to_file), total=len(tar_files), desc="Extracting tar files"):
            filename, error = future.result()
            results.append((filename, error))

    # 记录出错的文件
    error_log_path = os.path.join(target_base_directory, 'extraction_errors.log')
    with open(error_log_path, 'w') as f:
        for filename, error in results:
            if error is not None:
                f.write(f"{filename}: {error}\n")

    print("All tar files have been extracted to the target directory.")
