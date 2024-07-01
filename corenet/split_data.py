import os
import shutil
import multiprocessing
from tqdm import tqdm

# 定义原始目录和目标目录
source_directory = '/ML-A100/team/mm/models/food101/food101/images'
target_directory = '/ML-A100/team/mm/models/food101/food101/train_images'

# 确保目标目录存在
os.makedirs(target_directory, exist_ok=True)

# 读取txt文件路径
txt_file = '/ML-A100/team/mm/models/food101/food101/meta_data/train_full.txt'

def get_files(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    files = [line.strip().split()[0] for line in lines]
    return files

def copy_file(task):
    source_path, target_path = task
    
    # 检查目标文件是否存在，若存在则跳过
    if os.path.exists(target_path):
        return
    
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    try:
        shutil.copy2(source_path, target_path)
    except Exception as e:
        print(f"Error copying {source_path}: {e}")

def get_tasks(files, source_directory, target_directory):
    tasks = []
    for file_path in tqdm(files, desc="Preparing tasks"):
        source_path = os.path.join(source_directory, file_path)
        target_path = os.path.join(target_directory, file_path)
        task = (source_path, target_path)
        tasks.append(task)
    return tasks

def process_tasks_in_parallel(tasks, num_workers):
    with multiprocessing.Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(copy_file, tasks), total=len(tasks), desc="Copying files"):
            pass

if __name__ == '__main__':
    files = get_files(txt_file)
    tasks = get_tasks(files, source_directory, target_directory)
    num_workers = multiprocessing.cpu_count()
    process_tasks_in_parallel(tasks, num_workers)
    print("所有文件复制完成。")

