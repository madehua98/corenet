import os
import shutil
import multiprocessing
from tqdm import tqdm

# 定义原始目录和目标目录
source_directory = '/ML-A100/team/mm/models/food172/food_172/images'
target_directory = '/ML-A100/team/mm/models/food172/food_172/test_images'

# 确保目标目录存在
os.makedirs(target_directory, exist_ok=True)

# 读取txt文件路径
txt_file = '/ML-A100/team/mm/models/food172/food_172/recognition/test_full.txt'

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

# if __name__ == '__main__':
#     files = get_files(txt_file)
#     tasks = get_tasks(files, source_directory, target_directory)
#     num_workers = multiprocessing.cpu_count()
#     process_tasks_in_parallel(tasks, num_workers)
#     print("所有文件复制完成。")

import os
import os

def count_files_in_subdirectories(directory):
    # 检查指定的目录是否存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在。")
        return
    
    # 创建一个字典来存储子目录及其文件数量
    subdirectory_file_count = {}
    
    # 遍历指定目录下的所有子目录
    for root, dirs, files in os.walk(directory):
        # 如果当前目录是子目录（即root不等于directory）
        if root != directory:
            # 获取子目录名称
            subdirectory_name = os.path.relpath(root, directory)
            # 获取子目录中的文件数量
            file_count = len(files)
            # 将子目录名称和文件数量存储在字典中
            subdirectory_file_count[subdirectory_name] = file_count
    
    return subdirectory_file_count

# 指定要检查的目录，例如 "目录a"
directory_to_check = "/ML-A100/team/mm/models/food101/food101/test_images"

# 获取所有子目录及其文件数量
file_counts = count_files_in_subdirectories(directory_to_check)

# 打印结果
if file_counts:
    for subdirectory, count in file_counts.items():
        print(f"Subdirectory: {subdirectory}, File Count: {count}")


