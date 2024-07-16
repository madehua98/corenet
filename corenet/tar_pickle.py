import os
import tarfile
import multiprocessing
from tqdm import tqdm

# 获取当前目录和目标目录
current_directory = '/ML-A100/team/mm/models/catlip_data/recipe1M+_1'
output_directory = '/ML-A100/team/mm/models/catlip_data/recipe1M+_1_tar'

# 确保输出目录存在
os.makedirs(output_directory, exist_ok=True)

def get_directories():
    # 获取当前目录中的所有子目录并按名称排序
    directories = sorted([d for d in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, d))])
    return directories

def create_tar_file(task):
    index, chunk = task
    tar_filename = os.path.join(output_directory, f'recipe1M+_{index}.tar')
    
    # 检查tar文件是否存在，若存在则跳过
    if os.path.exists(tar_filename):
        return

    os.makedirs(os.path.dirname(tar_filename), exist_ok=True)
    try:
        with tarfile.open(tar_filename, 'w') as tar:
            for dir_name in chunk:
                dir_path = os.path.join(current_directory, dir_name)
                tar.add(dir_path, arcname=dir_name)
    except Exception as e:
        print(f"Error processing {tar_filename}: {e}")

def get_tasks(directories):
    # 将目录分组，每组100个
    chunks = [directories[i:i + 100] for i in range(0, len(directories), 100)]
    tasks = [(index, chunk) for index, chunk in enumerate(chunks)]
    return tasks

def process_tasks_in_parallel(tasks, num_workers):
    with multiprocessing.Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(create_tar_file, tasks), total=len(tasks)):
            pass

if __name__ == '__main__':
    directories = get_directories()
    tasks = get_tasks(directories)
    num_workers = multiprocessing.cpu_count()
    process_tasks_in_parallel(tasks, num_workers)
    print("所有目录打包完成。")
