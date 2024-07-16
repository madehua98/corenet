import os
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import json

def count_parquet_rows(directory):
    total_rows = 0
    parquet_files = []

    # 遍历目录下的所有文件，收集Parquet文件路径
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                parquet_files.append(file_path)

    # 使用进度条遍历Parquet文件
    for file_path in tqdm(parquet_files, desc="Counting rows in Parquet files"):
        # 使用PyArrow读取Parquet文件
        parquet_file = pq.ParquetFile(file_path)
        
        # 计算文件的行数并累加
        total_rows += parquet_file.metadata.num_rows

    return total_rows

# # 指定目录路径
# directory_path = '/ML-A100/team/mm/models/datacomp_1b/basic_filter_metadata'

# # 获取总行数
# total_rows = count_parquet_rows(directory_path)
# print(f'Total rows in all Parquet files: {total_rows}')

def load_jsonl(filename):
    objects = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line_number, line in tqdm(enumerate(lines), total=len(lines)):
            try:
                line = json.loads(line)
                objects.append(line)
            except Exception as e:
                print(f"Unexpected error on line {line_number}: {e}")
    return objects

filename = '/ML-A100/team/mm/models/cc12m/threshold_record.jsonl'
objects = load_jsonl(filename)
count = 0
for object in objects:
    for k, v in object.items():
        if v >= 5:
            count += 1
print(count)