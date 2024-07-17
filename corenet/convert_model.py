# import torch
# from transformers import CLIPModel

# # 加载 CLIP 模型
# clip_model = CLIPModel.from_pretrained('/ML-A100/team/mm/models/vit-base')

# # 提取 ViT 模型部分
# vit_model = clip_model.vision_model

# # 保存 ViT 模型为 .pt 文件
# torch.save(vit_model.state_dict(), '/ML-A100/team/mm/models/vit-base/vit_model.pt')

# print("ViT model has been extracted from CLIP and saved to vit_model.pt")
import json
from tqdm import tqdm
import numpy as np
import os

def save_json(filename, object):
    with open(filename, mode='w') as file:
        json.dump(object, file, indent=4)

def get_sorted_subdirectories(directory):
    # 获取所有子目录
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    # 对子目录进行排序
    subdirectories.sort()
    return subdirectories

def create_directory_dict(subdirectories):
    directory_dict = {}
    for i, subdirectory in enumerate(subdirectories):
        # 键为0-100
        directory_dict[i] = subdirectory
    return directory_dict

# 指定目录路径
path = "/ML-A100/team/mm/models/food101/food101/images"

# 获取排序后的子目录列表
sorted_subdirectories = get_sorted_subdirectories(path)

# 创建字典
directory_dict = create_directory_dict(sorted_subdirectories)
print(directory_dict)

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
file_path = '/ML-A100/team/mm/models/catlip_data/single_small_500_dci/output.jsonl'
matric = np.zeros((101,101))
objects = load_jsonl(file_path)
count = {}
for object in objects:
    if object["targets"] in count:
        count[object["targets"]] += 1
    else:
        count[object["targets"]] = 1

    preb_label = object["pred_label"]
    targets = object["targets"]
    matric[targets,preb_label] += 1 
count = dict(sorted(count.items(), key=lambda item: item[1], reverse=True))
count_new = {}
for k,v in count.items():
    k_new = directory_dict[k]
    count_new[k_new] = v
filename = "/ML-A100/team/mm/models/catlip_data/single_small_500_dci/error_count.json"
save_json(count_new, filename)

indices = np.argwhere(matric>=5)
# 同时获取对应的值
values_with_indices = [(index, matric[tuple(index)]) for index in indices]
sorted_values_with_indices = sorted(values_with_indices, key=lambda x: x[1])

print("大于阈值的元素的索引和对应的值：")
with open('/ML-A100/team/mm/models/catlip_data/single_small_500_dci/sorted_values.txt', 'w') as file:
    for index, value in sorted_values_with_indices:
        index = list(index)
        index = [directory_dict[index[0]], directory_dict[index[1]]]
        file.write(f"索引: {index}, 值: {value}")

