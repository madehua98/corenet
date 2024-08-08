import json
import random
import shutil
import os

import json
import random
import shutil
import os
from tqdm import tqdm

def extract_sample_images(jsonl_file_path, output_json_path, sample_size=100000, prompts=None):
    all_data = []
    result_data = []

    if prompts is None:
        prompts = ["Render a clear and concise summary of the photo"]

    with open(jsonl_file_path, "r") as file:
        for line in file:
            all_data.append(json.loads(line))

    sample_data = random.sample(all_data, sample_size)

    for data in tqdm(sample_data, total=len(sample_data)):
        # 处理和复制图像文件
        original_image_path = data['image']
        new_image_path = original_image_path.replace('extracted_shards', 'sample_extracted_shards')
        new_image_relpath = '/'.join(new_image_path.split('/')[-3:])
        
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        shutil.copy2(original_image_path, new_image_path)

        # 创建新的 JSON 对象
        conversation = [
            {"from": "human", "value": f"<image>\n{random.choice(prompts)}"},
            {"from": "gpt", "value": data['texts']}
        ]
        new_json_object = {
            "image": new_image_relpath,
            "conversations": conversation
        }
        result_data.append(new_json_object)

    # 写入 JSON 数据到文件
    with open(output_json_path, 'w') as outfile:
        json.dump(result_data, outfile, indent=4)

prompts = [
    'Render a clear and concise summary of the photo.',
    '"Write a terse but informative summary of the picture.',
    'What is this?',
    'What is in the photo?',
    'Describe the image concisely.',
    'Share a concise interpretation of the image provided.',
    'Give a brief description of the image.',
    "Present a compact description of the photo's key features.",
    'Share a concise interpretation of the image provided.',
    'Provide a brief description of the given image.',
    'What is in the photo?',
    'Summarize the visual content of the image.'
]

# JSONL 文件路径和输出 JSON 文件路径
# 指定 JSONL 文件路径
jsonl_file_path = '/ML-A100/team/mm/models/datacomp_1b/images2text.jsonl'
output_json_path = '/ML-A100/team/mm/models/datacomp_1b/datacomp_1b.json'

# 调用函数
extract_sample_images(jsonl_file_path, output_json_path, 100000, prompts)