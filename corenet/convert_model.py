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
objects = load_jsonl(file_path)
count = {}
for object in objects:
    if object["targets"] in count:
        count[object["targets"]] += 1
    else:
        count[object["targets"]] = 1


count = dict(sorted(count.items(), key=lambda item: item[1], reverse=True))

print(count)

