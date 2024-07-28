import torch
from transformers import CLIPModel
from transformers import AutoModel, CLIPImageProcessor

# # 加载 CLIP 模型
# clip_model = CLIPModel.from_pretrained('/ML-A100/team/mm/models/vit-base')


# # 提取 ViT 模型部分
# vit_model = clip_model.vision_model

# print(vit_model)



import torch
from transformers import CLIPModel
from transformers import AutoModel, CLIPImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(
    '/ML-A100/team/mm/models/ViTamin-B',
    trust_remote_code=True).to(device).eval()

# print(model)
vitamin_model = model.visual.trunk
#image_processor = CLIPImageProcessor.from_pretrained('/ML-A100/team/mm/models/ViTamin-B')

print(vitamin_model.state_dict().keys())
# print(vit_model.state_dict().keys())

# vit_base = torch.load('/ML-A100/team/mm/models/vit_base.pt')
# print(vit_base.keys())

# 保存 ViT 模型为 .pt 文件
# torch.save(vit_model.state_dict(), '/ML-A100/team/mm/models/vit-base/vit_model.pt')

# print("ViT model has been extracted from CLIP and saved to vit_model.pt")

