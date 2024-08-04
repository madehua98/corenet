from torchvision.models import resnet101
from thop import profile
import torch
import timm

# 加载 ViT base 模型
# model1 = timm.create_model('vit_base_patch16_224', pretrained=True)
model = resnet101()
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))
from thop import clever_format
macs, params = clever_format([macs, params], "%.3f")
print(macs)
print(params)