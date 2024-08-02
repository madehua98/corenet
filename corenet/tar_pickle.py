import matplotlib.pyplot as plt
import numpy as np

# 数据定义
models = [
    "CLIP ViT B/16",
    "CatLIP ViT B/16",
    "FoodCLIP-S",
    "FoodCLIP-B"
]

data = [
    [92.1, 91.14, 71.12, 97.79, 88.75, 79.7, 28.7, 63.27],  # CLIP ViT B/16
    [93.2, 92.17, 71.8, 98.27, 89.58, 80.96, 35.78, 69.57],  # CatLIP ViT B/16
    [94.64, 92.29, 73.79, 98.5, 88.99, 82.97, 37.27, 70.25],  # FoodCLIP-S
    [96.26, 94.18, 77.0, 99.28, 90.91, 85.71, 42.22, 73.3]   # FoodCLIP-B
]

labels = ['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4', 'Dataset 5', 'Dataset 6', 'Dataset 7', 'Dataset 8']

# 角度
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
data = np.asarray(data)
data = np.concatenate((data, data[:,[0]]), axis=1)
angles += angles[:1]

# 找到最小和最大限制
max_value = np.max(data)
max_limit = int(np.ceil(max_value / 10.0)) * 10
min_limit = max_limit / 2

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 设置雷达图的范围
ax.set_ylim(min_limit, max_limit)

# 线颜色和样式
colors = ['blue', 'green', 'red', 'cyan']
line_styles = ['-', '--', '-.', ':']

# 绘制折线图
for idx, (d, color, style) in enumerate(zip(data, colors, line_styles)):
    ax.plot(angles, d, linestyle=style, linewidth=2, label=models[idx], color=color)
    ax.fill(angles, d, color=color, alpha=0.25)

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, size=13, weight='bold')

# 网格
ax.grid(True)

# 图例
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# 保存图形
plt.savefig("adjusted_figure.png", bbox_inches='tight')