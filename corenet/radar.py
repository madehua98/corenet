import matplotlib.pyplot as plt
import numpy as np

# Data definition
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

labels = ['food101(fr)', 'food172(fr)', 'food200(fr)', 'food101(ir)', 'food172(ir)', 'food200(ir)', 'FoodSeg103', 'UEC']

# Convert data to a NumPy array
data = np.array(data)

# Normalize data with a small offset to avoid 0 values
normalized_data = np.zeros_like(data)
for i in range(len(labels)):
    column = data[:, i]
    min_val = np.min(column)
    max_val = np.max(column)
    normalized_data[:, i] = (column - min_val) / (max_val - min_val) * 50 + 50  # Scale to [10, 100]

# Add the first column to the end to close the radar chart
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
normalized_data = np.concatenate((normalized_data, normalized_data[:, [0]]), axis=1)
angles += angles[:1]

# Create the radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Line colors and styles
colors = ['#00aaff', '#aa00ff', '#ff7f0e', '#ff0000'] 
line_styles = ['-.', '--', ':', '-']  # 保留原有的线型样式

# Plot the lines
for idx, (d, color, style) in enumerate(zip(normalized_data, colors, line_styles)):
    ax.plot(angles, d, linestyle=style, linewidth=2, label=models[idx], color=color)
    ax.fill(angles, d, color=color, alpha=0.25)

# Set the labels
ax.set_xticks(angles[:-1])

# Calculate the rotation angle for each label
label_angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
label_rotations = [angle * 180 / np.pi - 90 for angle in label_angles]
label_rotations = [-0.0, -45.0, -90.0, 45.0, 0.0, -45.0, 90.0, 45.0]
# Set the labels with different rotation angles
for i, label in enumerate(labels):
    ax.text(angles[i], 105, label, ha='center', va='center', rotation=label_rotations[i], size=13, weight='bold')


# Remove intermediate grid labels
ax.set_xticklabels([""] * len(labels))  # 隐藏原始 x 轴标签
ax.set_yticklabels([])

# Set the range
ax.set_ylim(0, 100)

# Grid
ax.grid(True)

# Legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=13)

# Save the figure
plt.savefig("radar.png", bbox_inches='tight')
plt.close(fig)