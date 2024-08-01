from process_data import save_jsonl, load_jsonl
import tqdm
import math
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerPatch
filename = '/ML-A100/team/mm/models/catlip_data/datacomp_1b/captions.jsonl'
filename1 = '/ML-A100/team/mm/models/catlip_data/laion2b/captions.jsonl'
filename2 = '/ML-A100/team/mm/models/catlip_data/cc12m/captions.jsonl'
filename3 = '/ML-A100/team/mm/models/recipe1M+_1/image2texts_new.jsonl'
#data = load_jsonl(filename)
#data1 = load_jsonl(filename1)
# data2 = load_jsonl(filename2)
# data3 = load_jsonl(filename3)

#datas = data1

def get_text_length_statistics(datas):
    length_ranges = {
        "0-25": 0,
        "25-50": 0,
        "50-100": 0,
        "100-150": 0,
        "150+": 0
    }
    total_length = 0
    
    for idx, data in tqdm.tqdm(enumerate(datas), total=len(datas)):
        try:
            text = data["captions"]
        except:
            text = data["texts"]
        text_length = len(text)
        total_length += text_length
        
        if text_length <= 25:
            length_ranges["0-25"] += 1
        elif text_length <= 50:
            length_ranges["25-50"] += 1
        elif text_length <= 100:
            length_ranges["50-100"] += 1
        elif text_length <= 150:
            length_ranges["100-150"] += 1
        else:
            length_ranges["150+"] += 1
    
    average_length = total_length / len(datas)
    return length_ranges, average_length

# length_ranges, average_length = get_text_length_statistics(datas)
# print("Length Ranges:", length_ranges)
# print("Average Length:", average_length)


def multiply_and_sum(dicts, coefficients):
    # 验证输入的有效性
    if len(dicts) != 4 or len(coefficients) != 4:
        raise ValueError("There should be exactly four dictionaries and four coefficients.")
    
    # 初始化结果字典
    result = {}

    # 遍历每个字典和对应的系数
    for dct, coeff in zip(dicts, coefficients):
        for key, value in dct.items():
            # 如果键不存在于结果字典中，初始化为0
            if key not in result:
                result[key] = 0
            # 加上当前字典此键对应的值乘以系数
            result[key] += value * coeff
    for key in result:
        result[key] = math.floor(result[key]) 

    return result

# 定义长度范围字典和系数
length_range1 = {'0-25': 4619162, '25-50': 11918885, '50-100': 8611426, '100-150': 1806882, '150+': 2109582}
length_range2 = {'0-25': 3005231, '25-50': 7717079, '50-100': 6857866, '100-150': 1683396, '150+': 1905069}
length_range3 = {'0-25': 4273, '25-50': 76105, '50-100': 131299, '100-150': 48954, '150+': 78033}
length_range4 = {'0-25': 64, '25-50': 1008, '50-100': 31597, '100-150': 138041, '150+': 13563424}

coefficients = [37/29, 51/21, 1, 1]  # 假设的系数列表，可按需要调整

# # 调用函数并打印结果
# result = multiply_and_sum([length_range1, length_range2, length_range3, length_range4], coefficients)
# print(result)



# length_ranges = {'0-25': 7628730, '25-50': 19713077, '50-100': 15632188, '100-150': 3677273, '150+': 17656108}

length_ranges = {'0-25': 13196168, '25-50': 34025443, '50-100': 27804705, '100-150': 6580574, '150+': 20959588}
# # # 计算总数
total = sum(length_ranges.values())

# 计算每个段的占比并写入新字典
percentages = {k: (v / total) * 100 for k, v in length_ranges.items()}

# 输出新字典
print(percentages)
import matplotlib.pyplot as plt
labels = percentages.keys()
sizes = percentages.values()
base_colors = [(1, 0, 1), (0, 1, 1), (0, 0, 1), (1, 0.65, 0), (1, 0.35, 0)]
gray = 0.0
saturated_colors = [(gray + (1 - gray) * r, gray + (1 - gray) * g, gray + (1 - gray) * b, 0.7) for r, g, b in base_colors]

 #自定义图例处理器
# 自定义图例处理器
class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width, 0.5 * height
        radius = 0.6 * width  # 调整这个值来改变圆点大小
        p = Circle(center, radius, transform=trans, facecolor=orig_handle.get_facecolor())
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

# 绘制饼图
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(sizes, labels=None, colors=saturated_colors, autopct='%1.1f%%', startangle=140)

# 添加标题
fig.suptitle('Text Length Distribution', fontsize=16)

# 自定义图例：圆点，不显示边框
custom_patches = [Circle((0, 0), 1, facecolor=w.get_facecolor()) for w in wedges]
ax.legend(custom_patches, labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, 
          handler_map={Circle: HandlerCircle()}, markerscale=2.0, labelspacing=3)

# 调整布局以避免图例和图表重叠
plt.subplots_adjust(left=0.1, right=0.8, top=1.0)

# 保存图表到高清的JPEG文件
#plt.savefig('figure_new.jpg', dpi=600)
plt.savefig('figure_new.svg', bbox_inches='tight')
# 显示图表
plt.show()




import matplotlib.pyplot as plt

# 定义段数据及其占比
#percentages = {'0-25': 11.862916005156237, '25-50': 30.65445711857377, '50-100': 24.308545881268735, '100-150': 5.718275614293452, '150+': 27.455805380707805}

import matplotlib.pyplot as plt

# 定义段数据及其占比
#percentages = {'0-25': 11.862916005156237, '25-50': 30.65445711857377, '50-100': 24.308545881268735, '100-150': 5.718275614293452, '150+': 27.455805380707805}

# 按照键的大小进行排序
#sorted_percentages = dict(sorted(percentages.items(), key=lambda item: item[0]))

# 标签和占比数据
# labels = list(percentages.keys())
# sizes = list(percentages.values())

# # 定义暖色调颜色
# colors = ['#FF9999', '#FFCC99', '#FFCC66', '#FF9966', '#FF6600']

# # 绘制饼图
# fig, ax = plt.subplots()
# ax.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=140)

# # 添加图例并放在右侧，增加偏移量
# ax.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5))

# # 添加图标题在整张图的正上方
# fig.suptitle('Text Length Distribution', fontsize=16)

# # 调整布局以避免图例和图表重叠
# plt.subplots_adjust(left=0.1, right=0.7, top=0.85)

# # 保存图表到高清的JPEG文件
# plt.savefig('figure_new_warm_sorted.jpg', dpi=300)

# # 显示图表
# plt.show()
