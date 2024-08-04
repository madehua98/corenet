import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
import itertools
from nltk.corpus import wordnet as wn

# 确保你已经下载了 wordnet 数据
import nltk
nltk.download('wordnet')

with open('corenet/data/datasets/classification/all_new_vocab.pkl', 'rb') as file:
    data = pickle.load(file)

# 获取前 1000 个数据
data = dict(itertools.islice(data.items(), 1000))

# 将 Synset 对象转换为字符串表示
data_new = {}
for k, v in data.items():
    pos = k[0]
    offset = int(k[1:])
    noun_synset = wn.synset_from_pos_and_offset(pos, offset)
    data_new[noun_synset.lemmas()[0].name()] = v

# 创建词云对象
wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(data_new)

# 绘制词云图
plt.figure(figsize=(8, 8))  # 调整显示比例为正方形
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # 关闭坐标轴

# 保存高分辨率图像
plt.savefig('/ML-A800/home/guoshuyue/madehua/code/corenet/word_cloud.png', format='png', bbox_inches='tight', pad_inches=0, dpi=300)

# 显示词云图
plt.show()
