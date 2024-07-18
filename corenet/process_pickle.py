import os
import pickle

def process_pickle_files(directory):
    # 获取目录下所有的文件
    files = os.listdir(directory)
    
    for file in files:
        # 构建完整的文件路径
        file_path = os.path.join(directory, file)
        
        # 检查文件是否是pickle文件
        if file_path.endswith('.pkl') or file_path.endswith('.pickle'):
            try:
                # 尝试打开并加载pickle文件
                with open(file_path, 'rb') as f:
                    pickle.load(f)
                print(f"{file} 加载成功")
            except (pickle.UnpicklingError, EOFError, FileNotFoundError, IOError):
                # 如果加载失败，则删除文件
                print(f"{file} 加载失败，删除文件")
                os.remove(file_path)

# # 指定目录路径
# directory = '/ML-A100/team/mm/models/catlip_data/cache/108'

# # 调用函数处理pickle文件
# process_pickle_files(directory)

import pickle
import matplotlib.pyplot as plt

def load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def plot_histogram(data, thresholds, save_path):
    threshold_counts = {threshold: sum(1 for freq in data.values() if freq > threshold) for threshold in thresholds}
    
    # Sorting thresholds and their counts for plotting
    sorted_thresholds = sorted(threshold_counts.keys())
    counts = [threshold_counts[threshold] for threshold in sorted_thresholds]
    print(counts)
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(sorted_thresholds)), counts, color='blue', width=0.4)  # Adjust the width of the bars
    plt.xlabel('Threshold')
    plt.ylabel('Number of Words')
    plt.title('Number of Words Above Each Threshold')
    plt.xticks(range(len(sorted_thresholds)), sorted_thresholds, rotation=45)  # Ensure the x-axis ticks are equally spaced and rotated
    
    # Save the plot to the specified path
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    filename = 'corenet/data/datasets/classification/all_vocab.pkl'
    datacomp_vocab = load_pickle(filename)

    thresholds = [10, 50, 100, 200, 500, 700, 1000]  # 设定多个阈值
    save_path = '/ML-A800/home/guoshuyue/madehua/code/corenet/word_frequencies_histogram.png'  # 保存图像的路径
    
    plot_histogram(datacomp_vocab, thresholds, save_path)

# filename = 'corenet/data/datasets/classification/datacomp_1_2B_vocab.pkl'
# datacomp_1b = load_pickle(filename)
# datacomp_1b = list(datacomp_1b.items())
# print(datacomp_1b[5000])


