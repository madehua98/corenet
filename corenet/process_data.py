import os
import pickle
import multiprocessing
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from corenet.data.datasets.classification.generate_vocab import get_vocab

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

def get_origin_data(filename, threshold):
    objects = load_jsonl(filename)
    new_objects = []
    for object in tqdm(objects, total=len(objects)):
        for k, v in object.items():
            if v > threshold:
                new_objects.append(k)
    return new_objects

def get_text(images):
    texts = []
    for image in images:
        text = image.replace('jpg', 'txt')
        texts.append(text)
    return texts

def recipe_get_data(filename):
    images = []
    texts = []
    with open(filename, mode='r') as file:
        lines = file.readlines()
    for data in tqdm(lines, total=len(lines)):
        data = json.loads(data)
        images.append(data["image"])
        texts.append(data["ingredients"])
    return images, texts
    

def write_image_text_pickle(task):
    image_path, text_path, pickle_name = task
    
    # 检查pickle文件是否存在，若存在则跳过
    if os.path.exists(pickle_name):
        return
    
    os.makedirs(os.path.dirname(pickle_name), exist_ok=True)
    try:
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()
        with open(text_path, 'r') as txt_file:
            text_data = txt_file.read()
        #text_data =   text_path
        # 创建pickle文件内容
        data = {
            'image': image_data,
            'text': text_data
        }
        # 写入pickle文件
        with open(pickle_name, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
    except Exception as e:
        print(f"Error processing {task}: {e}")

def get_tasks(images, texts, root_dir):
    tasks = []
    folder_number = 0
    pickle_number = 0
    for i in tqdm(range(len(images)), total=len(images)):
        image = images[i]
        text = texts[i]
        pickle_name = root_dir + f'/{folder_number}' + f'/{pickle_number}.pkl'
        pickle_number += 1
        folder_number = int(pickle_number / 10000)
        task = (image, text, pickle_name)
        tasks.append(task)
    return tasks

def process_tasks_in_parallel(tasks, num_workers):
    with multiprocessing.Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(write_image_text_pickle, tasks), total=len(tasks)):
            pass

# images = get_origin_data('/ML-A100/team/mm/models/cc12m/threshold_record.jsonl', 5)
# texts = get_text(images=images)
# tasks = get_tasks(
#     images=images,
#     texts=texts,
#     root_dir='/ML-A100/team/mm/models/catlip_data/cc12m'
# )

# # 设置工作进程数量
# num_workers = 64

# # 多线程处理任务并显示进度条
# process_tasks_in_parallel(tasks, num_workers)

#filename = '/ML-A100/team/mm/models/recipe1M+/image2ingredients.jsonl'

# with open(filename, mode='r') as file:
#     lines_new = []
#     lines = file.readlines()
#     for line in tqdm(lines, total=len(lines)):
#         line_new = {}
#         line = json.loads(line)
#         line_new = line
#         line_new['image'] = line_new['image'].replace('/media/fast_data/recipe1M+/image/', '/ML-A100/team/mm/models/recipe1M+/')
#         lines_new.append(line_new)

# filename_new = '/ML-A100/team/mm/models/recipe1M+/image2ingredients_new.jsonl'
# with open(filename_new, mode='a+') as fw:
#     for line in tqdm(lines_new, total=len(lines_new)):
#         line = json.dumps(line)
#         fw.write(line)
#         fw.write('\n')

# filename = '/ML-A100/team/mm/models/recipe1M+/image2ingredients_new.jsonl'
# images, texts = recipe_get_data(filename)
# tasks = get_tasks(
#     images=images,
#     texts=texts,
#     root_dir='/ML-A100/team/mm/models/catlip_data/cc12m'
# )

# # 设置工作进程数量
# num_workers = 64

# # 多线程处理任务并显示进度条
# process_tasks_in_parallel(tasks, num_workers)


"""
根据文本生成词表
"""
# images = get_origin_data('/ML-A100/team/mm/models/laion2b/threshold_record.jsonl', 5)
# #text_paths = get_text(images=images)
# print(len(images))
# images = get_origin_data('/ML-A100/team/mm/models/laion2b/threshold_record.jsonl', 1)
# #text_paths = get_text(images=images)
# print(len(images))
# images = get_origin_data('/ML-A100/team/mm/models/laion2b/threshold_record.jsonl', 8)
# #text_paths = get_text(images=images)
# print(len(images))
# images = get_origin_data('/ML-A100/team/mm/models/cc12m/threshold_record.jsonl', 5)
# text_paths = get_text(images=images)
texts = []
def read_text_file(text_path):
    try:
        with open(text_path, 'r') as txt_file:
            text_data = txt_file.read()
            text_dict = {
                "path": text_path,
                "captions": text_data
            }
            return text_dict
    except FileNotFoundError:
        print(f"File not found: {text_path}")
    except IOError:
        print(f"Error reading file: {text_path}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {text_path}: {e}")
        return None

def save_to_jsonl(filename, data):
    with open(filename, 'w') as jsonl_file:
        for entry in data:
            jsonl_file.write(json.dumps(entry) + '\n')

def process_files(text_paths, output_file, num_threads=multiprocessing.cpu_count()):
    texts = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(read_text_file, path): path for path in tqdm(text_paths, total=len(text_paths))}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                texts.append(result)
    save_to_jsonl(output_file, texts)
output_file = '/ML-A100/team/mm/models/catlip_data/cc12m/captions.jsonl'
# process_files(text_paths, output_file)


# output_file = '/ML-A100/team/mm/models/recipe1M+/image2ingredients_new.jsonl'
# vocab_dict = {}
# lines = load_jsonl(output_file)
# texts = []
# for line in lines:
#     text = line["ingredients"]
#     texts.append(text)
# for text in tqdm(texts, total=len(texts)):
#     vocab_dict = get_vocab(text, vocab_dict)

# vocab_dict_sorted = dict(sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True))
# print(len(vocab_dict_sorted))

# file_path = 'corenet/data/datasets/classification/recipe1M+_vocab.pkl'
# with open(file_path, 'wb') as file:
#     pickle.dump(vocab_dict_sorted, file)
# print(vocab_dict_sorted)


# output_file = '/ML-A100/team/mm/models/catlip_data/cc12m/captions.jsonl'
# vocab_dict = {}
# lines = load_jsonl(output_file)
# texts = []
# for line in lines:
#     text = line["captions"]
#     texts.append(text)
# for text in tqdm(texts, total=len(texts)):
#     vocab_dict = get_vocab(text, vocab_dict)

# vocab_dict_sorted = dict(sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True))
# print(len(vocab_dict_sorted))

# file_path = 'corenet/data/datasets/classification/cc12m_vocab.pkl'
# with open(file_path, 'wb') as file:
#     pickle.dump(vocab_dict_sorted, file)

"""
输出打印词表中词的含义

"""
# import pickle

# file_path = 'corenet/data/datasets/classification/datacomp_vocab.pkl'
# with open(file_path, 'rb') as file:
#     vocab_dict_sorted = pickle.load(file)

# # 将字典转换为键值对的列表
# vocab_items = list(vocab_dict_sorted.items())

# from nltk.corpus import wordnet as wn
# def convert_string_to_synset(synset_str: str):
#     # 提取词性字符和偏移值
#     pos = synset_str[0]
#     offset = int(synset_str[1:])
#     return wn.synset_from_pos_and_offset(pos, offset)

# # 打印前100个键值对
# for i in range(100):
#     k ,v = vocab_items[i]
#     k = convert_string_to_synset(k)
#     original_word = k.lemmas()[0].name()
#     print(original_word)


"""
写入的pickle文件中有缺失的文件，使用最后一个目录内的文件对其进行补充。
"""

import os
import shutil

def format_number(number):
    return f"{number:04d}"

def check_files(directory, backup_directory):
    # 获取目录中所有文件的名称（包括路径）
    try:
        existing_files = set(os.listdir(directory))
    except:
        return 1
    #existing_files = set(os.listdir(directory))
    directory_number = int(directory.split('/')[-1])
    # 生成所有应存在的文件名称
    if directory_number != 0:
        expected_files = {f"{directory_number}{format_number(i)}.pkl" for i in range(10000)}
    else:
        expected_files = {f"{i}.pkl" for i in range(10000)}
    # 找出缺失的文件
    missing_files = expected_files - existing_files
    
    # 输出缺失文件的完整路径
    if missing_files:
        print(1)
        print(f"缺失文件的目录为{directory_number}")
        print(f"缺失的文件数量: {len(missing_files)}")
        
        # 从备份目录中寻找文件进行替补
        backup_files = os.listdir(backup_directory)
        backup_index = 0
        
        for missing_file in sorted(missing_files):
            if backup_index < len(backup_files):
                backup_file = backup_files[backup_index]
                source_path = os.path.join(backup_directory, backup_file)
                target_path = os.path.join(directory, missing_file)
                
                if os.path.isfile(source_path):
                    shutil.copyfile(source_path, target_path)
                    print(f"已从{source_path}复制到{target_path}")
                    
                backup_index += 1
            else:
                print("备份目录中的文件不足以替补所有缺失的文件。")
                break

# 示例使用
# directorys = "/ML-A100/team/mm/models/catlip_data/laion2b/"
# backup_directory = os.path.join(directorys, '2116')

# for directory in os.listdir(directorys):
#     if directory == '2116':
#         continue
#     check_files(directorys + directory, backup_directory)





"""
阅读pickle文件。

"""
# import pickle

# metadata = {
#     "total_tar_files": 2905,
#     "max_files_per_tar": 10000,
#     "tar_file_names": [f"{i}.tar.gz" for i in range(2906)]
# }
# # # 使用二进制模式打开文件
# with open('/ML-A100/team/mm/models/catlip_data/cache/metadata.pkl', mode='wb') as file:
#     pickle.dump(metadata, file)


# # 使用二进制模式打开文件
# with open('/ML-A100/team/mm/models/catlip_data/cache/108/1086988.pkl', mode='rb') as file:
#     metadata = pickle.load(file)
#     print(metadata)


"""
合并词表
"""

import pickle

# 读取所有词表
with open('./corenet/data/datasets/classification/laion2b_vocab.pkl', mode='rb') as file:
    laion2b_vocab = pickle.load(file)
with open('./corenet/data/datasets/classification/datacomp_vocab.pkl', mode='rb') as file:
    datacomp_vocab = pickle.load(file)
with open('./corenet/data/datasets/classification/cc12m_vocab.pkl', mode='rb') as file:
    cc12m_vocab = pickle.load(file)
with open('./corenet/data/datasets/classification/recipe1M+_vocab.pkl', mode='rb') as file:
    recipe1M_vocab = pickle.load(file)

# 合并词表
final_vocab = {}

def merge_vocab(final_vocab, vocab):
    for key, value in vocab.items():
        if key in final_vocab:
            final_vocab[key] += value
        else:
            final_vocab[key] = value

merge_vocab(final_vocab, laion2b_vocab)
merge_vocab(final_vocab, datacomp_vocab)
merge_vocab(final_vocab, cc12m_vocab)
merge_vocab(final_vocab, recipe1M_vocab)

# 将合并后的词表写入文件
with open('./corenet/data/datasets/classification/all_vocab.pkl', mode='wb') as file:
    pickle.dump(final_vocab, file)


