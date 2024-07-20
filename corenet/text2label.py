import argparse
import fcntl
import glob
import io
import os
import pickle
import random
import shutil
import tarfile
from pathlib import Path
from typing import Any, List, Mapping, Tuple
from urllib.parse import urlsplit

import pybase64
import torch
from PIL import Image, ImageFile

try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.corpus.reader.wordnet import Synset
    from nltk.stem import WordNetLemmatizer
    nltk_data_path = '/home/data_llm/anaconda3/envs/corenet/nltk_data'
    nltk.data.path.append(nltk_data_path)

    # 下载常用数据包
    packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']

    # 检查并下载缺失的数据包
    # for package in packages:
    #     if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers', package)) and \
    #     not os.path.exists(os.path.join(nltk_data_path, 'corpora', package)):
    #         try:
    #             nltk.download(package, download_dir=nltk_data_path)
    #             print(f"Successfully downloaded {package}")
    #         except Exception as e:
    #             print(f"Failed to download {package}: {e}")

    NLTK_INSTALLED = True
except ModuleNotFoundError:
    wn = None
    Synset = None
    WordNetLemmatizer = None

    NLTK_INSTALLED = False


from corenet.constants import DATA_CACHE_DIR, LAION_CACHE_DIR, RECIPE_CACHE_DIR, CC12M_CACHE_DIR, DATACOMP_COUNT, LAION_COUNT, RECIPE_COUNT, CC12M_COUNT
from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.classification.base_image_classification_dataset import (
    BaseImageClassificationDataset,
)
from corenet.data.datasets.dataset_base import BaseImageDataset
from corenet.data.datasets.utils.text import caption_preprocessing
from corenet.data.io.transfer_clients import BaseClient, get_transfer_client
from corenet.data.transforms import BaseTransformation
from corenet.utils import logger
from corenet.utils.download_utils import get_local_path
import json
import multiprocessing
from tqdm import tqdm

def load_jsonl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        lines = [json.loads(line.strip()) for line in file]
    return lines

def save_jsonl(file_path, data):
    try:
        with open(file_path, 'a+') as file:
            file.write(json.dumps(data) + '\n')
    except Exception as e:
        print(f"Error saving to file: {e}")

def check_valid_noun_synset(synset: Synset, word: str) -> bool:
    """Check if input synset and word are the same.

    Args:
        synset: Input synset.
        word: Input word.

    Returns:
        A boolen indicating if input synset and word are the same or not.
    """
    return synset.name() == f"{word}.n.01"

def extract_pos_offset_info_from_synset(synset: Synset) -> str:
    """Extracts part-of-speech and offset information from the input @synset.

    Args:
        synset: WordNet synset.

    Returns:
        A string containing part-of-speech and offset information about the synset.
    """
    offset = synset.offset()
    pos = synset.pos()
    return f"{pos}{offset}"

def convert_caption_to_labels(captions_str: str) -> List[int]:
    """Converts the caption into multi-class labels.

    The input caption is tokenized into words, and noun synsets are extracted for each word. Subsequently, the
    parts of speech (POS) and offsets of the extracted noun synsets are compared with those in the vocabulary
    to generate a list of multi-class labels.

    Args:
        captions_str: Input caption as a string.
    Returns:
        A list of integers, where each integer corresponds to the index of the matching synset in the vocabulary.
        In case there are no matching synsets, an empty list is returned.
    """
    captions_str = caption_preprocessing(captions_str)  # 清洗caption的格式
    # process caption and find synsets

    tagged_words = nltk.pos_tag(nltk.word_tokenize(captions_str))  # [('flavored', 'VBN'), ('snail', 'NN'), ('meat', 'NN')]
    lemmatzr = WordNetLemmatizer()
    labels = []  # 存储的是名词label的索引
    for word, pos in tagged_words:
        # use lemmatizer to reduce text ambiguity.
        # words like bicycle and bicycles are converted to bicycle
        try:
            word = lemmatzr.lemmatize(word)
            noun_synset = wn.synset(f"{word}.n.01")
        except nltk.corpus.reader.wordnet.WordNetError:
            # No lemma 'is' with part of speech 'n'
            continue

        if not check_valid_noun_synset(noun_synset, word):
            continue
        noun_synset = extract_pos_offset_info_from_synset(noun_synset)
        labels.append(noun_synset)
    labels = list(set(labels))
    return labels

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

def process_line(line):
    try:
        line_new = {}
        img_path = line["image"]
        text = line["texts"]
        labels = convert_caption_to_labels(text)
        line_new["image"] = img_path
        line_new["labels"] = labels
        return line_new
    except Exception as e:
        print(f"Error processing line: {e}")
        return None

def process_lines_in_parallel(lines, output_file, num_workers):
    with multiprocessing.Pool(num_workers) as pool:
        for result in tqdm(pool.imap(process_line, lines), total=len(lines)):
            if result:
                save_jsonl(output_file, result)

if __name__ == '__main__':
    input_file = '/ML-A100/team/mm/models/datacomp_1b/images2text.jsonl'
    output_file = '/ML-A100/team/mm/models/datacomp_1b/image2labels_new.jsonl'
    lines = load_jsonl(input_file)
    num_workers = multiprocessing.cpu_count()  # Or specify the number of workers you want to use
    process_lines_in_parallel(lines, output_file, num_workers)

    input_file = '/ML-A100/team/mm/models/laion2b/images2text.jsonl'
    output_file = '/ML-A100/team/mm/models/laion2b/image2labels_new.jsonl'
    lines = load_jsonl(input_file)
    num_workers = multiprocessing.cpu_count()  # Or specify the number of workers you want to use
    process_lines_in_parallel(lines, output_file, num_workers)

    input_file = '/ML-A100/team/mm/models/cc12m/images2text.jsonl'
    output_file = '/ML-A100/team/mm/models/cc12m/image2labels_new.jsonl'
    lines = load_jsonl(input_file)
    num_workers = multiprocessing.cpu_count()  # Or specify the number of workers you want to use
    process_lines_in_parallel(lines, output_file, num_workers)




"""
将datacom_1b、laion2b和cc12m的图像和文本转为jsonl的格式

"""
def process_line(text_file):
    try:
        with open(text_file, 'r') as file:
            text_content = file.read().strip()
        return {"image": text_file.replace('txt', 'jpg'), "texts": text_content}
    except:
        return None


def main(input_path, output_path):
    images = get_origin_data(input_path, 5)
    texts = get_text(images=images)

    num_workers = multiprocessing.cpu_count()  # 可以根据需要调整进程数
    with multiprocessing.Pool(num_workers) as pool:
        results = []
        for result in tqdm(pool.imap(process_line, texts), total=len(texts)):
            if result != None:
                save_jsonl(output_path, result)

# if __name__ == '__main__':
#     input_path = '/ML-A100/team/mm/models/cc12m/threshold_record.jsonl'
#     output_path = '/ML-A100/team/mm/models/cc12m/images2text.jsonl'
#     main(input_path, output_path)
