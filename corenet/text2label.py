#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

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
    nltk_data_path='/home/data_llm/anaconda3/envs/corenet/nltk_data'
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
    with open(file_path, 'r') as file:
        lines = [json.loads(line.strip()) for line in file]
    return lines

def save_jsonl(file_path, data):
    with open(file_path, 'a') as file:
        file.write(json.dumps(data) + '\n')

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
            except Exception as e:
                # No lemma 'is' with part of speech 'n', then nltk.corpus.reader.wordnet.WordNetError is raised.
                # Skip such cases
                continue

            if not check_valid_noun_synset(noun_synset, word):
                continue
            noun_synset = extract_pos_offset_info_from_synset(noun_synset)
            labels.append(noun_synset)
        return labels

def process_line(line):
    line_new = {}
    id = line["id"]
    img_path = line["image"]
    text = line["texts"]
    labels = convert_caption_to_labels(text)
    line_new["id"] = id
    line_new["image"] = img_path
    line_new["labels"] = labels
    return line_new

def process_lines_in_parallel(lines, output_file, num_workers):
    with multiprocessing.Pool(num_workers) as pool:
        for result in tqdm(pool.imap(process_line, lines), total=len(lines)):
            save_jsonl(output_file, result)

if __name__ == '__main__':
    input_file = '/ML-A100/team/mm/models/recipe1M+_1/image2texts_new.jsonl'
    output_file = '/ML-A100/team/mm/models/recipe1M+_1/image2labels_new.jsonl'
    lines = load_jsonl(input_file)
    num_workers = multiprocessing.cpu_count()  # Or specify the number of workers you want to use
    process_lines_in_parallel(lines, output_file, num_workers)
