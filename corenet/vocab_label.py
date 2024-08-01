import argparse
import json
import os
import pickle
import multiprocessing
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from corenet.data.datasets.utils.text import caption_preprocessing
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.data import find

nltk_data_path = '/home/data_llm/anaconda3/envs/corenet/nltk_data'
nltk.data.path.append(nltk_data_path)

# 检查并下载常用数据
packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']
for package in packages:
    try:
        find(f'tokenizers/{package}')
    except LookupError:
        nltk.download(package, download_dir=nltk_data_path)

def load_jsonl(file_path: str) -> List[dict]:
    with open(file_path, 'r') as file:
        lines = [json.loads(line.strip()) for line in file]
    return lines

def save_jsonl(file_path: str, data: dict):
    with open(file_path, 'a') as file:
        file.write(json.dumps(data) + '\n')

def check_valid_noun_synset(synset, word: str) -> bool:
    return synset.name() == f"{word}.n.01"

def extract_pos_offset_info_from_synset(synset) -> str:
    offset = synset.offset()
    pos = synset.pos()
    return f"{pos}{offset}"

def convert_caption_to_labels_and_update_vocab(captions_str: str, vocab_dict: dict) -> List[str]:
    lemmatizer = WordNetLemmatizer()
    captions_str = caption_preprocessing(captions_str)
    tagged_words = nltk.pos_tag(nltk.word_tokenize(captions_str))
    labels = []

    for word, pos in tagged_words:
        try:
            word = lemmatizer.lemmatize(word)
            noun_synset = wn.synset(f"{word}.n.01")
        except:
            continue

        if not check_valid_noun_synset(noun_synset, word):
            continue
        noun_synset_str = extract_pos_offset_info_from_synset(noun_synset)
        labels.append(noun_synset_str)
        vocab_dict[noun_synset_str] += 1

    # Remove duplicate labels
    labels = list(set(labels))

    return labels

def process_line(line, vocab_dict: dict, lock: Lock) -> dict:
    id = line["id"]
    img_path = line["image"]
    text = line["texts"]

    with lock:
        labels = convert_caption_to_labels_and_update_vocab(text, vocab_dict)

    return {"id": id, "image": img_path, "labels": labels}

def process_lines_in_parallel(lines: List[dict], output_file: str, vocab_dict: dict, num_workers: int, batch_size: int = 10000):
    lock = Lock()
    total_lines = len(lines)
    total_batches = total_lines // batch_size + (total_lines % batch_size > 0)

    with tqdm(total=total_batches, desc="Processing Batches") as pbar:
        for i in range(0, total_lines, batch_size):
            batch_lines = lines[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_line, line, vocab_dict, lock) for line in batch_lines]
                for future in as_completed(futures):
                    line_new = future.result()
                    save_jsonl(output_file, line_new)
            pbar.update(1)

if __name__ == '__main__':
    input_file = '/ML-A100/team/mm/models/recipe1M+_1/image2texts_new.jsonl'
    output_file = '/ML-A100/team/mm/models/recipe1M+_1/image2labels_new.jsonl'
    vocab_file = 'corenet/data/datasets/classification/recipe1M+_new_vocab.pkl'

    lines = load_jsonl(input_file)
    vocab_dict = defaultdict(int)
    num_workers = multiprocessing.cpu_count()

    process_lines_in_parallel(lines, output_file, vocab_dict, num_workers)

    vocab_dict_sorted = dict(sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True))
    with open(vocab_file, 'wb') as file:
        pickle.dump(vocab_dict_sorted, file)

    print("Processing complete. Vocabulary and labels have been saved.")
