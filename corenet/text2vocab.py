import json
import pickle
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
from threading import Lock
from corenet.data.datasets.classification.generate_vocab import get_vocab

def load_jsonl(file_path):
    objects = []
    with open(file_path, 'r') as file:
        for line in file:
            objects.append(json.loads(line.strip()))
    return objects

def process_text(text):
    return get_vocab(text, defaultdict(int))

def update_vocab_dict(vocab_partial, vocab_dict, lock):
    with lock:
        for word, count in vocab_partial.items():
            vocab_dict[word] += count

if __name__ == '__main__':
    output_file = '/ML-A100/team/mm/models/recipe1M+_1/image2texts_new.jsonl'
    file_path = 'corenet/data/datasets/classification/recipe1M+_new_vocab.pkl'
    
    lines_generator = load_jsonl(output_file)
    print(len(lines_generator))
    vocab_dict = defaultdict(int)
    lock = Lock()
    num_threads = 64  # 设置线程数量
    batch_size = 100000  # 设置批处理大小
    texts_batch = []

    def process_and_update(text):
        try:
            vocab_partial = process_text(text)
            update_vocab_dict(vocab_partial, vocab_dict, lock)
        except Exception as e:
            print(f"Error processing text: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_text = {}
        for line in tqdm(lines_generator, total=len(lines_generator)):  # 总条目数
            texts_batch.append(line["texts"])
            if len(texts_batch) >= batch_size:
                for text in texts_batch:
                    future = executor.submit(process_and_update, text)
                    future_to_text[future] = text
                texts_batch = []
                # 清理完成的任务以释放内存
                for future in concurrent.futures.as_completed(future_to_text):
                    future_to_text.pop(future)

        # 处理最后一批
        if texts_batch:
            for text in texts_batch:
                future = executor.submit(process_and_update, text)
                future_to_text[future] = text
            texts_batch = []
            for future in concurrent.futures.as_completed(future_to_text):
                future_to_text.pop(future)

    vocab_dict_sorted = dict(sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True))
    print(len(vocab_dict_sorted))

    with open(file_path, 'wb') as file:
        pickle.dump(vocab_dict_sorted, file)
    
    print(vocab_dict_sorted)
