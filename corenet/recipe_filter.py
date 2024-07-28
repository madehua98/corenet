import json
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent

def filter_and_write(item, txt_data):
    if item['id'] not in txt_data:
        return json.dumps(item)
    else:
        print(f"ID {item['id']} found in txt data")
    return None

def read_txt_files(txt_file, test_file):
    txt_data = set()
    # Read first txt file
    with open(txt_file, 'r') as f:
        txt_data.update(line.strip().split('/')[-1] for line in f)
    # Read additional test file
    with open(test_file, 'r') as f:
        txt_data.update(line.strip().split('/')[-1] for line in f)
    return txt_data

def read_jsonl_file(jsonl_file):
    with open(jsonl_file, 'r', encoding='utf-8') as infile:
        return [json.loads(line) for line in infile]

def process_items(items, txt_data, num_threads):
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(filter_and_write, (item, txt_data)) for item in items]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                results.append(result)
    return results

def process_tasks_in_parallel(jsonl_file, txt_file, test_file, output_file, num_processes=8, num_threads=4):
    txt_data = read_txt_files(txt_file, test_file)
    items = read_jsonl_file(jsonl_file)
    
    chunk_size = len(items) // num_processes
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    with multiprocessing.Pool(num_processes) as pool:
        results = pool.starmap(process_items, [(chunk, txt_data, num_threads) for chunk in chunks])

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for chunk_results in results:
            for result in chunk_results:
                outfile.write(result + '\n')

    print("数据过滤完成，并保存到", output_file)

if __name__ == "__main__":
    jsonl_file = '/ML-A100/team/mm/models/recipe1M+_1/image2labels_new.jsonl'
    txt_file = '/ML-A100/team/mm/models/recipe1M+_1/jpg_files.txt'
    test_file = '/ML-A100/team/mm/models/recipe1M+_1/test_recipe1m_id.txt'
    output_file = '/ML-A100/team/mm/models/recipe1M+_1/image2labels_filter.jsonl'
    process_tasks_in_parallel(jsonl_file, txt_file, test_file, output_file)
