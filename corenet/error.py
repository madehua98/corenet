import json
from tqdm import tqdm
import numpy as np
import os

def save_json(filename, object):
    with open(filename, mode='w') as file:
        json.dump(object, file, indent=4)

def get_sorted_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    subdirectories.sort()
    return subdirectories

def create_directory_dict(subdirectories):
    return {i: subdirectory for i, subdirectory in enumerate(subdirectories)}

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

def process_jsonl_objects(objects, directory_dict):
    matric = np.zeros((101, 101))
    count = {}
    for obj in objects:
        target = obj["targets"]
        pred_label = obj["pred_label"]
        matric[target, pred_label] += 1
        
        if target in count:
            count[target] += 1
        else:
            count[target] = 1

    count_sorted = dict(sorted(count.items(), key=lambda item: item[1], reverse=True))
    count_new = {directory_dict[k]: v for k, v in count_sorted.items()}

    indices = np.argwhere(matric >= 5)
    values_with_indices = [(index, matric[tuple(index)]) for index in indices]
    sorted_values_with_indices = sorted(values_with_indices, key=lambda x: x[1], reverse=True)

    return count_new, sorted_values_with_indices

def save_sorted_values(filename, sorted_values_with_indices, directory_dict):
    with open(filename, 'w') as file:
        for index, value in sorted_values_with_indices:
            index_list = [directory_dict[index[0]], directory_dict[index[1]]]
            file.write(f"正确/错误标签分别为{index_list},, 出现次数为: {value}\n")

def main():
    # Paths
    images_path = "/ML-A100/team/mm/models/food101/food101/images"
    jsonl_file_path = '/ML-A100/team/mm/models/catlip_data/single_small_500_dci/output.jsonl'
    error_count_filename = "/ML-A100/team/mm/models/catlip_data/single_small_500_dci/error_count.json"
    sorted_values_filename = '/ML-A100/team/mm/models/catlip_data/single_small_500_dci/sorted_values.txt'

    # Get sorted subdirectories and create dictionary
    sorted_subdirectories = get_sorted_subdirectories(images_path)
    directory_dict = create_directory_dict(sorted_subdirectories)
    
    # Load JSONL objects
    objects = load_jsonl(jsonl_file_path)
    
    # Process objects and generate results
    count_new, sorted_values_with_indices = process_jsonl_objects(objects, directory_dict)
    
    # Save results
    save_json(error_count_filename, count_new)
    save_sorted_values(sorted_values_filename, sorted_values_with_indices, directory_dict)

    # Print dictionary
    print(directory_dict)

if __name__ == "__main__":
    main()
