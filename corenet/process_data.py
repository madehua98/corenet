import os
import tarfile
import pickle
import multiprocessing
import re
from tqdm import tqdm

# # Read the text file and populate result_dict
# text_path = '/media/fast_data/Food2k_complete/food2k_label2name_en.txt'
# texts = []
# result_dict = {}
# with open(text_path, 'r', encoding='utf-8') as fr:
#     for line in fr.readlines():
#         match = re.match(r"(\d+)--(.+)", line)
#         if match:
#             number_str = match.group(1).lstrip('0')
#             number = int(number_str) if number_str else 0
#             text = match.group(2)
#             result_dict[number] = text
#         else:
#             print("No match found")

# # Collect image paths and their corresponding text data
# image_dirs = [f'/media/fast_data/Food2k_complete/{i}' for i in range(2000)]
# image_paths = []
# text_datas = []

# for idx, image_dir in enumerate(tqdm(image_dirs, desc="Reading image directories")):
#     files = os.listdir(image_dir)
#     text_data = result_dict[idx]
#     for file in files:
#         if file.endswith('.jpg'):
#             image_paths.append(os.path.join(image_dir, file))
#             text_datas.append(text_data)

# print(len(image_paths))
# print(len(text_datas))

# root_dir = '/media/fast_data/catlip_data/image_text_data'
# max_files_per_tar = 10000

# os.makedirs(root_dir, exist_ok=True)

# metadata = {
#     'total_tar_files': 0,
#     'max_files_per_tar': max_files_per_tar,
#     'tar_file_names': []
# }

# manager = multiprocessing.Manager()
# shared_metadata = manager.dict({
#     'total_tar_files': 0,
#     'max_files_per_tar': max_files_per_tar,
#     'tar_file_names': manager.list()
# })

# def convert_image(image_path):
#     with open(image_path, 'rb') as image_file:
#         binary_data = image_file.read()
#     return binary_data

# def save_batch(start_index, end_index, shared_metadata):
#     local_metadata = {
#         'tar_file_names': []
#     }
#     for i in tqdm(range(start_index, end_index, max_files_per_tar), desc="Saving tar files"):
#         tar_index = i // max_files_per_tar
#         tar_file_name = f'{tar_index}.tar.gz'
#         tar_file_path = os.path.join(root_dir, tar_file_name)
#         local_metadata['tar_file_names'].append(tar_file_name)

#         with tarfile.open(tar_file_path, 'w:gz') as tar:
#             for j in range(i, min(i + max_files_per_tar, end_index)):
#                 index_in_tar = j % max_files_per_tar
#                 image_text_dict = {'image': convert_image(image_paths[j]), 'text': text_datas[j]}
#                 pkl_file_name = f'{index_in_tar}.pkl'
#                 pkl_file_path = os.path.join(root_dir, pkl_file_name)
                
#                 with open(pkl_file_path, 'wb') as pkl_file:
#                     pickle.dump(image_text_dict, pkl_file)
                
#                 tar.add(pkl_file_path, arcname=pkl_file_name)
#                 os.remove(pkl_file_path)
    
#     shared_metadata['tar_file_names'].extend(local_metadata['tar_file_names'])
#     shared_metadata['total_tar_files'] += len(local_metadata['tar_file_names'])

# total_pairs = len(image_paths)
# batch_start, batch_end = 0, total_pairs

# pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

# # Process the entire range without batches
# pool.apply_async(save_batch, args=(batch_start, batch_end, shared_metadata))

# pool.close()
# pool.join()

# metadata['total_tar_files'] = shared_metadata['total_tar_files']
# metadata['tar_file_names'] = list(shared_metadata['tar_file_names'])

# metadata_file_path = os.path.join(root_dir, 'metadata.pkl')
# with open(metadata_file_path, 'wb') as metadata_file:
#     pickle.dump(metadata, metadata_file)

# print("Dataset organization complete")



import pybase64
import torch
from PIL import Image, ImageFile
import io



# file_name = '/media/fast_data/catlip_data/cache/0/8271.pkl'
# with open(file_name, "rb") as handle:
#     data = pickle.load(handle)

# print(data['image'], data['text'])
# # img_bytes = pybase64.b64decode(data["image"], validate=True)  

# image = Image.open(io.BytesIO(data['image'])).convert("RGBA").convert("RGB")
# print(image)


import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Define the directories
source_dir = "/media/fast_data/catlip_data/image_text_data"
destination_dir = "/media/fast_data/catlip_data/cache"

# Check if the source directory exists
if not os.path.exists(source_dir):
    raise FileNotFoundError(f"The directory {source_dir} does not exist.")

# List all tar.gz files in the source directory
tar_files = [f for f in os.listdir(source_dir) if f.endswith('.tar.gz')]

# Function to extract a tar.gz file to a specific directory
def extract_tar_file(tar_file):
    tar_path = os.path.join(source_dir, tar_file)
    extract_path = os.path.join(destination_dir, tar_file.replace('.tar.gz', ''))
    os.makedirs(extract_path, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    return extract_path

# Function to rename files in the extracted directory
def rename_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl') and file[:-4].isdigit():  # Check if filename is a digit and ends with .pkl
                old_path = os.path.join(root, file)
                dir_name = os.path.basename(root)
                new_filename = str(int(dir_name) * 10000 + int(file[:-4]) % 10000) + '.pkl'
                new_path = os.path.join(root, new_filename)
                os.rename(old_path, new_path)
                print(f'Renamed: {old_path} to {new_path}')

# Function to extract and rename files
def extract_and_rename(tar_file):
    extracted_dir = extract_tar_file(tar_file)
    rename_files(extracted_dir)

# Extract and rename all tar.gz files to the destination directory with threading and a progress bar
if __name__ == '__main__':
    with tqdm(total=len(tar_files), desc="Extracting and renaming tar.gz files") as pbar:
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(extract_and_rename, tar_file) for tar_file in tar_files]
            for future in as_completed(futures):
                pbar.update(1)

    # List the contents of the destination directory to verify extraction and renaming
    extracted_dirs = os.listdir(destination_dir)
    print(extracted_dirs)

