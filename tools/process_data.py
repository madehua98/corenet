import pickle
from file_tool import write_pkl_file, read_pkl_file



#file_path = '/ML-A100/team/mm/models/catlip_data/cache/metadata.pkl'
file_new_path = '/ML-A100/team/mm/models/catlip_data/cache/metadata.pkl'
metadata = {}
metadata['total_tar_files'] = 1
metadata['max_files_per_tar'] = 10000
metadata['tar_file_names'] = '0.tar.gz'
write_pkl_file(metadata, file_new_path)