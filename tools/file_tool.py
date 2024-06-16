import pickle

def read_pkl_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except Exception as e:
        print(f"Error reading the .pkl file: {e}")
        return None
    
def write_pkl_file(data, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
            print(f"Successfully wrote data to {file_path}")
    except Exception as e:
        print(f"Error writing the .pkl file: {e}")