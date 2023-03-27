import json
import pickle


def save_file(file_name, data):
    if file_name.endswith('.json'):
        with open(file_name, 'w') as f:
            json.dump(data, f)
    elif file_name.endswith('.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError("Not yet supported output format")