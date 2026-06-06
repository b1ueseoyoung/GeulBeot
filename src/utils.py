import json
import pickle

def load_cache(cache_path, pickle_file=False):
    try:
        if pickle_file:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(cache_path, 'r') as f:
                return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache, cache_path, pickle_file=False):
    import os
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if pickle_file:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
    else:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
