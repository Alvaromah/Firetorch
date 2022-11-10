import json
import pickle

def load_json(fn):
    with open(fn, 'r', encoding='UTF-8') as fp:
        return json.load(fp)

def save_json(obj, fn):
    with open(fn, 'w', encoding='UTF-8') as fp:
        json.dump(obj, fp, indent=2)

def load_object(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

def save_object(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp, pickle.HIGHEST_PROTOCOL)

