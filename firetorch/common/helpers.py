import pickle

def load_object(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

def save_object(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp, pickle.HIGHEST_PROTOCOL)

