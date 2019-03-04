import pickle
import sys
from os import listdir
from os.path import isfile, join


if __name__ == "__main__":
    pickle_path = "./datasets/sampleable/results/"
    file_paths = [pickle_path + f for f in listdir(pickle_path) if isfile(join(pickle_path, f))]
    
    for index, file_path in enumerate(file_paths):
        try:
            result = pickle.load(open(file_path, 'rb'))
            print(index + 1, ": ", file_path, ": ", result["message"])
        except AttributeError as e:
            print(e)

        

