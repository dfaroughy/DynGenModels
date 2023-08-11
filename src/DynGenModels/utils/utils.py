import numpy as np
import os.path
import shutil
import itertools
from tabulate import tabulate


def make_dir(path, overwrite=False, sub_dirs=False, verbose=True):  
    Directory = path
    if overwrite:
        shutil.rmtree(Directory, ignore_errors=True)
        os.mkdir(Directory)
    else:
        for I in itertools.count():
            Directory = path + '__' + str(I+1)
            if not os.path.isdir(Directory):
                os.mkdir(Directory)
                break
            else:
                continue
    if sub_dirs:
        for d in sub_dirs: 
            os.mkdir(Directory+'/'+d)
    if verbose:
        info = "INFO: created directory: {}".format(Directory)
        print("#"+"="*len(info))
        print(info)
        print("#"+"="*len(info))
    return Directory

def print_table(data):
    table = []
    for key, value in data.items():
        if isinstance(value, dict):
            value = "\n".join([f"{k}: {v}" for k, v in value.items()])
        table.append([key, value])
    print(tabulate(table, headers=["Key", "Value"], tablefmt="pretty", colalign=("left", "left")))

def save_data(samples: dict, name: str, workdir : str, verbose: bool = True):
    for key in samples.keys():
        sample = samples[key].numpy()
        path = '{}/results/{}_{}.npy'.format(workdir, name, key) 
        np.save(path, sample)
    if verbose:
        print("INFO: saved {} data in {}".format(name, workdir))

def savefig(filename, extension="png"):
    counter = 1
    base_filename, ext = os.path.splitext(filename)
    if ext == "":
        ext = f".{extension}"
    unique_filename = f"{base_filename}{ext}"
    while os.path.exists(unique_filename):
        unique_filename = f"{base_filename}_{counter}{ext}"
        counter += 1
    return unique_filename        