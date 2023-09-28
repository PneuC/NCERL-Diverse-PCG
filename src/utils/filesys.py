import json
import os
import re
import pandas as pds
from root import PRJROOT


def auto_dire(path=None, name='trial'):
    dire_id = 0
    folder = PRJROOT if path is None else getpath(path)
    tar = os.path.join(folder, f'{name}{dire_id}')
    while os.path.exists(tar):
        dire_id += 1
        tar = os.path.join(folder, f'{name}{dire_id}')
    os.makedirs(tar)
    return tar


def getpath(*args):
    """ if is absolute root_folder or related path, return {path}, else return {PRJROOT/path} """
    path = os.path.join(*args)
    if os.path.isabs(path) or re.match(r'\.+[\\/].*', path):
        return path
    else:
        return os.path.join(PRJROOT, path)

def load_dict_json(path, *keys):
    with open(getpath(path), 'r') as f:
        data = json.load(f)
    if len(keys) == 1:
        return data[keys[0]]
    return tuple(data[key] for key in keys)

def load_singlerow_csv(path, *keys):
    data = pds.read_csv(getpath(path))
    if len(keys) == 1:
        return data[keys[0]][0]
    return (data[key][0] for key in keys)

if __name__ == '__main__':
    print(os.path.join('smn', '**', 'dsf'))
    pass
