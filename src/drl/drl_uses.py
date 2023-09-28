from src.utils.filesys import *


def load_cfgs(path, *keys):
    return load_dict_json(getpath(path, 'cfgs.json'), *keys)


def load_performance(path, *keys):
    return load_singlerow_csv(getpath(path, 'performance.csv'), *keys)
    pass
