# This script is to store util functions

import json


def get_config(path):
    # for now, config file is a json file.
    with open(path, 'r', encoding='utf-8') as f:
        dic = json.load(f)

    # return config class
    return Config(dic)

class Config(object):
    def __init__(self, dic):
        for j, k in dic.items():
            if isinstance(k, dict):
                setattr(self, j, Config(k))
            else:
                if k is not 'None':
                    setattr(self, j, k)
                else:
                    setattr(self, j, None)
        self.dic = dic

    def __len__(self):
        return len(self.dic.keys())

    def __getitem__(self, key):
        return self.dic[key]
    
    def keys(self):
        return self.dic.keys()
    
    def values(self):
        return self.dic.values()
