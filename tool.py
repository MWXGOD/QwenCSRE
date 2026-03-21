import json

# 对比的时候需要把实体关系都变成小写

class Hyperargs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k ,v)


def read_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
    
