import json
import torch
import numpy as np

class Config(object):
    def __init__(self, from_json):
        with open(from_json, 'r') as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(self, k, v)
        # 一些进一步处理的参数
        self.embedding_pretrained = torch.tensor(
            np.load(config["embedding_pretrained"])["embeddings"].astype('float32')) # 预训练词向量
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备