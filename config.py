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
        if 'embedding_pretrained_file' in config:
            self.set_embedding_pretrained(config["embedding_pretrained_file"])  # 预训练词向量
        else:
            self.embedding_pretrained = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')   # 设备
    
    def set_embedding_pretrained(self, embedding_pretrained_file):
        self.embedding_pretrained = torch.tensor(
                np.load(embedding_pretrained_file)["embeddings"].astype('float32'))  # 预训练词向量
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度

