# %%
import os
import json
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset


class THUCNewsDataset(Dataset):
    def __init__(self, tokenizer, file_name, padding_length=128, shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.ori_list = self.load_train(file_name)
        if shuffle:
            random.shuffle(self.ori_list)

    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        if ori_list[-1] == '':
            ori_list = ori_list[:-1]
        return ori_list
    
    def get_vocab_len(self):
        return len(self.tokenizer.vocab)

    def __getitem__(self, idx):
        item = self.ori_list[idx]
        sentence, label = item.split('\t')
        labels = int(label)
        T = self.tokenizer(sentence, max_length=self.padding_length)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_ids = torch.tensor(T['token_type_ids'])
        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'token_type_ids': token_type_ids,
            'labels': torch.tensor(labels)
        }

    def __len__(self):
        return len(self.ori_list)
