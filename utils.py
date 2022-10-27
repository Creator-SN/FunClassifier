# coding: UTF-8
import os
import torch
import pickle as pkl

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

class Tokenizer():
    def __init__(self, datasets=None, vocab_path=None):
        if vocab_path is not None:
            vocab = pkl.load(open(vocab_path, 'rb'))
        elif datasets is not None:
            vocab = self.build_vocab(datasets)
            pkl.dump(vocab, open('new_vocab.pkl', 'wb'))

        self.vocab = vocab
        print(f"Vocab size: {len(vocab)}")

    def build_vocab(self, datasets):
        word_to_id = {}
        for dataset in datasets:
            with open(dataset, 'r', encoding='utf-8') as f:
                for line in f:
                    content, _ = line.strip().split('\t')
                    for word in content:
                        if word not in word_to_id:
                            word_to_id[word] = len(word_to_id)
        word_to_id.update({UNK: len(word_to_id), PAD: len(word_to_id) + 1})
        return word_to_id
    
    def encode(self, sentence, max_length):
        tokens = [self.vocab.get(token, self.vocab[UNK]) for token in sentence]
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        final_tokens = tokens + [self.vocab[PAD]] * (max_length - len(tokens))
        input_ids = final_tokens
        attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
        token_type_ids = [0] * max_length
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
    
    def __call__(self, sentence, max_length=2048):
        return self.encode(sentence, max_length)

class Analysis():

    def __init__(self):
        self.train_record = {}
        self.eval_record = {}
        self.model_record = {}

    '''
    append data record of train
    train_record_item: dict
    '''

    def append_train_record(self, train_record_item):
        for key in train_record_item:
            if key not in self.train_record:
                self.train_record[key] = []
            self.train_record[key].append(train_record_item[key])

    '''
    append data record of eval
    eval_record_item: dict
    '''

    def append_eval_record(self, eval_record_item):
        for key in eval_record_item:
            if key not in self.eval_record:
                self.eval_record[key] = []
            self.eval_record[key].append(eval_record_item[key])

    '''
    append data record of model
    uid: model uid
    '''

    def append_model_record(self, uid):
        key = "model_uid"
        if key not in self.model_record:
            self.model_record[key] = []
        self.model_record[key].append(uid)

    def save_all_records(self, uid):
        self.save_record('train_record', uid)
        self.save_record('eval_record', uid)
        self.save_record('model_record', uid)

    def save_record(self, record_name, uid):
        record_dict = getattr(self, record_name)
        path = f'./data_record/{uid}'
        if not os.path.exists(path):
            os.makedirs(path)
        head = []
        for key in record_dict:
            head.append(key)
        result = ''
        for idx in range(len(record_dict[head[0]])):
            for key in head:
                result += str(record_dict[key][idx]) + '\t'
            result += '\n'

        result = "\t".join(head) + '\n' + result

        with open(f'{path}/{record_name}.csv', encoding='utf-8', mode='w+') as f:
            f.write(result)

        return uid