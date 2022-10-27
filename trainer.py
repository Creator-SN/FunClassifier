import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
from models.TextCNN import TextCNN
from models.TextRNN import TextRNN
from models.Transformer import Transformer
from dataloader import THUCNewsDataset
from utils import Tokenizer, Analysis
from tqdm import tqdm


class Trainer():

    def __init__(self, train_path, eval_path, model_config, vocab_path=None, padding_length=50, batch_size=16, batch_size_eval=64, task_name='TextClassification'):
        self.train_path = train_path
        self.eval_path = eval_path
        self.vocab_path = vocab_path
        self.tokenizer = Tokenizer([train_path, eval_path], vocab_path)
        self.task_name = task_name
        self.padding_length = padding_length
        self.config = model_config
        self.model_name = model_config.model_name
        self.config.n_vocab = len(self.tokenizer.vocab)
        self.analysis = Analysis()
        self.dataloader_init(train_path, eval_path,
                             padding_length, batch_size, batch_size_eval)
        self.model_init()

    def model_init(self):
        print('AutoModel Choose Model: {}\n'.format(self.config.model_name))
        if self.config.model_name == 'TextCNN':
            self.model = TextCNN(self.config)
        elif self.config.model_name == 'TextRNN':
            self.model = TextRNN(self.config)
        elif self.config.model_name == 'Transformer':
            self.model = Transformer(self.config)

    def dataloader_init(self, train_path, eval_path, padding_length, batch_size, batch_size_eval):
        self.train_loader = DataLoader(THUCNewsDataset(
            self.tokenizer, train_path, padding_length), batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(THUCNewsDataset(
            self.tokenizer, eval_path, padding_length), batch_size=batch_size_eval, shuffle=False)

    def __call__(self, resume_path=None, resume_step=None, num_epochs=30, lr=1e-3, gpu=[0], eval_call_epoch=None):
        return self.train(resume_path=resume_path, resume_step=resume_step,
                          num_epochs=num_epochs, lr=lr, gpu=gpu, eval_call_epoch=eval_call_epoch)

    def train(self, resume_path=None, resume_step=None, num_epochs=30, lr=1e-3, gpu=[0], eval_call_epoch=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()

        if resume_path is not None:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            model_dict = torch.load(resume_path).module.state_dict()
            self.model.module.load_state_dict(model_dict)
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)

        current_uid = str(uuid.uuid1()).split('-')[0]

        train_step = resume_step if resume_step is not None else 0
        for epoch in range(num_epochs):
            train_count = 0
            train_loss = 0
            train_acc = []

            train_iter = tqdm(self.train_loader)
            self.model.train()

            for it in train_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])

                model_output = self.model(it['input_ids'])
                loss = F.cross_entropy(model_output, it['labels'])

                loss = loss.mean()

                loss.backward()
                scheduler.step()
                optimizer.step()
                self.model.zero_grad()

                train_loss += loss.data.item()
                train_step += 1
                train_count += 1

                gold = it['labels']
                pred = torch.argmax(model_output, dim=1)
                train_acc.append((gold == pred).sum().item() / len(gold))

                train_iter.set_description(
                    'Train: {}/{}'.format(epoch + 1, num_epochs))
                train_iter.set_postfix(
                    train_loss=train_loss / train_count, train_acc=np.mean(train_acc))

            self.analysis.append_train_record({
                'epoch': epoch + 1,
                'train_loss': train_loss / train_count,
                'train_acc': np.mean(train_acc)
            })

            model_uid = self.save_model(train_step)
            if eval_call_epoch is None or eval_call_epoch(epoch):
                self.eval(epoch)

            self.analysis.save_all_records(
                uid=current_uid if self.task_name is None else self.task_name)
            yield (epoch, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, model_uid)

    def save_model(self, current_step=0):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        if not os.path.exists('./save_model/{}/{}'.format(dir, self.model_name)):
            os.makedirs('./save_model/{}/{}'.format(dir, self.model_name))
        torch.save(
            self.model, './save_model/{}/{}/{}_{}.pth'.format(dir, self.model_name, self.model_name, current_step))
        self.analysis.append_model_record(current_step)
        return current_step

    def eval(self, epoch):
        with torch.no_grad():
            eval_count = 0
            eval_loss = 0
            eval_acc = []

            eval_iter = tqdm(self.eval_loader)
            self.model.eval()

            for it in eval_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])

                model_output = self.model(it['input_ids'])
                loss = F.cross_entropy(model_output, it['labels'])

                loss = loss.mean()

                eval_loss += loss.data.item()
                eval_count += 1

                gold = it['labels']
                pred = torch.argmax(model_output, dim=1)
                eval_acc.append((gold == pred).sum().item() / len(gold))

                eval_iter.set_description(
                    'Eval: {}'.format(epoch + 1))
                eval_iter.set_postfix(
                    eval_loss=eval_loss / eval_count, eval_acc=np.mean(eval_acc))

            self.analysis.append_eval_record({
                'epoch': epoch + 1,
                'eval_loss': eval_loss / eval_count,
                'eval_acc': np.mean(eval_acc)
            })

    def cuda(self, inputX):
        if type(inputX) == tuple:
            if torch.cuda.is_available():
                result = []
                for item in inputX:
                    result.append(item.cuda())
                return result
            return inputX
        else:
            if torch.cuda.is_available():
                return inputX.cuda()
            return inputX
