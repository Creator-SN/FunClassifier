import torch
from utils import Tokenizer
from models.TextCNN import TextCNN
from models.TextRNN import TextRNN
from models.Transformer import Transformer


class Predictor():

    def __init__(self, config, resume_path=False, gpu=0):
        self.config = config
        self.vocab_path = config.vocab_path
        self.tokenizer = Tokenizer(vocab_path=self.vocab_path)
        self.model_init()

        device = torch.device("cuda:{}".format(
            gpu) if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        if not resume_path == False:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            model_dict = torch.load(resume_path).module.state_dict()
            self.model.load_state_dict(model_dict)
        self.model.to(device)
        self.model.eval()

    def model_init(self):
        print('AutoModel Choose Model: {}\n'.format(self.config.model_name))
        if self.config.model_name == 'TextCNN':
            self.model = TextCNN(self.config)
        elif self.config.model_name == 'TextRNN':
            self.model = TextRNN(self.config)
        elif self.config.model_name == 'Transformer':
            self.model = Transformer(self.config)

    def data_process(self, X):
        if type(X) == str:
            X = [X]
        input_ids = [self.tokenizer(item)['input_ids'] for item in X]
        self.padding_length = max([len(item) for item in input_ids])
        input_ids = [item + [0] * (self.padding_length - len(item))
                     for item in input_ids]
        input_ids = torch.tensor(input_ids)
        return input_ids

    def __call__(self, X):
        return self.predict(X)

    def predict(self, X):
        inputd_ids = self.cuda(self.data_process(X))
        pred = self.model(inputd_ids)
        return torch.max(pred, dim=1)[1].tolist()

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
