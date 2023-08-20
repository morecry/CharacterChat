import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam
from torch import nn
import torch.nn.functional as F
import json
import datetime
import numpy as np
import random

# training arguments
class Config(object):
    def __init__(self):
        self.model_name = 'model_persona_score'
        self.save_path = 'model/output/saved_dict/'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_epoch = 1
        self.batch_size = 8
        self.pad_size = 512
        self.learning_rate = 1e-5
        self.bert_path='model/models/bert-base-uncased'
        self.tokenizer=BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size=768

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert1 = BertModel.from_pretrained(config.bert_path)
        self.bert2 = BertModel.from_pretrained(config.bert_path)
        for param in self.bert1.parameters():
            param.requires_grad = False
        for param in self.bert2.parameters():
            param.requires_grad = False
        
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.fc1 = nn.Linear(1, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        text1 = x[0]
        mask1 = x[1]
        text2 = x[2]
        mask2 = x[3]

        
        _, encoded_input1 = self.bert1(text1, attention_mask=mask1)
        _, encoded_input2 = self.bert2(text2, attention_mask=mask2)
        encoded_input = self.cos(encoded_input1, encoded_input2)
        encoded_input = encoded_input.unsqueeze(-1)
        
        encoded_input = self.fc1(encoded_input)
        out = self.fc2(encoded_input)
        return out

def load_dataset(file_path, config):
    pad_tok, cls_tok, sep_tok = '[PAD]', '[CLS]', '[SEP]'
    pad_size = config.pad_size
    contents=[]
    with open(file_path, 'r+', encoding='utf-8') as f:
        line = f.readline()
        while line:
            data = json.loads(line)
            data = data['conversations']
            seeker_statement = data['seeker_statement']
            supporter_statement = data['supporter_statement']
            score = data['rating']
            seeker_statement_ids = config.tokenizer.tokenize(seeker_statement)
            supporter_statement_ids = config.tokenizer.tokenize(supporter_statement)
            seeker_statement_ids = [cls_tok] + seeker_statement_ids + [sep_tok]
            supporter_statement_ids = [cls_tok] + supporter_statement_ids + [sep_tok]
            seeker_statement_ids = config.tokenizer.convert_tokens_to_ids(seeker_statement_ids)
            supporter_statement_ids = config.tokenizer.convert_tokens_to_ids(supporter_statement_ids)
            if len(seeker_statement_ids) <= pad_size:
                seeker_statement_ids = seeker_statement_ids + ([0] * (pad_size - len(seeker_statement_ids)))
                seeker_statement_mask = [1] * len(seeker_statement_ids) + ([0] * (pad_size - len(seeker_statement_ids)))
            else:
                seeker_statement_ids = seeker_statement_ids[:pad_size]
                seeker_statement_mask = [1] * pad_size
            if len(supporter_statement_ids) <= pad_size:
                supporter_statement_ids = supporter_statement_ids + ([0] * (pad_size - len(supporter_statement_ids)))
                supporter_statement_mask = [1] * len(supporter_statement_ids) + ([0] * (pad_size - len(supporter_statement_ids)))
            else:
                supporter_statement_ids = supporter_statement_ids[:pad_size]
                supporter_statement_mask = [1] * pad_size

            contents.append((seeker_statement_ids, seeker_statement_mask, supporter_statement_ids, supporter_statement_mask, score))

            line = f.readline()
    
    random.shuffle(contents)
    return contents

class DatasetIterator(object):
    def __init__(self, dataset, batch_size, device='cuda:0'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0
        self.device = device
        self.n_batches = len(dataset) // batch_size
        self.residue = False

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index*self.batch_size:len(self.dataset)]
            self.index += 1
            batches=self._to_tensor(batches)
        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index += 1
            if not batches:
                self.index = 0
                raise StopIteration
            batches = self._to_tensor(batches)
        return batches

    def _to_tensor(self, datas):
        x1 = torch.LongTensor([item[0] for item in datas]).to(self.device)
        mask1 = torch.LongTensor([item[1] for item in datas]).to(self.device)
        x2 = torch.LongTensor([item[2] for item in datas]).to(self.device)
        mask2 = torch.LongTensor([item[3] for item in datas]).to(self.device)
        score = torch.FloatTensor([[item[4]] for item in datas]).to(self.device)
        return (x1, mask1, x2, mask2), score

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches
        else:
            return self.n_batches + 1
        
def train(config, model, train_iter, dev_iter):
    start_time = datetime.datetime.now()
    print('start training...', start_time)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    total_batch=0
    dev_best_loss=10
    flag=False

    for epoch in range(config.num_epoch):
        print('Epoch[{}/{}]'.format(epoch+1,config.num_epoch))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                dev_loss = evaluate(model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss=dev_loss
                    torch.save(model.state_dict(), config.save_path+config.model_name+'_best.ckpt')
                    improve = '*'
                else:
                    improve = ''
                
                msg = 'Iter: {} / {}, Train Loss: {}, {} {}'.format(total_batch, len(train_iter)*config.num_epoch, dev_loss, datetime.datetime.now(), improve)
                print(msg)
                model.train()

            total_batch += 1
            
        if flag:
            break

        torch.save(model.state_dict(), config.save_path+config.model_name+'_epoch_{}.ckpt'.format(str(epoch)))

def evaluate(model, dev_iter):
    model.eval()

    loss_total = 0
    predict_all= np.array([],dtype=int)
    labels_all= np.array([],dtype=int)
    criterion = nn.MSELoss()

    with torch.no_grad():
        for texts, labels in dev_iter:
            outputs=model(texts)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            
            predict = torch.max(outputs.data,1)[1].cpu().numpy()
            labels_all=np.append(labels_all, labels)
            predict_all=np.append(predict_all, predict)
    return loss_total / len(dev_iter)

if __name__ == '__main__':
    random.seed(0)
    dataset_path = 'model/dataset/model_persona_score_dataset.json'
    model_name = 'persona_score'
    config = Config()
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic=True

    print('load dataset...')
    dataset = load_dataset(dataset_path, config)
    train_num = int(len(dataset) * 0.95)
    train_dataset = dataset[:train_num]
    dev_dataset = dataset[train_num:]
    train_iter = DatasetIterator(train_dataset, config.batch_size)
    dev_iter = DatasetIterator(dev_dataset, config.batch_size)

    
    model = Model(config).to(config.device)
    train(config, model, train_iter, dev_iter)