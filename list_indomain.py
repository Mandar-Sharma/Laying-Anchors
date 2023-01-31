import argparse
 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-type", "--model_type", help="Model Type")
parser.add_argument("-m", "--model", help="Model Folder")
parser.add_argument("-mode", "--list_mode", help="List Mode - MAX or MIN")

args = parser.parse_args()
config = vars(args)

import math
import torch
import random
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from random import sample
from decimal import Decimal
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast

with open('./WikiText103/nums', 'rb') as fp:
    numerals = pickle.load(fp)

numerals = [int(x) for x in numerals if round(math.log(int(x), 10)) < 10]
numerals = sorted(list(set(numerals)))

_100 = sorted(numerals[:100])

seen_set_100 = []
train_100 = []
for i in tqdm(range(800)):
    sam = random.sample(_100, 5)
    while sam in seen_set_100:
        sam = random.sample(_100, 5)
    seen_set_100.append(sam)
    train_100.append(sam)

test_100 = []
for i in tqdm(range(200)):
    sam = random.sample(_100, 5)
    while sam in seen_set_100:
        sam = random.sample(_100, 5)
    seen_set_100.append(sam)
    test_100.append(sam)

print("test_100 : {} and train_100 : {} samples.".format(len(test_100), len(train_100)))

_1000 = numerals[100:1000]
seen_set_1000 = []
train_1000 = []
for i in tqdm(range(800)):
    sam = random.sample(_1000, 5)
    while sam in seen_set_1000:
        sam = random.sample(_1000, 5)
    seen_set_1000.append(sam)
    train_1000.append(sam)

test_1000 = []
for i in tqdm(range(200)):
    sam = random.sample(_1000, 5)
    while sam in seen_set_1000:
        sam = random.sample(_1000, 5)
    seen_set_1000.append(sam)
    test_1000.append(sam)

_10000 = []
for i in list(range(1000, 10001)):
    if i in numerals:
        _10000.append(i)

seen_set_10000 = []
train_10k = []
for i in tqdm(range(800)):
    sam = random.sample(_10000, 5)
    while sam in seen_set_10000:
        sam = random.sample(_10000, 5)
    seen_set_10000.append(sam)
    train_10k.append(sam)

test_10k = []
for i in tqdm(range(200)):
    sam = random.sample(_10000, 5)
    while sam in seen_set_10000:
        sam = random.sample(_10000, 5)
    seen_set_10000.append(sam)
    test_10k.append(sam)

train_B10k, test_B10k = [], []
_B10k = numerals[4852:]
seen_set_B10k = []
for i in tqdm(range(800)):
    sam = random.sample(_B10k, 5)
    while sam in seen_set_B10k:
        sam = random.sample(_B10k, 5)
    seen_set_B10k.append(sam)
    train_B10k.append(sam)

for i in tqdm(range(200)):
    sam = random.sample(_B10k, 5)
    while sam in seen_set_B10k:
        sam = random.sample(_B10k, 5)
    seen_set_B10k.append(sam)
    test_B10k.append(sam)

train, test = [], []
train.extend(train_100)
train.extend(train_1000)
train.extend(train_10k)
train.extend(train_B10k)

test.extend(test_100)
test.extend(test_1000)
test.extend(test_10k)
test.extend(test_B10k)

with open('./WikiText103/gmm_means/means_1000', 'rb') as fp:
    means = pickle.load(fp)
with open('./WikiText103/gmm_means/log/means_1000', 'rb') as fp:
    means_log = pickle.load(fp)
means = list(set(sorted([round(x) for x in means.flatten()])))
means_log = list(set(sorted([round(x) for x in means_log.flatten()])))

def log_squash_2(num):
    if num > 1:
        return round((np.log(num) + 1) * 10)
    elif num < -1:
        return round((-np.log(-num) -1) * 10)
    else:
        return round(num * 10)
    
def find_anc(i):
    anchor = min(means, key=lambda x:abs(x-i))
    return anchor

def find_anc_log(i):
    anchor = min(means_log, key=lambda x:abs(x-log_squash_2(i)))
    return anchor

class LSTM(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=4, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        self.act = nn.Sigmoid()

    def forward(self, x):
        lstm_out, (ht, ct) = self.lstm(x)
        return self.act(self.linear(ht[-1]))


class NumeralDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def loss_batch(model, xb, yb, opt=None):
    preds = model(xb)
    loss = F.cross_entropy(preds, yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def evaluate(model, valid_dl):
    with torch.no_grad():
        results = [loss_batch(model, xb, yb) for xb, yb in valid_dl]
        losses, nums = zip(*results)
        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums)) / total
    return avg_loss, total


def fit(epochs, lr, model, train_dl, valid_dl):
    min_loss = float('inf')

    losses = []
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        for xb, yb in train_dl:
            loss, _ = loss_batch(model, xb, yb, opt)
        result = evaluate(model, valid_dl)
        val_loss, total = result

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), './C1.pt')

        losses.append(val_loss)

        if epoch % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, val_loss))

    return losses


def accuracy(model, valid_dl):
    acc = []
    for xb, yb in valid_dl:
        preds = model(xb)
        preds = np.argmax(preds.detach().cpu().numpy() > 0.5, axis=1)
        acc.append(accuracy_score(yb.detach().cpu().numpy(), preds))
    return np.mean(acc)

llm_device = torch.device('cuda:0')

tokenizer = BertTokenizerFast.from_pretrained(config['model'])
llm_model = BertForMaskedLM.from_pretrained(config['model'], output_hidden_states=True)
llm_model.to(llm_device)


def get_embeddings_list(numbers_list, mode=config['list_mode']):
    X = []
    y = []

    for i in tqdm(numbers_list):
        X_sub = []
        for j in i:
            if config['model_type'] == 'anc':
                anc = find_anc(j)
                input_str = str(j) + " <ANC> " + str(anc)
            elif config['model_type'] == 'lr_anc':
                anc = find_anc(j)
                if (j - anc) > 0:
                    input_str = str(j) + ' <LA> ' + str(anc)
                else:
                    input_str = str(j) + ' <RA> ' + str(anc)
            elif config['model_type'] == 'log_anc':
                anc = find_anc_log(j)
                input_str = str(j) + " <ANC> " + str(anc)
            elif config['model_type'] == 'lr_log_anc':
                anc = find_anc_log(j)
                if (log_squash_2(j) - anc) > 0:
                    input_str = str(j) + ' <LA> ' + str(anc)
                else:
                    input_str = str(j) + ' <RA> ' + str(anc)
            else:
                input_str = str(j)
            input_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0)
            outputs = llm_model(input_ids.to(llm_device))
            last_four = torch.stack(outputs['hidden_states'][-4:]).sum(0)
            emb = last_four[0][1:-1].mean(dim=0)
            X_sub.append(emb.detach().cpu().numpy())
            
            del input_ids
            del outputs
            del last_four
            del emb
            torch.cuda.empty_cache()
            
        X.append(X_sub)
        idx = [0, 0, 0, 0, 0]
        if mode == 'MAX':
            idx[i.index(max(i))] = 1
        elif mode == 'MIN':
            idx[i.index(min(i))] = 1
        y.append(idx)

    return X, y

def main():
    X_train_100, y_train_100 = get_embeddings_list(train_100)
    X_test_100, y_test_100 = get_embeddings_list(test_100)

    X_train = np.array([np.array(x) for x in X_train_100])
    X_test = np.array([np.array(x) for x in X_test_100])
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)

    y_train = np.array([y.index(1) for y in y_train_100])
    y_train = y_train.astype(np.int64)
    y_test = np.array([y.index(1) for y in y_test_100])
    y_test = y_test.astype(np.int64)

    train_dataset = NumeralDataset(X_train, y_train)
    test_dataset = NumeralDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)

    device = torch.device('cuda:0')
    train_dl = DeviceDataLoader(train_loader, device)
    valid_dl = DeviceDataLoader(test_loader, device)

    model = LSTM(768, 1536)
    model.to(device)

    losses = fit(150, 1e-4, model, train_dl, valid_dl)

    model_eval = LSTM(768, 1536)

    model_eval.load_state_dict(torch.load('./C1.pt'))
    model_eval.to(device)

    print("Range [0,100]")
    print(accuracy(model, valid_dl))
    print()

    X_train_1000, y_train_1000 = get_embeddings_list(train_1000)
    X_test_1000, y_test_1000 = get_embeddings_list(test_1000)

    X_train = np.array([np.array(x) for x in X_train_1000])
    X_test = np.array([np.array(x) for x in X_test_1000])
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)

    y_train = np.array([y.index(1) for y in y_train_1000])
    y_train = y_train.astype(np.int64)
    y_test = np.array([y.index(1) for y in y_test_1000])
    y_test = y_test.astype(np.int64)

    train_dataset = NumeralDataset(X_train, y_train)
    test_dataset = NumeralDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)

    train_dl = DeviceDataLoader(train_loader, device)
    valid_dl = DeviceDataLoader(test_loader, device)

    model = LSTM(768, 1536)
    model.to(device)

    losses = fit(150, 1e-4, model, train_dl, valid_dl)

    model_eval = LSTM(768, 1536)

    model_eval.load_state_dict(torch.load('./C1.pt'))
    model_eval.to(device)

    print("Range [100,1000]")
    print(accuracy(model, valid_dl))
    print()

    X_train_10k, y_train_10k = get_embeddings_list(train_10k)
    X_test_10k, y_test_10k = get_embeddings_list(test_10k)

    X_train = np.array([np.array(x) for x in X_train_10k])
    X_test = np.array([np.array(x) for x in X_test_10k])
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)

    y_train = np.array([y.index(1) for y in y_train_10k])
    y_train = y_train.astype(np.int64)
    y_test = np.array([y.index(1) for y in y_test_10k])
    y_test = y_test.astype(np.int64)

    train_dataset = NumeralDataset(X_train, y_train)
    test_dataset = NumeralDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)

    train_dl = DeviceDataLoader(train_loader, device)
    valid_dl = DeviceDataLoader(test_loader, device)

    model = LSTM(768, 1536)
    model.cuda()
    model.to(device)

    losses = fit(150, 1e-4, model, train_dl, valid_dl)

    model_eval = LSTM(768, 1536)

    model_eval.load_state_dict(torch.load('./C1.pt'))
    model_eval.to(device)

    print("Range [1000,10000]")
    print(accuracy(model, valid_dl))
    print()

    X_train_B10k, y_train_B10k = get_embeddings_list(train_B10k)
    X_test_B10k, y_test_B10k = get_embeddings_list(test_B10k)

    X_train = np.array([np.array(x) for x in X_train_B10k])
    X_test = np.array([np.array(x) for x in X_test_B10k])
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)

    y_train = np.array([y.index(1) for y in y_train_B10k])
    y_train = y_train.astype(np.int64)
    y_test = np.array([y.index(1) for y in y_test_B10k])
    y_test = y_test.astype(np.int64)

    train_dataset = NumeralDataset(X_train, y_train)
    test_dataset = NumeralDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)

    train_dl = DeviceDataLoader(train_loader, device)
    valid_dl = DeviceDataLoader(test_loader, device)

    model = LSTM(768, 1536)
    model.to(device)

    losses = fit(150, 1e-4, model, train_dl, valid_dl)

    model_eval = LSTM(768, 1536)

    model_eval.load_state_dict(torch.load('./C1.pt'))
    model_eval.to(device)

    print("Range B10k")
    print(accuracy(model, valid_dl))
    print()

    X_train, y_train = get_embeddings_list(train)
    X_test, y_test = get_embeddings_list(test)

    X_train = np.array([np.array(x) for x in X_train])
    X_test = np.array([np.array(x) for x in X_test])
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)

    y_train = np.array([y.index(1) for y in y_train])
    y_train = y_train.astype(np.int64)
    y_test = np.array([y.index(1) for y in y_test])
    y_test = y_test.astype(np.int64)

    train_dataset = NumeralDataset(X_train, y_train)
    test_dataset = NumeralDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)

    train_dl = DeviceDataLoader(train_loader, device)
    valid_dl = DeviceDataLoader(test_loader, device)

    model = LSTM(768, 1536)
    model.to(device)

    losses = fit(150, 1e-4, model, train_dl, valid_dl)

    model_eval = LSTM(768, 1536)

    model_eval.load_state_dict(torch.load('./C1.pt'))
    model_eval.to(device)

    print("Full Dataset")
    print(accuracy(model, valid_dl))
    print()

if name == '__main__':
    main()