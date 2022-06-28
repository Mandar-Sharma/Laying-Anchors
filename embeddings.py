import math
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from random import sample
from decimal import Decimal
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast

with open('./WikiText103/nums', 'rb') as fp:
    numerals = pickle.load(fp)
with open('./WikiText103/means', 'rb') as fp:
    means = pickle.load(fp)
with open('./WikiText103/log_means', 'rb') as fp:
    log_means = pickle.load(fp)

means = list(set(sorted([round(x) for x in means.flatten()])))
log_means = list(set(sorted([round(x) for x in log_means.flatten()])))

numerals = [int(x) for x in numerals if round(math.log(int(x), 10)) < 10]
numerals = sorted(list(set(numerals)))

test_100 = sorted(random.sample(numerals[:100], 20))
train_100 = list(set(numerals[:100]) - set(test_100))

print("test_100 : {} and train_100 : {} samples.".format(len(test_100), len(train_100)))

test_1000 = sorted(random.sample(numerals[100:1000], 180))
train_1000 = list(set(numerals[100:1000]) - set(test_1000))

print("test_1000 : {} and train_1000 : {} samples.".format(len(test_1000), len(train_1000)))

#1000 to 10k
in_10000, out_10000 = [], []
for i in list(range(1000,10001)):
    if i in numerals:
        in_10000.append(i)
    else:
        out_10000.append(i)
out_10000 = random.sample(out_10000, 3854)

print("in_10000 : {} and out_10000 : {} samples.".format(len(in_10000), len(out_10000)))

in_train_10k, out_train_10k, in_test_10k, out_test_10k = [], [], [], []
in_test_10k = sorted(random.sample(in_10000, 770))
in_train_10k = list(set(in_10000) - set(in_test_10k))
out_test_10k = sorted(random.sample(out_10000, 770))
out_train_10k = list(set(out_10000) - set(out_test_10k))

print("in_train_10k : {}, in_test_10k : {}, out_train_10k : {}, out_test_10k : {} samples.".format(len(in_train_10k),len(in_test_10k),len(out_train_10k), len(out_test_10k)))

#Beyond 10k
in_B10k = numerals[4852:]
out_B10k = []

flag = True
for i in tqdm(in_B10k):
    while flag:
        op = random.choice(['+','-'])
        bias = random.randint(1, 100)
        j = i + bias if op == '+' else i - bias
        if j not in in_B10k:
            out_B10k.append(j)
            flag = False
    flag = True

in_train_B10k, out_train_B10k, in_test_B10k, out_test_B10k = [], [], [], []
in_test_B10k = sorted(random.sample(in_B10k, 872))
in_train_B10k = list(set(in_B10k) - set(in_test_B10k)) 
out_test_B10k = sorted(random.sample(out_B10k, 872))
out_train_B10k = list(set(out_B10k) - set(out_test_B10k))

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def log_squash(num):
    if num > 1:
        return round((np.log(num) + 1) * 10)
    elif num < -1:
        return round((-np.log(-num) -1) * 10)
    else:
        return round(num * 10)

def find_anc(i):
    anchor = min(means, key=lambda x:abs(x-int(i)))
    return str(anchor)

def find_anc_log(i):
    anchor = min(log_means, key=lambda x:abs(x-log_squash(int(i))))
    return str(anchor)

def get_embeddings_base(tokenizer_path, model_path, numbers_list, save_path)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(model_path, output_hidden_states=True)

    X = []
    y = []

    for i in tqdm(numbers_list):
        input_ids = torch.tensor(tokenizer.encode(str(i))).unsqueeze(0)
        outputs = model(input_ids)
        last_four = torch.stack(outputs['hidden_states'][-4:]).sum(0)
        X.append(last_four[0][1:-1].mean(dim=0))
        y.append(i)

    with open(save_path + /'X', 'wb') as fp:
        pickle.dump(X, fp)
    with open(save_path + /'y', 'wb') as fp:
        pickle.dump(y, fp)

def get_embeddings_exp(tokenizer_path, model_path, numbers_list, save_path, cues=False)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(model_path, output_hidden_states=True)

    if cues == False:
        X = []
        y = []

        for i in tqdm(numbers_list):
            input_ids = torch.tensor(tokenizer.encode(str(i))).unsqueeze(0)
            outputs = model(input_ids)
            last_four = torch.stack(outputs['hidden_states'][-4:]).sum(0)
            X.append(last_four[0][1:-1].mean(dim=0))
            y.append(i)

        with open(save_path + /'X', 'wb') as fp:
            pickle.dump(X, fp)
        with open(save_path + /'y', 'wb') as fp:
            pickle.dump(y, fp)
    
    elif cues == True:
        X = []
        y = []

        for i in tqdm(numbers_list):
            input_str = str(i) + " <EXP> " + str(fexp(i))
            input_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0)
            outputs = model(input_ids)
            last_four = torch.stack(outputs['hidden_states'][-4:]).sum(0)
            X.append(last_four[0][1:-1].mean(dim=0))
            y.append(i)

        with open(save_path + /'X', 'wb') as fp:
            pickle.dump(X, fp)
        with open(save_path + /'y', 'wb') as fp:
            pickle.dump(y, fp)


def get_embeddings_anc(tokenizer_path, model_path, numbers_list, save_path, cues=False)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(model_path, output_hidden_states=True)

    if cues == False:
        X = []
        y = []

        for i in tqdm(numbers_list):
            input_ids = torch.tensor(tokenizer.encode(str(i))).unsqueeze(0)
            outputs = model(input_ids)
            last_four = torch.stack(outputs['hidden_states'][-4:]).sum(0)
            X.append(last_four[0][1:-1].mean(dim=0))
            y.append(i)

        with open(save_path + /'X', 'wb') as fp:
            pickle.dump(X, fp)
        with open(save_path + /'y', 'wb') as fp:
            pickle.dump(y, fp)
    
    elif cues == True:
        X = []
        y = []

        for i in tqdm(numbers_list):
            anc = find_anc(i)
            input_str = str(i) + " <ANC> " + str(anc)
            input_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0)
            outputs = model(input_ids)
            last_four = torch.stack(outputs['hidden_states'][-4:]).sum(0)
            X.append(last_four[0][1:-1].mean(dim=0))
            y.append(i)

        with open(save_path + /'X', 'wb') as fp:
            pickle.dump(X, fp)
        with open(save_path + /'y', 'wb') as fp:
            pickle.dump(y, fp)

def get_embeddings_lr_anc(tokenizer_path, model_path, numbers_list, save_path, cues=False)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(model_path, output_hidden_states=True)

    if cues == False:
        X = []
        y = []

        for i in tqdm(numbers_list):
            input_ids = torch.tensor(tokenizer.encode(str(i))).unsqueeze(0)
            outputs = model(input_ids)
            last_four = torch.stack(outputs['hidden_states'][-4:]).sum(0)
            X.append(last_four[0][1:-1].mean(dim=0))
            y.append(i)

        with open(save_path + /'X', 'wb') as fp:
            pickle.dump(X, fp)
        with open(save_path + /'y', 'wb') as fp:
            pickle.dump(y, fp)
    
    elif cues == True:
        X = []
        y = []

        for i in tqdm(numbers_list):
            anc = find_anc(i)
            if (i - anc) > 0:
                input_str = str(i) + ' <LA> ' + str(anc)
            else:
                input_str = str(i) + ' <RA> ' + str(anc)
            input_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0)
            outputs = model(input_ids)
            last_four = torch.stack(outputs['hidden_states'][-4:]).sum(0)
            X.append(last_four[0][1:-1].mean(dim=0))
            y.append(i)

        with open(save_path + /'X', 'wb') as fp:
            pickle.dump(X, fp)
        with open(save_path + /'y', 'wb') as fp:
            pickle.dump(y, fp)

def get_embeddings_loganc(tokenizer_path, model_path, numbers_list, save_path, cues=False)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(model_path, output_hidden_states=True)

    if cues == False:
        X = []
        y = []

        for i in tqdm(numbers_list):
            input_ids = torch.tensor(tokenizer.encode(str(i))).unsqueeze(0)
            outputs = model(input_ids)
            last_four = torch.stack(outputs['hidden_states'][-4:]).sum(0)
            X.append(last_four[0][1:-1].mean(dim=0))
            y.append(i)

        with open(save_path + /'X', 'wb') as fp:
            pickle.dump(X, fp)
        with open(save_path + /'y', 'wb') as fp:
            pickle.dump(y, fp)
    
    elif cues == True:
        X = []
        y = []

        for i in tqdm(numbers_list):
            anc = find_anc_log(i)
            input_str = str(i) + " <ANC> " + str(anc)
            input_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0)
            outputs = model(input_ids)
            last_four = torch.stack(outputs['hidden_states'][-4:]).sum(0)
            X.append(last_four[0][1:-1].mean(dim=0))
            y.append(i)

        with open(save_path + /'X', 'wb') as fp:
            pickle.dump(X, fp)
        with open(save_path + /'y', 'wb') as fp:
            pickle.dump(y, fp)

def get_embeddings_lr_loganc(tokenizer_path, model_path, numbers_list, save_path, cues=False)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(model_path, output_hidden_states=True)

    if cues == False:
        X = []
        y = []

        for i in tqdm(numbers_list):
            input_ids = torch.tensor(tokenizer.encode(str(i))).unsqueeze(0)
            outputs = model(input_ids)
            last_four = torch.stack(outputs['hidden_states'][-4:]).sum(0)
            X.append(last_four[0][1:-1].mean(dim=0))
            y.append(i)

        with open(save_path + /'X', 'wb') as fp:
            pickle.dump(X, fp)
        with open(save_path + /'y', 'wb') as fp:
            pickle.dump(y, fp)
    
    elif cues == True:
        X = []
        y = []

        for i in tqdm(numbers_list):
            anc = find_anc_log(i)
            if (log_squash(i) - anc) > 0:
                input_str = str(i) + ' <LA> ' + str(anc)
            else:
                input_str = str(i) + ' <RA> ' + str(anc)
            input_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0)
            outputs = model(input_ids)
            last_four = torch.stack(outputs['hidden_states'][-4:]).sum(0)
            X.append(last_four[0][1:-1].mean(dim=0))
            y.append(i)

        with open(save_path + /'X', 'wb') as fp:
            pickle.dump(X, fp)
        with open(save_path + /'y', 'wb') as fp:
            pickle.dump(y, fp)


def main():
    #Call the function for the respective model embeddings you'd like
    #The global variables defined cover In-domain and Out-domain numerals from 1 to 10^10
    pass

if name == '__main__':
    main()