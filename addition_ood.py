import argparse
 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-type", "--model_type", help="Model Type")
parser.add_argument("-m", "--model", help="Model Folder")

args = parser.parse_args()
config = vars(args)

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

numerals = [int(x) for x in numerals if round(math.log(int(x), 10)) < 10]
numerals = sorted(list(set(numerals)))

#1000 to 10k
_10000 = []
for i in list(range(1000,10001)):
    if i not in numerals:
        _10000.append(i)

seen_set_10000 = []
train_10k = []
for i in tqdm(range(800)):
    sam = random.sample(_10000, 2)
    while sam in seen_set_10000:
        sam = random.sample(_10000, 2)
    seen_set_10000.append(sam)
    sam.append(sum(sam))
    train_10k.append(sam)
    
test_10k = []
for i in tqdm(range(200)):
    sam = random.sample(_10000, 2)
    while sam in seen_set_10000:
        sam = random.sample(_10000, 2)
    seen_set_10000.append(sam)
    sam.append(sum(sam))
    test_10k.append(sam)

train_B10k, test_B10k = [], []
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

seen_set_B10k= []
for i in tqdm(range(800)):
    sam = random.sample(out_B10k, 2)
    while sam in seen_set_B10k:
        sam = random.sample(out_B10k, 2)
    seen_set_B10k.append(sam)
    sam.append(sum(sam))
    train_B10k.append(sam)
    
for i in tqdm(range(200)):
    sam = random.sample(out_B10k, 2)
    while sam in seen_set_B10k:
        sam = random.sample(out_B10k, 2)
    seen_set_B10k.append(sam)
    sam.append(sum(sam))
    test_B10k.append(sam)

from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error, mean_squared_log_error

xbg_model = XGBRegressor(n_estimators = 1000, max_depth = 5, learning_rate = 0.01, gamma=0, tree_method='gpu_hist', gpu_id=0)

def log_squash(nums):
    return_list = []
    for num in nums:
        if num > 1:
            return_list.append(np.log(num) + 1)
        elif num < -1:
            return_list.append(-np.log(-num) -1)
        else:
            return_list.append(num)
    return return_list

log_scaler = FunctionTransformer(log_squash)

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

tokenizer = BertTokenizerFast.from_pretrained(config['model'])
model = BertForMaskedLM.from_pretrained(config['model'], output_hidden_states=True)

device = torch.device('cuda:0')
model.to(device)

def get_embeddings_add(numbers_list):
    X = []
    y = []

    for i in tqdm(numbers_list):
        X_sub = []
        for j in i[:-1]:
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
            outputs = model(input_ids.to(device))
            last_four = torch.stack(outputs['hidden_states'][-4:]).sum(0)
            emb = last_four[0][1:-1].mean(dim=0)
            X_sub.append(emb.detach().cpu().numpy())
            
            del input_ids
            del outputs
            del last_four
            torch.cuda.empty_cache()
            
        X.append(np.concatenate((X_sub[0], X_sub[1])))
        y.append(i[-1])
    return X,y


def main():
    X_train_10000, y_train_10000 = get_embeddings_add(train_10k)
    X_test_10000, y_test_10000 = get_embeddings_add(test_10k)

    X_train_10000 = np.array([x for x in X_train_10000])
    X_test_10000 = np.array([x for x in X_test_10000])

    y_train_10000 = log_scaler.fit_transform(y_train_10000)
    y_test_10000 = log_scaler.transform(y_test_10000)

    xbg_model.fit(X_train_10000, y_train_10000)

    y_pred_10000 = xbg_model.predict(X_test_10000)
    print("OOD Range [1k,10k]")
    print(mean_squared_error(y_test_10000, y_pred_10000, squared=False))
    print(np.sqrt(mean_squared_log_error(y_test_10000, y_pred_10000)))

    X_train_B10k, y_train_B10k = get_embeddings_add(train_B10k)
    X_test_B10k, y_test_B10k = get_embeddings_add(test_B10k)

    X_train_B10k = np.array([x for x in X_train_B10k])
    X_test_B10k = np.array([x for x in X_test_B10k])

    y_train_B10k = log_scaler.fit_transform(y_train_B10k)
    y_test_B10k = log_scaler.transform(y_test_B10k)

    xbg_model.fit(X_train_B10k, y_train_B10k)

    y_pred_B10k = xbg_model.predict(X_test_B10k)
    print("OOD Range [10k,10^10]")
    print(mean_squared_error(y_test_B10k, y_pred_B10k, squared=False))
    print(np.sqrt(mean_squared_log_error(y_test_B10k, y_pred_B10k)))


if name == '__main__':
    main()