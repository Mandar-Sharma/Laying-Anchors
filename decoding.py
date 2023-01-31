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

train, test = [], []
train.extend(train_100)
train.extend(train_1000)
train.extend(in_train_10k)
train.extend(in_train_B10k)

test.extend(test_100)
test.extend(test_1000)
test.extend(in_test_10k)
test.extend(in_test_B10k)

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

device = torch.device('cuda:0')
tokenizer = BertTokenizerFast.from_pretrained(config['model'])
model = BertForMaskedLM.from_pretrained(config['model'], output_hidden_states=True)
model.to(device)

def get_embeddings(numeral_list):    
    
    X = []
    y = []

    for i in tqdm(numeral_list):
        if config['model_type'] == 'anc':
            anc = find_anc(i)
            input_str = str(i) + " <ANC> " + str(anc)
        elif config['model_type'] == 'lr_anc':
            anc = find_anc(i)
            if (i - anc) > 0:
                input_str = str(i) + ' <LA> ' + str(anc)
            else:
                input_str = str(i) + ' <RA> ' + str(anc)
        elif config['model_type'] == 'log_anc':
            anc = find_anc_log(i)
            input_str = str(i) + " <ANC> " + str(anc)
        elif config['model_type'] == 'lr_log_anc':
            anc = find_anc_log(i)
            if (log_squash_2(i) - anc) > 0:
                input_str = str(i) + ' <LA> ' + str(anc)
            else:
                input_str = str(i) + ' <RA> ' + str(anc)
        else:
            input_str = str(i)
        input_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0)
        outputs = model(input_ids.to(device))
        last_four = torch.stack(outputs['hidden_states'][-4:]).sum(0)
        last_four = last_four[0][1:-1].mean(dim=0).detach().cpu()
        X.append(last_four)
        y.append(i)
        
        del input_ids
        del outputs
        del last_four
        torch.cuda.empty_cache()
    
    print(len(y))
    print(len(X))
        
    return X, y

def main():
    #The global variables defined cover In-domain and Out-domain numerals from 1 to 10^10
    
    #In-domain
    X_train_100, y_train_100 = get_embeddings(train_100)
    X_test_100, y_test_100 = get_embeddings(test_100)

    X_train_100 = np.array([x.numpy() for x in X_train_100])
    X_test_100 = np.array([x.numpy() for x in X_test_100])

    y_train_100 = log_scaler.fit_transform(y_train_100)
    y_test_100 = log_scaler.transform(y_test_100)

    xbg_model.fit(X_train_100, y_train_100)

    y_pred_100 = xbg_model.predict(X_test_100)
    print("Range - [0,100]")
    print(mean_squared_error(y_test_100, y_pred_100, squared=False))
    print(np.sqrt(mean_squared_log_error(y_test_100, y_pred_100)))
        
    X_train_1000, y_train_1000 = get_embeddings(train_1000)
    X_test_1000, y_test_1000 = get_embeddings(test_1000)

    X_train_1000 = np.array([x.numpy() for x in X_train_1000])
    X_test_1000 = np.array([x.numpy() for x in X_test_1000])

    y_train_1000 = log_scaler.fit_transform(y_train_1000)
    y_test_1000 = log_scaler.transform(y_test_1000)

    xbg_model.fit(X_train_1000, y_train_1000)

    y_pred_1000 = xbg_model.predict(X_test_1000)
    print("Range - [100,1000]")
    print(mean_squared_error(y_test_1000, y_pred_1000, squared=False))
    print(np.sqrt(mean_squared_log_error(y_test_1000, y_pred_1000)))
    print()

    X_train_10000, y_train_10000 = get_embeddings(in_train_10k)
    X_test_10000, y_test_10000 = get_embeddings(in_test_10k)

    X_train_10000 = np.array([x.numpy() for x in X_train_10000])
    X_test_10000 = np.array([x.numpy() for x in X_test_10000])

    y_train_10000 = log_scaler.fit_transform(y_train_10000)
    y_test_10000 = log_scaler.transform(y_test_10000)

    xbg_model.fit(X_train_10000, y_train_10000)

    y_pred_10000 = xbg_model.predict(X_test_10000)
    print("Range - [1000,10k]")
    print(mean_squared_error(y_test_10000, y_pred_10000, squared=False))
    print(np.sqrt(mean_squared_log_error(y_test_10000, y_pred_10000)))
    print()

    X_train_B10k, y_train_B10k = get_embeddings(in_train_B10k)
    X_test_B10k, y_test_B10k = get_embeddings(in_test_B10k)

    X_train_B10k = np.array([x.numpy() for x in X_train_B10k])
    X_test_B10k = np.array([x.numpy() for x in X_test_B10k])

    y_train_B10k = log_scaler.fit_transform(y_train_B10k)
    y_test_B10k = log_scaler.transform(y_test_B10k)

    xbg_model.fit(X_train_B10k, y_train_B10k)
    print("Beyond 10k")
    y_pred_B10k = xbg_model.predict(X_test_B10k)
    print(mean_squared_error(y_test_B10k, y_pred_B10k, squared=False))
    print(np.sqrt(mean_squared_log_error(y_test_B10k, y_pred_B10k)))
    print()

    X_train, y_train = get_embeddings(train)
    X_test, y_test = get_embeddings(test)

    X_train = np.array([x.numpy() for x in X_train])
    X_test = np.array([x.numpy() for x in X_test])

    y_train = log_scaler.fit_transform(y_train)
    y_test = log_scaler.transform(y_test)

    xbg_model.fit(X_train, y_train)

    y_pred = xbg_model.predict(X_test)
    print("Entire Corpus")
    print(mean_squared_error(y_test, y_pred, squared=False))
    print(np.sqrt(mean_squared_log_error(y_test, y_pred)))
    print()

    #Out-of-domain

    X_train_10000, y_train_10000 = get_embeddings(out_train_10k)
    X_test_10000, y_test_10000 = get_embeddings(out_test_10k)

    X_train_10000 = np.array([x.numpy() for x in X_train_10000])
    X_test_10000 = np.array([x.numpy() for x in X_test_10000])

    y_train_10000 = log_scaler.fit_transform(y_train_10000)
    y_test_10000 = log_scaler.transform(y_test_10000)

    xbg_model.fit(X_train_10000, y_train_10000)

    y_pred_10000 = xbg_model.predict(X_test_10000)
    print("Range - [1000,10k]")
    print(mean_squared_error(y_test_10000, y_pred_10000, squared=False))
    print(np.sqrt(mean_squared_log_error(y_test_10000, y_pred_10000)))
    print()

    X_train_B10k, y_train_B10k = get_embeddings(out_train_B10k)
    X_test_B10k, y_test_B10k = get_embeddings(out_test_B10k)

    X_train_B10k = np.array([x.numpy() for x in X_train_B10k])
    X_test_B10k = np.array([x.numpy() for x in X_test_B10k])

    y_train_B10k = log_scaler.fit_transform(y_train_B10k)
    y_test_B10k = log_scaler.transform(y_test_B10k)

    xbg_model.fit(X_train_B10k, y_train_B10k)
    print("Beyond 10k")
    y_pred_B10k = xbg_model.predict(X_test_B10k)
    print(mean_squared_error(y_test_B10k, y_pred_B10k, squared=False))
    print(np.sqrt(mean_squared_log_error(y_test_B10k, y_pred_B10k)))
    print()

if name == '__main__':
    main()