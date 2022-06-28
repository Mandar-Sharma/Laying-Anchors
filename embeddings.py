import math
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from random import sample
from transformers import AutoTokenizer, AutoModel
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