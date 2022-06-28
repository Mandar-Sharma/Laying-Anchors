import re
import json
import glob
import math
import random
import pickle
import numpy as np
from tqdm import tqdm
from decimal import Decimal

with open('./WikiText103/means', 'rb') as fp:
    means = pickle.load(fp)
with open('./WikiText103/log_means', 'rb') as fp:
    log_means = pickle.load(fp)

means = list(set(sorted([round(x) for x in means.flatten()])))
log_means = list(set(sorted([round(x) for x in log_means.flatten()])))

max_val = max(log_means)

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def log_squash(num):
    if num > 1:
        try:
            return round((np.log(num) + 1) * 10)
        except TypeError:
            return round(max_val)
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

files = sorted(glob.glob("./raw/*"))

#Training corpus for <ANC> model
corpus = []
for file in files:
    print("\n Working: ", file)
    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.read().split('\n')
    for line in tqdm(lines):
        if line != ' ':
            nums_list =  re.findall(r'\b\d+\b', line.replace(',', ''))
            if nums_list != []:
                line_copy = line[:]
                nums = sorted(list(set(nums_list)), key=len)
                anchors = [find_anc(num) for num in nums]
                nums_sorted = []
                for anchor in anchors:
                    if anchor in nums:
                        nums_sorted.append(anchor)
                nums_sorted = list(set(nums_sorted))
                for num in nums:
                    if num not in nums_sorted:
                        nums_sorted.append(num)           
                for nums in nums_sorted:
                    anchor = min(means, key=lambda x:abs(x-int(nums)))   
                    line_copy = re.sub(r"\b%s\b" % nums , str(int(nums)) + " <ANC> " + str(anchor), line_copy)
                corpus.append(line_copy)
            else:
                corpus.append(line)

file_dir = './WikiText103/Training/Anc/train_corpus.txt'
with open(file_dir, 'w') as f:
    for item in corpus:
        f.write("%s\n" % item)

# For <LA> <RA> model
corpus = []    
for file in files:
    print("\n Working: ", file)
    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.read().split('\n')
    for line in tqdm(lines):
        if line != ' ':
            nums_list =  re.findall(r'\b\d+\b', line.replace(',', ''))
            if nums_list != []:
                line_copy = line[:]
                nums = sorted(list(set(nums_list)), key=len)
                anchors = [find_anc(num) for num in nums]
                nums_sorted = []
                for anchor in anchors:
                    if anchor in nums:
                        nums_sorted.append(anchor)
                nums_sorted = list(set(nums_sorted))
                for num in nums:
                    if num not in nums_sorted:
                        nums_sorted.append(num)           
                for nums in nums_sorted:
                    anchor = min(means, key=lambda x:abs(x-int(nums)))   
                    if (int(nums) - anchor) > 0:
                        line_copy = re.sub(r"\b%s\b" % nums , str(int(nums)) + " <LA> " + str(anchor), line_copy)
                    else:
                        line_copy = re.sub(r"\b%s\b" % nums , str(int(nums)) + " <RA> " + str(anchor), line_copy)
                corpus.append(line_copy)
            else:
                corpus.append(line)

file_dir = './WikiText103/Training/LR_Anc/train_corpus.txt'
with open(file_dir, 'w') as f:
    for item in corpus:
        f.write("%s\n" % item)

# For Log <ANC>
corpus = []
for file in files:
    print("\n Working: ", file)
    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.read().split('\n')
    for line in tqdm(lines):
        if line != ' ':
            nums_list =  re.findall(r'\b\d+\b', line.replace(',', ''))
            if nums_list != []:
                line_copy = line[:]
                nums = sorted(list(set(nums_list)), key=len)
                anchors = [find_anc_log(num) for num in nums]
                nums_sorted = []
                for anchor in anchors:
                    if anchor in nums:
                        nums_sorted.append(anchor)
                nums_sorted = list(set(nums_sorted))
                for num in nums:
                    if num not in nums_sorted:
                        nums_sorted.append(num)           
                for nums in nums_sorted:
                    anchor = min(means_log, key=lambda x:abs(x-log_squash(int(nums))))   
                    line_copy = re.sub(r"\b%s\b" % nums , str(int(nums)) + " <ANC> " + str(anchor), line_copy)
                corpus.append(line_copy)
            else:
                corpus.append(line)

file_dir = './WikiText103/Training/LogAnc/train_corpus.txt'
with open(file_dir, 'w') as f:
    for item in corpus:
        f.write("%s\n" % item)

corpus = []

for file in files:
    print("\n Working: ", file)
    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.read().split('\n')
    for line in tqdm(lines):
        if line != ' ':
            nums_list =  re.findall(r'\b\d+\b', line.replace(',', ''))
            if nums_list != []:
                line_copy = line[:]
                nums = sorted(list(set(nums_list)), key=len)
                anchors = [find_anc_log(num) for num in nums]
                nums_sorted = []
                for anchor in anchors:
                    if anchor in nums:
                        nums_sorted.append(anchor)
                nums_sorted = list(set(nums_sorted))
                for num in nums:
                    if num not in nums_sorted:
                        nums_sorted.append(num)           
                for nums in nums_sorted:
                    anchor = find_anc_log(nums)   
                    if (log_squash(int(nums)) - int(anchor)) > 0:
                        line_copy = re.sub(r"\b%s\b" % nums , nums + " <LA> " + str(anchor), line_copy)
                    else:
                        line_copy = re.sub(r"\b%s\b" % nums , nums + " <RA> " + str(anchor), line_copy)
                corpus.append(line_copy)

file_dir = './WikiText103/Training/LR_LogAnc/train_corpus.txt'
with open(file_dir, 'w') as f:
    for item in corpus:
        f.write("%s\n" % item)

#For <EXP> model

corpus = []

for file in files:
    print("\n Working: ", file)
    with open(file, 'r', encoding='utf-8') as fp:
        lines = fp.read().split('\n')
    for line in tqdm(lines):
        if line != ' ':
            nums_list =  re.findall(r'\b\d+\b', line.replace(',', ''))
            if nums_list != []:
                line_copy = line[:]
                nums = sorted(list(set(nums_list)), key=len)      
                for num in nums:
                    line_copy = re.sub(r"\b%s\b" % num , num + " <EXP> " + str(fexp(int(num))), line_copy)
                corpus.append(line_copy)

file_dir = './WikiText103/Training/EXP/train_corpus.txt'
with open(file_dir, 'w') as f:
    for item in corpus:
        f.write("%s\n" % item)