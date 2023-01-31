import argparse
 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--base_tokenizer", help="Tokenizer Folder")
parser.add_argument("-b", "--base_model", help="Base Model Folder")
parser.add_argument("-m", "--model", help="Model Folder")
parser.add_argument("-type", "--model_type", help="Model Type")
parser.add_argument("-d", "--data", help="Training Dataset Directory")
args = parser.parse_args()
config = vars(args)

import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

tokenizer = BertTokenizerFast.from_pretrained(config['base_tokenizer'])
model = BertForMaskedLM.from_pretrained(config['base_model'])

if config['model_type'] == 'anc' or config['model_type'] == 'log_anc':
    special_tokens_dict = {'additional_special_tokens': ['<ANC>']}
elif config['model_type'] == 'lr_anc' or config['model_type'] == 'lr_log_anc':
    special_tokens_dict = {'additional_special_tokens': ['<LA>','<RA>']}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

dataset = load_dataset(config['data'])

def encode_with_truncation(example):
    sample_masked = example['text'].split()
    for idx, val in enumerate(sample_masked):
        if val in special_tokens_dict['additional_special_tokens']:
            len_masks = len(tokenizer(tokenizer.tokenize(sample_masked[idx+1]),is_split_into_words =True)['input_ids'][1:-1])
            sample_masked[idx + 1] = '[MASK]' * len_masks
    return tokenizer(' '.join(sample_masked), padding='max_length', truncation=True, max_length=512, \
                     return_special_tokens_mask=True)

#Tokenizing the train dataset
train_dataset = dataset['train'].map(encode_with_truncation)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'special_tokens_mask'])

tokenizer.save_pretrained(config['model'])

label_col = []
for idx, val in tqdm(enumerate(dataset['train'])):
    label_ids = []
    sample_split = val['text'].split()
    for idx_, value in enumerate(sample_split):
        if value in special_tokens_dict['additional_special_tokens']:
            label_ids.append(tokenizer(tokenizer.tokenize(sample_split[idx_+1]),is_split_into_words =True)['input_ids'][1:-1])
    label_ids = [item for sublist in label_ids for item in sublist]
    mask_idx = (train_dataset[idx]['input_ids'] == 103).nonzero(as_tuple=True)[0]
    labels = [-100] * 512
    for i, v in enumerate(label_ids):
        if i < len(mask_idx):
            labels[mask_idx[i]] = v
    label_col.append(labels)

train_dataset = train_dataset.add_column('labels', label_col)

training_args = TrainingArguments(
    output_dir=config['model'],
    overwrite_output_dir=True,      
    num_train_epochs=6,           
    per_device_train_batch_size=8,
    logging_steps=500,           
    save_steps=1000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model(config['model'])