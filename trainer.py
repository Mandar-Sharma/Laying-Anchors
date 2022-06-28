import argparse
 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--tokenizer", help="Tokenizer Folder")
parser.add_argument("-b", "--base_model", help="Base Model Folder")
parser.add_argument("-m", "--model", help="Model Folder")
parser.add_argument("-type", "--model_type", help="Model Type")
parser.add_argument("-d", "--data", help="Training Dataset Directory")
args = parser.parse_args()
config = vars(args)

import torch
import pickle
from datasets import load_dataset
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

tokenizer = BertTokenizerFast.from_pretrained(config['tokenizer'])
model_config = BertConfig(vocab_size=30522, hidden_size = 264, num_hidden_layers = 4, intermediate_size = 1056, max_position_embeddings=512)
model = BertForMaskedLM.from_pretrained(config['base_model'], config=model_config)

if config['model_type'] == 'anc' or config['model_type'] == 'loganc':
    special_tokens_dict = {'additional_special_tokens': ['<ANC>']}
elif config['model_type'] == 'lr_anc' or config['model_type'] == 'lr_loganc':
    special_tokens_dict = {'additional_special_tokens': ['<LA>','<RA>']}
elif config['model_type'] == 'exp':
    special_tokens_dict = {'additional_special_tokens': ['<EXP>']}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

dataset = load_dataset(config['data'])

def encode_with_truncation(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_special_tokens_mask=True)

# tokenizing the train dataset
train_dataset = dataset['train'].map(encode_with_truncation, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", 'token_type_ids', 'attention_mask', 'special_tokens_mask'])

tokenizer.save_pretrained(config['model'])

device = torch.device("cuda")
model.cuda()
model = model.to(device)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=config['model'],          
    overwrite_output_dir=True,      
    num_train_epochs=5,            
    per_device_train_batch_size=16, 
    gradient_accumulation_steps=8,  
    logging_steps=500,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()