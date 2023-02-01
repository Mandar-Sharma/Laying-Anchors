# Laying Anchors
This repository anonymously hosts the codebase, datasets, and models for the ACL 23 submission "Laying Anchors: Semantically Priming Numerals in Language Modeling" 

> **Note:** As the saved PyTorch models are larger in size, [Git LFS](https://git-lfs.github.com/) was used to push these models to Git. Please make sure you have it installed in your system.


Before using this repository, please make sure you have the following packages installed in your environment:
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Tqdm](https://github.com/tqdm/tqdm)
- [Sklearn](https://scikit-learn.org/)
- [Xgboost](https://xgboost.readthedocs.io/)
- [PyTorch](https://pytorch.org/)
- [Huggingface Transformers & Dataset](https://huggingface.co/)

> **Note:** The versions of these packages are not specified as PyTorch (and subsequently Huggingface) versioning is highly dependent on the user GPU and CUDA version.

Please follow the step-wise description below to replicate the results published in "Laying Anchors: Semantically Priming Numerals in Language Modeling". If you decide to change the directory structures, please make sure to make corresponding changes in the code as well. If left undisturbed, the current directory structure allows for execution of the scripts without change.

## Step I: WikiText-103 and Preprocessing

First, please download the [WikiText-103 word-level corpus](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) and place the contents (text files) inside the directory /WikiText103/raw.
```
mkdir WikiText103
mkdir WikiText103/raw
```

Run the preprocessing script:
```
python preprocess.py
```

This will populate the WikiText103 directory with pickle-ized objects:
- nums: The list of numerals extracted from WikiText-103
- means: The means of the Gaussians components extracted from nums
- log_means: The log-normalized means of the Gaussians components extracted from nums

> **Note:** To alter the number of gaussian components, please modify the **gmm** function inside preprocess.py

## Step II: Training Models

> **Note:** The pre-trained version of all the models used in the paper are readily available for out-of-the-box use in this repository, thus you can skip to Step III if you wish to skip re-training the models from scratch. 

First run the build_corpus script:
```
python build_corpus.py
```
This will populate the WikiText103/Training directory with model-specific training corpus. Using these corpus, we can now train our models.
```
python trainer.py -t bert-base-uncased -b bert-base-uncased -m "/path/to/save-model" -type {"anc", "log_anc", "lr_anc", or "lr_log_anc"} -d "path/to/training-corpus" 
```

## Step III: Decoding, Addition, and List Min/Max Tasks

There are 5 scripts for generating model embeddings as per the tasks described in the paper, wherein either XGBoost-based regressors or LSTM-based classifiers are employed to compare model performance across tasks.

- decoding.py: For the decoding task, both the in-domain and the out-of-domain numeral ranges are defined as global variables.
```
python decoding.py -m ./Models/Anc -t anc
python decoding.py -m ./Models/LR\ Anc -t lr_anc
python decoding.py -m ./Models/Log\ Anc -t log_anc
python decoding.py -m ./Models/LR\ Log\ Anc -t lr_log_anc
```
- addition_indomain.py: Performance on in-domain numerals for the addition task.
```
python addition_indomain.py -m ./Models/Anc -t anc
python addition_indomain.py -m ./Models/LR\ Anc -t lr_anc
python addition_indomain.py -m ./Models/Log\ Anc -t log_anc
python addition_indomain.py -m ./Models/LR\ Log\ Anc -t lr_log_anc
```
- addition_ood.py: Performance on out-of-domain numerals for the addition task.
```
python addition_ood.py -m ./Models/Anc -t anc
python addition_ood.py -m ./Models/LR\ Anc -t lr_anc
python addition_ood.py -m ./Models/Log\ Anc -t log_anc
python addition_ood.py -m ./Models/LR\ Log\ Anc -t lr_log_anc
```
- list_indomain.py: Performance on in-domain numerals for the list max/min task, same as before, but with an additional {MIN, MAX} mode flag (-mode) for list min/max task.
```
python list_indomain.py -m ./Models/Anc -t anc -mode MIN
python list_indomain.py -m ./Models/LR\ Anc -t lr_anc -mode MIN
python list_indomain.py -m ./Models/Log\ Anc -t log_anc -mode MIN
python list_indomain.py -m ./Models/LR\ Log\ Anc -t lr_log_anc -mode MIN
python list_indomain.py -m ./Models/Anc -t anc -mode MAX
python list_indomain.py -m ./Models/LR\ Anc -t lr_anc -mode MAX
python list_indomain.py -m ./Models/Log\ Anc -t log_anc -mode MAX
python list_indomain.py -m ./Models/LR\ Log\ Anc -t lr_log_anc -mode MAX
```
- list_ood.py: Performance on in-domain numerals for the list max/min task, same as before, but with an additional {MIN, MAX} mode flag (-mode) for list min/max task.
```
python list_ood.py -m ./Models/Anc -t anc -mode MIN
python list_ood.py -m ./Models/LR\ Anc -t lr_anc -mode MIN
python list_ood.py -m ./Models/Log\ Anc -t log_anc -mode MIN
python list_ood.py -m ./Models/LR\ Log\ Anc -t lr_log_anc -mode MIN
python list_ood.py -m ./Models/Anc -t anc -mode MAX
python list_ood.py -m ./Models/LR\ Anc -t lr_anc -mode MAX
python list_ood.py -m ./Models/Log\ Anc -t log_anc -mode MAX
python list_ood.py -m ./Models/LR\ Log\ Anc -t lr_log_anc -mode MAX
```
The authors of our baselines have provided their pre-trained models in a manner similar to ours, ready to be used without training. Please find them here:
- [GenBERT](https://github.com/ag1988/injecting_numeracy)
- [MWP-BERT](https://github.com/LZhenwen/MWP-BERT)

If our research aids yours, please cite us. Thanks!
