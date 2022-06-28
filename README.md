# Laying Anchors
This repository anonymously hosts the codebase, datasets, and models for the EMNLP 22 submission "Laying Anchors: Semantically Priming Numerals in Language Modeling" 

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

First, please download the [WikiText-103 word-level corpus](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) and place the contents (text files) inside /WikiText103/raw.

Run the preprocessing script:
```
python preprocess.py
```

This will populate the WikiText103 directory with pickle-ized objects:
- nums: The list of numerals extracted from WikiText-103
- means: The means of the Gaussians components extracted from nums
- log_means: The log-normalized means of the Gaussians components extracted from nums

> **Note:** To alter the number of gaussian components, please toy with the **gmm** function inside preprocess.py

## Step II: Training Models

> **Note:** All the models used in the paper are readily available in this repository, thus you can skip to Step III if you wish to skip re-training the models from scratch. 

First run the build_corpus script:
```
python build_corpus.py
```
This will populate the WikiText103/Training directory with model-specific training corpus. Using these corpus, we can now train our models.
```
python trainer.py -t "/path/to/tokenizer" -b "/path/to/base-model" -m "/path/to/save-model" -type {"exp", "anc", "loganc", "lr_anc", or "lr_loganc"} -d "path/to/training-corpus" 
```

## Step III: Extract Embeddings

There are 3 scripts for generating model embeddings as per the tasks described in the paper:
- embeddings.py: For the decoding task, the numeral ranges are defined as global variables. Use the model-specific functions to extract embeddings for the numeral range of interest. Toggle the cues flag as True or False depending on whether anchoring/exponent cues are to be provided to the models.
- add_embeddings.py: For the addition task, same as before, without the cues flag.
- list_embeddings.py: For the list max/min task, same as before, but with an additional {MIN, MAX} mode flag for list min/max task.

### Step IV: Regressors and Classifiers:

Now that we have embeddings for the models saved, we can use our regressors and classifiers to compare model performance across the tasks. Both the Xgboost-based regressor and the LSTM-based classifier take in path arguments for the saved location of X_train, y_train, X_test, y_test embeddings from step III.

Running the regressor:
```
python regressor.py -X "/path/to/X-train" -y "/path/to/y-train" -Xtest "/path/to/X-test" -ytest "/path/to/y-test"
```

Running the classifier:
```
python classifier.py -X "/path/to/X-train" -y "/path/to/y-train" -Xtest "/path/to/X-test" -ytest "/path/to/y-test"
```

Thanks!
