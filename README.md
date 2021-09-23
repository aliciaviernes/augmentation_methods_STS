# Exploration of Data Augmentation Methods for Semantic Textual Similarity Tasks

## Requirements

SBERT requirements & run `pip install -r requirements.txt`

## Baseline - SBERT

Baseline training obtained with 'bert-base-multilingual-cased' and SBERT pooling.

1. Regression training basic command:
`python3 train_regr.py [--stsb|--sick]`

2. Classification training basic command:
`python3 train_class.py`

## Data Augmentation Methods

### 1. EDA:
Modified code from Jason Wei and Kai Zou (insert citation).
Operations:
- Random Insertion (modified: contextually close word)
- Random Deletion
- Random Swap

Add `-e` to basic command.


### 2. TransPlacement:
Replace words in each data point with their translation.

Requirements:
- Translations of the train file (in the form `source sentence ||| target sentence`).
- Alignments between source and target sentences (in the form `0-0 1-1 2-3 3-3`).

Or use existing files (`data/tp_lookups/...`).

Add `-t` to basic command.


### 3. Synonym Replacement with Word Embeddings:

Word2Vec obtained with Gensim. For public datasets, pretrained Word2Vec is loaded.

Add `-w` to basic command.


### 4. Contextual Augmentation

A masked language model (MLM) is used, in this case t5-base.

Add `-c` to basic command.
