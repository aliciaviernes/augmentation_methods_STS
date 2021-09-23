from baseline.util import *
import EDA.mod_eda as mod_eda

"""
ONLY function to use here:
eda_train(train)
Important parameters:
    - shuffle
    - vector (False if dataset is public else True)
"""

import itertools, random

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopwords_EN = set(stopwords.words('english'))
stopwords_DE = set(stopwords.words('german'))
stopwords_VE = stopwords_EN.union(stopwords_DE)


def add_to_set(tokenlist, voc, stopwords): # helping function for eda_gen_voc
    for token in tokenlist:
        if token not in stopwords and token not in voc:
            voc.add(token)
    return voc


def eda_gen_voc(train, stopwords):  # helping function for eda_train
    voc = set()
    for example in train:
        texts = example.texts
        for text in texts:
            tokens = word_tokenize(mod_eda.get_only_chars(text))
            voc = add_to_set(tokens, voc, stopwords)
    return voc


def eda_train(train, vector=False, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, num_aug=3, shuffle=True):
    
    stopwords = stopwords_VE if vector else stopwords_EN
    augExamples = list()
    # Create voc
    voc = eda_gen_voc(train, stopwords)
    # Get co-occurences
    occs_2, occs_3 = eda_get_occurences(train, stopwords)
    # pairs2 = eda_get_context_pairs(occs_2)
    # Create pairs - threshold: 3 occurences
    pairs3 = eda_get_context_pairs(occs_3)
    for example in train:
        augExamples.append(example)  # train is added
        label = example.label; texts = example.texts
        # left sentence augmentation
        aug_l = mod_eda.eda(texts[0], voc, pairs3, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        for i in range(len(aug_l)):
            inp_example = InputExample(texts=[aug_l[i], texts[1]], label=label)
            augExamples.append(inp_example)
        # right sentence augmentation
        aug_r = mod_eda.eda(texts[1], voc, pairs3, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        for i in range(len(aug_r)):
            inp_example = InputExample(texts=[texts[0], aug_r[i]], label=label)
            augExamples.append(inp_example)
    
    if shuffle:
        random.shuffle(augExamples)

    return augExamples 


def add_to_dict(tokenlist, stopwords, occurences):  # helping function for eda_get_occurences
    for token in tokenlist:
        if token not in stopwords:
            neighbors = [word for word in tokenlist if not word in stopwords]
            neighbors.remove(token)
            if token not in occurences:                      
                occurences[token] = dict()
                for neighbor in neighbors:
                    occurences[token][neighbor] = 1
            else:
                for neighbor in neighbors:
                    if neighbor not in stopwords:
                        if neighbor in occurences[token]:
                            occurences[token][neighbor] += 1
                        else:
                            occurences[token][neighbor] = 1
    return occurences


def threshold_occs(token, occurences, t):  # helping function for eda_get_occurences
    new_neighbors = set()
    for neighbor in occurences[token]:
        if occurences[token][neighbor] > t:
            new_neighbors.add(neighbor)
    return new_neighbors


def eda_get_occurences(train, stopwords):  # helping function for eda_train
    occurences = dict()
    occs_2 = dict(); occs_3 = dict()

    for example in train:
        texts = example.texts
        for text in texts:
            tokens = word_tokenize(mod_eda.get_only_chars(text))
            occurences = add_to_dict(tokens, stopwords, occurences)

    for token in occurences:
        new_neighbors_2 = threshold_occs(token, occurences, t=2)
        new_neighbors_3 = threshold_occs(token, occurences, t=3)
        if new_neighbors_2 != set():
            occs_2[token] = new_neighbors_2
        if new_neighbors_3 != set():
            occs_3[token] = new_neighbors_3

    return occs_2, occs_3


def eda_get_context_pairs(occurences):  # helping function for eda_train
    pairs = dict()
    for a, b in itertools.combinations(occurences, 2):
        overlap = occurences[a] & occurences[b]
        if overlap != set:
            if len(overlap) > 0.5 * len(occurences[a]) and len(overlap) > 0.5 * len(occurences[b]):
                pairs[a] = b
                pairs[b] = a
    return pairs
