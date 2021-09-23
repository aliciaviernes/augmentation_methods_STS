import TP.tp_util as tpu
from baseline.util import *
import random, time

starttime = time.time()

# Introduces number of augmentations (how many augmented datapoints do I want?)
def tp_augment(alignmentfile, sentencefile, nr_aug=3):
    augmentations_dict = dict()
    for i in range(nr_aug):
        augmentations_single = tpu.trans_place_all(alignmentfile, sentencefile)
        for a in range(len(augmentations_single)):
            if a not in augmentations_dict:
                augmentations_dict[a] = set()
            augmentations_dict[a].add(augmentations_single[a])
    return augmentations_dict


def tp_train(corpus, train, alignmentid='ende.gold', nr_aug=3, shuffle=True):
    # NOTE possible corpora:
    # 'stsb', 'sick', 'msrp'
    # Can change alignmentid to something else.
    augExamples = list()
    all_augmentations = tp_augment(
                alignmentfile=f'data/tp_lookups/{corpus}_{alignmentid}', 
                sentencefile=f'data/tp_lookups/{corpus}_ende.src-tgt',
                nr_aug=nr_aug
                )
    for i in range(len(train)):
        augExamples.append(train[i])
        label = train[i].label; texts = train[i].texts
        aug_idx_l = i * 2
        aug_idx_r = aug_idx_l + 1
        for aug_sent in all_augmentations[aug_idx_l]:
            if aug_sent != texts[0]:
                inp_example = InputExample(texts=[aug_sent, texts[1]], label=label)
                augExamples.append(inp_example)
        for aug_sent in all_augmentations[aug_idx_r]:
            if aug_sent != texts[1]:
                inp_example = InputExample(texts=[texts[0], aug_sent], label=label)
                augExamples.append(inp_example)
    
    if shuffle:
        random.shuffle(augExamples)
    
    return augExamples
