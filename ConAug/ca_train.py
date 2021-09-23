import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import random, itertools, time
from tqdm import tqdm

from ConAug.t5_aug_base import *
from baseline.util import *


def chop_aug_set(text, aug_batch, nr_aug):
    aug_batch = set(aug_batch)  # unique
    if text in aug_batch:
        aug_batch.remove(text)  # remove non augmented
    if len(aug_batch) <= nr_aug:
        return aug_batch  # return all because they are less
    else:
        return random.sample(list(aug_batch), nr_aug)  # return right amount


def aug_prep(train_batch, nr_augs):
    only_sentences = list()
    for inpEx in train_batch:
        for text in inpEx.texts:
            reps = list(itertools.repeat(text, nr_augs * 2))  # for the case of cases
            only_sentences.extend(reps)
    return only_sentences


def prep_and_augment(train_batch, nr_augs, tokenizer, mlm, device):
    return context_aug_batch(aug_prep(train_batch, nr_augs), tokenizer, mlm, device)


def ca_augment_in_batches(train, batch_size, nr_augs, tokenizer, mlm, device):
    train_batch = list(); aug_train = list()
    copyOfTrain = train.copy()
    while len(copyOfTrain) > batch_size:
        # Slicing
        train_batch = copyOfTrain[:batch_size]
        copyOfTrain = copyOfTrain[batch_size:]
        # Augmentation (double as much as we need, now)
        train_batch = prep_and_augment(train_batch, nr_augs, tokenizer, mlm, device)
        aug_train.extend(train_batch)
    # Last batch is small
    train_batch = copyOfTrain
    train_batch = prep_and_augment(train_batch, nr_augs, tokenizer, mlm, device)
    aug_train.extend(train_batch)
    # nr_augs * 2 because of buffer
    # * 2 because inputExample has two texts
    assert len(train)* nr_augs * 2 * 2 == len(aug_train)
    return aug_train


def ca_train(train, batch_size, nr_augs, device, v=False, shuffle=True):
    # NOTE if v == True, the vector finetuned model is used.
    model = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model)
    config = T5Config.from_pretrained(model)
    mlm = T5ForConditionalGeneration.from_pretrained(model, config=config).to(device)
    starttime = time.time()
    augExamples = list()
    augmentations = ca_augment_in_batches(train, batch_size, nr_augs, tokenizer, mlm, device)
    augtime = time.time()
    print(f"Augmentation duration: {hours_minutes_seconds(augtime - starttime)}")
    assert len(augmentations) % 12 == 0
    # create new train
    for i in tqdm(range(len(train)), desc=f"Combining basic corpus and augmentations"):
        augExamples.append(train[i])
        label = train[i].label
        text_l = train[i].texts[0]; text_r = train[i].texts[1]
        relevant = augmentations[:12]; augmentations = augmentations[12:]
        augs_l = chop_aug_set(text_l, relevant[:6], nr_augs)
        augs_r = chop_aug_set(text_r, relevant[6:], nr_augs)
        # LEFT SIDE AUGMENTATION
        if len(augs_l) != 0:
            for aug in augs_l:
                inpEx = InputExample(texts=[aug, text_r], label=label)
                augExamples.append(inpEx)
        # RIGHT SIDE AUGMENTATION
        if len(augs_r) != 0:
            for aug in augs_r:
                inpEx = InputExample(texts=[text_l, aug], label=label)
                augExamples.append(inpEx)
    
    if shuffle:
        random.shuffle(augExamples)
    
    endtime = time.time()
    print(f"Total time: {hours_minutes_seconds(endtime - starttime)}")
    
    return augExamples
