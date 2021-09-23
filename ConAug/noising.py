"""
    https://discuss.huggingface.co/t/train-t5-from-scratch/1781?u=amlarraz
"""

import random
from transformers import T5Tokenizer


def racha_detection(lista):
    # It returns a list of lists where each sub-list contains the consecutive tokens in the list
    rachas = []
    racha = []
    for i, element in enumerate(lista):
        if (i<len(lista)-1) and (lista[i+1] == element+1):
            racha.append(element)
        else:
            if len(racha)>0:
                rachas.append(racha + [element])          
            else:# (i!=len(lista)-1):
                rachas.append([element])
            racha = []
    return rachas


def masking(tokenized_sentence, rachas, tokenizer):
    # Function to mask a tokenized_sentence (token ids) following the rachas described in rachas
    # Only one sentinel_token per racha
    sent_token_id = 0
    enmascared = tokenized_sentence.copy()
    for racha in rachas:
        sent_token = f'<extra_id_{sent_token_id}>'
        sent_id = tokenizer.encode(sent_token)[0]
        for i, idx in enumerate(racha):
            if i==0:
                enmascared[idx] = sent_id
            else:
                enmascared[idx] = -100
        sent_token_id += 1
    
    enmascared = [t for t in enmascared if t!=-100] 

    return enmascared


def add_noise(tokenized_sentence, tokenizer, percent=0.15):
    # Function that takes a sentence, tokenizer and a noise percentage and returns
    # the masked input_ids and masked target_ids accordling with the T5 paper and HuggingFace docs

    idxs_2_mask = sorted(random.sample(range(len(tokenized_sentence)), 
                                       int(len(tokenized_sentence)*percent)))
    rachas = racha_detection(idxs_2_mask)
    enmascared_input = masking(tokenized_sentence, rachas, tokenizer)
    idxs_2_mask = [idx for idx in range(len(tokenized_sentence)) if idx not in idxs_2_mask]
    rachas = racha_detection(idxs_2_mask)
    enmascared_target = masking(tokenized_sentence, rachas, tokenizer)

    return tokenizer.decode(enmascared_input), tokenizer.decode(enmascared_target)
