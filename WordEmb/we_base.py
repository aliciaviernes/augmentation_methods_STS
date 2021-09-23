# from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec
import random
from nltk.corpus import stopwords
from nltk import word_tokenize

from baseline.util import *

stop_words = set(stopwords.words('english'))
stop_words.add('A')
stop_words.add('The')

"""
    Word embedding similarity: HOWTO
    1. choose proportion
    2. choose random word(s) - should not be a stopword.
    3. for each word:
        find the most similar one - Variation: find one of the most similar ones.
    4. replace
"""

def UNK_treatment(tokens, idx, idxs, tobereplaced, model, c=0):
    new_choice = idxs - tobereplaced
    if len(new_choice) == 0:
        similar_word = tokens[idx]
    else:
        new_idx = random.sample(new_choice, 1)
        new_word, uc = transform_for_replacement(tokens[new_idx[0]])
        if new_word in model:
            similar_word = model.most_similar(positive=[new_word], topn=1)[0][0]
        else:
            if c < 5:
                c += 1
                newTBR = tobereplaced.union(set(new_idx))
                similar_word = UNK_treatment(tokens, idx, idxs, #BUG
                                            newTBR, model, c)
            else:
                similar_word = tokens[idx]  # initial word stays the same, we quit.
        similar_word = transform_for_sentence(similar_word, uc)
    return similar_word
        

def transform_for_replacement(word):
    if word[0].isupper():
        return word.lower(), True
    else:
        return word, False


def transform_for_sentence(word, uc):
    if uc == True:
        word = word.replace(word[0], word[0].upper(), 1)
    return word


def gather_lex_words(tokenlist, stopwords):
    lex_tokens = list()
    punc = {'.', ',', '!', '?', '-', '\''}
    for w in tokenlist:
        if w not in stopwords and w != '' and w not in punc:
            lex_tokens.append(w)
    return lex_tokens


def lex_idxs(tokens, lex_s):  # check which (positions of) tokens are lexical
    idxs = set()
    for i in range(len(tokens)):
        if tokens[i] in lex_s:
            idxs.add(i)
    return idxs


def wordemb_replace(idx, tokens, model, idxs, tobereplaced):
    tokens[idx], uc = transform_for_replacement(tokens[idx])
    if tokens[idx] in model:
        similar_word = model.most_similar(positive=[tokens[idx]], topn=1)[0][0]
        similar_word = transform_for_sentence(similar_word, uc)
    else:
        similar_word = UNK_treatment(tokens, idx, idxs, tobereplaced, model)
        # print(f'Replacement: {similar_word}')
    tokens[idx] = similar_word 
    return tokens

################################# *°~ Main function ~°* #################################

def replace_random_words(s, model, stop_words=stop_words, a=0.1):
    # Tokenizing sentence
    tokens = word_tokenize(s)
    if len(tokens) == 1:
        return tokens[0]
    else:
        # Gathering all lexical items (no punc or stopwords)
        lex_s = gather_lex_words(tokens, stop_words)
        # Converting them to positions
        idxs = lex_idxs(tokens, lex_s)
        # Proportion of words to be augmented
        num_we = max(int(len(tokens) * a), 1)
        # These are the words to be replaced.
        tobereplaced = set(random.sample(idxs, num_we)) 
        for idx in tobereplaced: 
            tokens = wordemb_replace(idx, tokens, model, idxs, tobereplaced)
        return ' '.join(tokens)
