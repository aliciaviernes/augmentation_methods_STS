# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou
# Adaptation for master's thesis.

import random
from random import shuffle
random.seed(1)  # NOTE this might be the cause for non-variance

#cleaning up text
import re

def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")

    for char in line:
        if char.isalpha() or char == ' ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


########################################################################
# Random deletion - Anagnostopoulou modification.
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = delete_word(new_words)
	return new_words

def delete_word(words):
	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	#randomly delete words with probability p
	new_words = []
	r = random.randint(0, len(words)-1)
	for i in range(len(words)):
		if r != i:
			new_words.append(words[i])

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

########################################################################
# Random swap - EDA.
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

########################################################################
# Random insertion - Anagnostopoulou modification.
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n, voc, pairs):  # adds this many new words!
	new_words = words.copy()
	for _ in range(n):
		new_words = add_word(new_words, voc, pairs)
	return new_words

def add_word(new_words, voc, pairs):
	random_word = new_words[random.randint(0, len(new_words)-1)]
	if random_word in pairs:
		new_insertion = pairs[random_word]
		random_idx = random.randint(0, len(new_words)-1)
		new_words.insert(random_idx, new_insertion)
	else:
		new_insertion = random.choice(list(voc))
		random_idx = random.randint(0, len(new_words)-1)
		new_words.insert(random_idx, new_insertion)
	
	return new_words


########################################################################
# main data augmentation function
########################################################################

def eda(sentence, voc, pairs, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
	
	sentence = get_only_chars(sentence)
	words = sentence.split(' ')
	words = [word for word in words if word != '']
	num_words = len(words)  # sentence preprocessing
	
	augmented_sentences = []
	num_new_per_technique = max(1, int(num_aug/3)) 

	#ri
	if (alpha_ri > 0):
		n_ri = max(1, int(alpha_ri*num_words)) # a = 0.1, nw = 5
		for _ in range(num_new_per_technique):
			a_words = random_insertion(words, n_ri, voc, pairs)  # 3 augmentations with inserted words
			augmented_sentences.append(' '.join(a_words))

	#rs
	if (alpha_rs > 0):
		n_rs = max(1, int(alpha_rs*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_swap(words, n_rs)  # 3 augmentations with swapped words
			augmented_sentences.append(' '.join(a_words))

	#rd
	if (p_rd > 0):
		n_rd = max(1, int(p_rd*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_deletion(words, n_rd) # 3 augmentations with deleted words
			augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
	shuffle(augmented_sentences)

	return augmented_sentences
