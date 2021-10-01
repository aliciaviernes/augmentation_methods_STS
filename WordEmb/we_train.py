from gensim.models import Word2Vec, KeyedVectors 
from WordEmb.we_base import *
# from gensim.scripts.glove2word2vec import glove2word2vec


# glove_file = 'glove/glove.6B.50d.txt'
tmp_file = 'WordEmb/glove.6B.50d-temp.txt'
# _ = glove2word2vec(glove_file, tmp_file)
wv_base = KeyedVectors.load_word2vec_format(tmp_file)

stop_words = set(stopwords.words('english'))
stop_words.add('A')
stop_words.add('The')

wv_vector = Word2Vec.load('WordEmb/vector/vector_model.bin')

def create_aug_set(nr_aug, text, model, stop_words):
    aug_set = set()
    for _ in range(nr_aug):
        aug = replace_random_words(text, model=model, stop_words=stop_words)
        if aug != text:
            aug_set.add(aug)
    return aug_set


def we_train(train, nr_aug=3, v=False, shuffle=True, stop_words=stop_words):
    augExamples = list()
    model = wv_base if v == False else wv_vector
    for inpEx in train:
        augExamples.append(inpEx)
        label = inpEx.label
        text_l = inpEx.texts[0]; text_r = inpEx.texts[1]
        # LEFT SIDE AUGMENTATION
        augs_l = create_aug_set(nr_aug=nr_aug, text=text_l, model=model, stop_words=stop_words)
        for aug in augs_l:
            inp_example = InputExample(texts=[aug, text_r], label=label)
            augExamples.append(inp_example)
        # RIGHT SIDE AUGMENTATION
        augs_r = create_aug_set(nr_aug=nr_aug, text=text_r, model=model, stop_words=stop_words)
        for aug in augs_r:
            inp_example = InputExample(texts=[text_l, aug], label=label)
            augExamples.append(inp_example)
    
    if shuffle:
        random.shuffle(augExamples)
    
    return augExamples
