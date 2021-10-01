from baseline.util import *
from nltk.tokenize import word_tokenize


##### Input Example Functions from baseline/util.py #####

# For regression training - STS: split samples into train, dev, test.
def split_sts(dataset):
    train, dev, test = [], [], []
    with gzip.open(dataset, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
            if row['split'] == 'dev':
                dev.append(inp_example)
            elif row['split'] == 'test':
                test.append(inp_example)
            else:
                train.append(inp_example)
    
    return train, dev, test

# For regression training - SICK-R: split samples into train, dev, test.
def split_sick(dataset):
    train, dev, test = [], [], []
    with open(dataset, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['relatedness_score']) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row['sentence_A'], row['sentence_B']], label=score)
            if row['SemEval_set'] == 'TRAIN':
                train.append(inp_example)
            elif row['SemEval_set'] == 'TEST':
                test.append(inp_example)
            else:
                dev.append(inp_example)
    return train, dev, test


def vector_base(filename):
    df = pd.read_csv(filename, sep=',')
    inputs = list()
    split = df.to_dict(orient='records')
    for item in split:
        if item['label'] == 'equal':
            label = 2
        elif item['label'] == 'nutral':
            label = 1
        else:
            label = 0
        inp_example = InputExample(texts=[item['textA'], item['textB']], label=label)
        inputs.append(inp_example)
    return inputs

# Base function for the MSRP dataset
def msrp_base(filename):
    df = pd.read_csv(filename, sep='\t')
    inputs = list()
    split = df.to_dict(orient='records')
    for item in split:
        inp_example = InputExample(texts=[item['#1 String'], item['#2 String']], label=item['Quality'])
        inputs.append(inp_example)
    return inputs

# prepare file for awesome-align
def write_alignment_sample(train_SRC, train_TGT, outputname):  
    f = open(outputname, 'w')
    for i in range(len(train_SRC)):
        left_source = ' '.join(word_tokenize(train_SRC[i].texts[0])) 
        left_target = ' '.join(word_tokenize(train_TGT[i].texts[0]))
        right_source = ' '.join(word_tokenize(train_SRC[i].texts[1]))
        right_target = ' '.join(word_tokenize(train_TGT[i].texts[1]))
        f.write(left_source + ' ||| ' + left_target + '\n')
        f.write(right_source + ' ||| ' + right_target + '\n')
    f.close()


if __name__ == "__main__":
    base = "../../data/"; suffix = "_ende.src-tgt"  
    # base directory is relative, can be any absolute path where
    # the corpora and the translations are stored.

    # STS-b
    stsb_en = f"{base}datasets/stsbenchmark.tsv.gz"; 
    stsb_de = ""
    stsb_SRC, _, _ = ""; stsb_TGT, _, _ = ""
    write_alignment_sample(stsb_SRC, stsb_TGT, f"{base}tp_lookups/stsb{suffix}")
    # SICK
    sick_en = f"{base}datasets/SICK/SICK_annotated.csv" 
    sick_de = f"{base}datasets/SICK/SICK_annotated_DE.csv"
    # MRPC
    mrpc_en = f"{base}datasets/MSRP/msr_paraphrase_train-new.csv" 
    mrpc_de = f"{base}datasets/MSRP/msr_paraphrase_train_DE.csv"
    # read vector & write_alignment_sample
