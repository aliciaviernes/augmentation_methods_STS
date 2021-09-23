from sentence_transformers.readers import InputExample
import gzip
import csv, json, re
import numpy as np
import pandas as pd
from statistics import mean, median, stdev, variance


####################### °*~ FILE READING AND PREPARATION FUNCTIONS ~*° #######################

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


# Base function for the German Dataset
def sts_de_base(filename):
    samples = list()
    with open(filename, 'rt', encoding='utf8') as fIn:
        reader = csv.reader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row[0]) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row[1], row[2]], label=score)
            samples.append(inp_example)    
    return samples

# For regression traning - STS-b German -- from a directory!
def split_sts_DE(directory):
    train = sts_de_base(directory + 'stsb_de_train.csv')
    dev = sts_de_base(directory + 'stsb_de_dev.csv')
    test = sts_de_base(directory + 'stsb_de_test.csv')
    return train, dev, test

# Base function for the Vector internal dataset
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


################################ °*~ STATISTICS FUNCTIONS ~*° ################################

def get_best_score_index(sourcelist):
    return(sourcelist.index(max(sourcelist)))


def sum_best_dev(filename):  
    with open(f'{filename}.json') as json_file:
        data = json.load(json_file)

    headers = ['train_loss', 'train_spearman_cosine', 'train_pearson_cosine', 'train_mse_cosine', 'train_mae_cosine',
                'dev_spearman_cosine', 'dev_pearson_cosine', 'dev_mse_cosine', 'dev_mae_cosine',
                'test_spearman_cosine', 'test_pearson_cosine', 'test_mse_cosine', 'test_mae_cosine']

    newdata = []
    for i in range(len(data)):
        source = data[str(i)]['dev']['spearman_cosine']
        bsi = get_best_score_index(source)  # bsi stands for Best Score Index
        row = dict()  # creates a row
        for k in data[str(i)].keys():
            if k == 'dev' or k =='train':
                for key in data[str(i)][k].keys():
                    row[k + '_' + key]= data[str(i)][k][key][bsi]
            elif k == 'test':
                for key in data[str(i)][k].keys():
                    row[k + '_' + key]= data[str(i)][k][key][0]
        newdata.append(row)
    try:
        with open(f'{filename}.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in newdata:
                writer.writerow(row)
    except IOError:
        print("I/O error")


def sum_best_dev_class(filename):  
    with open(f'{filename}.json') as json_file:
        data = json.load(json_file)
   
    headers = ['train_loss', 'train_accuracy', 'train_macro_precision', 'train_macro_recall', 
                'train_macro_f1', 'train_micro_precision', 'train_micro_recall', 'train_micro_f1',
                'dev_accuracy', 'dev_macro_precision', 'dev_macro_recall', 'dev_macro_F1', 
                'dev_micro_precision', 'dev_micro_recall', 'dev_micro_F1', 
                'test_accuracy', 'test_macro_precision', 'test_macro_recall', 'test_macro_F1',
                'test_micro_precision', 'test_micro_recall', 'test_micro_F1']

    newdata = []
    for i in range(len(data)):
        source = data[str(i)]['dev']['macro_F1']
        bsi = get_best_score_index(source)  # bsi stands for Best Score Index
        row = dict()  # creates a row
        for k in data[str(i)].keys():
            if k == 'dev' or k =='train':
                for key in data[str(i)][k].keys():
                    row[k + '_' + key]= data[str(i)][k][key][bsi]
            elif k == 'test':
                for key in data[str(i)][k].keys():
                    row[k + '_' + key]= data[str(i)][k][key][0]
        newdata.append(row)
    try:
        with open(f'{filename}.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in newdata:
                writer.writerow(row)
    except IOError:
        print("I/O error")


def add_statistics(csvfile):
    df = pd.read_csv(csvfile)
    headers = list()
    for col in df.columns:
        headers.append(col)
    anotherdict = {'median':[], 'mean':[], 'stdev':[], 'var':[], 'min':[], 'max':[]}
    for metric in headers:
        values = df[metric].values.tolist() # log them
        
        anotherdict['mean'].append(mean(values))
        
        anotherdict['median'].append(median(values))
        
        anotherdict['stdev'].append(stdev(values))
        
        anotherdict['var'].append(variance(values))
        
        anotherdict['min'].append(min(values))
        
        anotherdict['max'].append(max(values))
    
    newdf = pd.DataFrame.from_dict(anotherdict, orient='index')
    newdf.columns = headers
    result = pd.concat([df, newdf])
    result.to_csv(csvfile, index=True)


################################# °*~ GENERAL FUNCTIONS ~*° #################################

def hours_minutes_seconds(elapsed):
    hours = elapsed // 3600
    rest = elapsed % 3600
    minutes = rest // 60
    seconds = rest % 60

    return str(int(hours)) + ' hours | ' + str(int(minutes)) + ' minutes | ' + str(round(seconds, 2)) + ' seconds'


"""
# DEPRECATED
def split_tickets(filename):
    df = pd.read_csv(filename, sep='|')
    train, dev, test = np.split(df.sample(frac=1, random_state=42), 
                       [int(.8*len(df)), int(.9*len(df))])  # partition: 80 - 10 - 10
                       # größe: train.shape[0]
    train_samples = vector_base(train)
    dev_samples = vector_base(dev)
    test_samples = vector_base(test)
    return train_samples, dev_samples, test_samples

# Randomly distribute MSRP train into train and dev.
# do this one-time
def newsplit(path2dataset, drop_rate=0.2):
    dev_id = set()
    with open(path2dataset, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if random.uniform(0,1) < drop_rate:
                dev_id.add(row['#1 ID'])
    
    return dev_id

# One time - convert txt to csv
with open('./datasets/SICK/SICK_annotated.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split("\t") for line in stripped if line)
    with open(path2sick, 'w') as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        writer.writerows(lines)
"""

def zeroaugment(train_samples):
    return train_samples


def read_save_csv(filename, delimiter):  # NOTE maybe add to util
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        data = list(reader)
    return data


def read_langs(filename): 
    langs = list()
    data = read_save_csv(filename, '\t')  # we know that the lang file has this delimiter.
    data = data[1:]  # chop header off
    for row in data:
        langs.extend((row[1], row[2]))
    return langs


def calculate_eval_steps(len_datapoints, evaluation_loops):
    eval_steps = len_datapoints // evaluation_loops + 1
    return eval_steps


def beautiful_text(text):
    return re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)
