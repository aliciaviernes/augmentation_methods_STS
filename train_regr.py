"""
Regression module. Structure as per SBERT suggestion.
Addition of MLflow logging info, iteration loops, and stats.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler
import logging
from datetime import datetime
import time
import os
import baseline.util as util
import argparse
import json
import torch

from baseline.modelling import modifiedSentenceTransformer
from baseline.loss import modifiedCosineSimilarityLoss

from baseline.evaluate import EmbeddingSimilarityEvaluator

from EDA.mod_augment import *
from TR.tp_train import *
from WordEmb.we_train import *
from ConAug.ca_train import *

parser = argparse.ArgumentParser(description="Training hyperparameters")
parser.add_argument('--batch_size', '-b', type=int, default=256, required=False, help="Train batch size")
parser.add_argument('--num_epochs', '-n', type=int, default=4, required=False, help="Number of epochs")
parser.add_argument('--eval_loops', '-v', type=int, default=8, required=False, help="Evaluation loops in each epoch")
parser.add_argument('--warmup_percent', type=float, default=0.1, required=False, help="Percent of warming steps (0.1 for 10%)")
parser.add_argument('--num_iterations', '-i', type=int, default=20, required=False, help="Number of iterations")
datagroup = parser.add_mutually_exclusive_group()
datagroup.add_argument('--stsb', action='store_true', help='STSb dataset')  
datagroup.add_argument('--sick', action='store_true', help='SICK dataset')
aug_group = parser.add_mutually_exclusive_group()
aug_group.add_argument('--eda', '-e', action='store_true', help='Easy Data Augmentation')
aug_group.add_argument('--tp', '-t', action='store_true', help='TransPlacement')
aug_group.add_argument('--we', '-w', action='store_true', help='Word Embedding Synonym Replacement')
aug_group.add_argument('--ca', '-c', action='store_true', help='Contextual Augmentation (T5)')
args = parser.parse_args()

if args.stsb:
    dataset = 'stsb'
    dataset_path = 'data/datasets/stsbenchmark.tsv.gz' 
elif args.sick:
    dataset = 'sick'
    dataset_path = 'data/datasets/SICK/SICK_annotated.csv'
else:
    print("Please choose a dataset [--stsb|-sick]")


# Specification of BERT model and save path.
model_name = 'distilbert-base-multilingual-cased'  # change to multilingual
model_save_path = f'output/training_regr_{dataset}_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

starttime = time.time()
iter_outer = dict()  # one dictionary for all iterations

for i in range(args.num_iterations):

    if i < 10:
        i = str(i).zfill(2) 

    if dataset == 'stsb':  # Loading of pre-split data
        train_samples, dev_samples, test_samples = util.split_sts(dataset_path)
    elif dataset == 'sick':
        train_samples, dev_samples, test_samples = util.split_sick(dataset_path)

    logging.info(f"ITERATION {i}")

    model = modifiedSentenceTransformer(model_name)
    
    iter_inner = {'epoch':[], 'global_steps':[]}  # inner dictionary for each iteration    
            
    logging.info(f"Augment {dataset} dataset - Train")  # NOTE augmentation not saved - okay?
    # PLACEHOLDER FOR DATA AUGMENTATION FUNCTION    
    if args.eda:
        train_samples = eda_train(train=train_samples, num_aug=3)
    elif args.tp:
        train_samples = tp_train(corpus=dataset, train=train_samples, nr_aug=3)
        # OPTIONAL argument: alignmentid
    elif args.we:
        train_samples = we_train(train=train_samples, nr_aug=3)
    elif args.ca:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_samples = ca_train(train=train_samples, batch_size=38, nr_augs=3, device=DEVICE)
    else:
        train_samples = util.zeroaugment(train_samples=train_samples)

    # NOTE this is not good coding but a quick fix.
    if len(train_samples) % 64 == 1:
        del train_samples[-1]
    logging.info(f"In this loop there are {len(train_samples)} train samples.")
    
    logging.info(f"Load {dataset} dataset, initialize loss and evaluators.")
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)
    evaluation_steps = calculate_eval_steps(len(train_dataloader), args.eval_loops)
    logging.info(f"Evaluation steps: {evaluation_steps}")
    
    # Loss & evaluators
    train_loss = modifiedCosineSimilarityLoss(model=model)
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name=f'{dataset}-dev')
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name=f'{dataset}-test')
    
    # Configure warm-up steps
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * args.warmup_percent)
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=args.num_epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            iter_dict=iter_inner,
            iteration=i)

    model = modifiedSentenceTransformer(model_save_path)
    test_evaluator(model, output_path=model_save_path, iter_dict=iter_inner, i=i)

    iter_inner['epoch'].remove(-1)  
    iter_inner['global_steps'].remove(-1)
    iter_outer[int(i)] = iter_inner  # keep i here!

# statistics (best metric for each iteration - mean / median / min / max / stdev / var)
outputfile = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(f'{dataset}_{outputfile}.json', 'w') as f:
    json.dump(iter_outer, f, indent=2)
util.sum_best_dev(f'{dataset}_{outputfile}')
util.add_statistics(f'{dataset}_{outputfile}.csv')


endtime = time.time()  
elapsed = endtime - starttime
logging.info(f"Time elapsed: {util.hours_minutes_seconds(elapsed)}")
