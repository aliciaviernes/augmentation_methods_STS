"""
Classification module. Structure as per SBERT suggestion.
Addition of MLflow logging info, iteration loops, and stats.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util 
import logging
from datetime import datetime
import time
import os
import baseline.util as util
import argparse
import json
import torch  # needed for classification - save state dict

from baseline.modelling import modifiedSentenceTransformer
from baseline.loss import modifiedSoftmaxLoss

from baseline.evaluate import LabelAccuracyEvaluator

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
aug_group = parser.add_mutually_exclusive_group()
aug_group.add_argument('--eda', '-e', action='store_true', help='Easy Data Augmentation')
aug_group.add_argument('--tp', '-t', action='store_true', help='TransPlacement')
aug_group.add_argument('--we', '-w', action='store_true', help='Word Embedding Synonym Replacement')
aug_group.add_argument('--ca', '-c', action='store_true', help='Contextual Augmentation (T5)')
args = parser.parse_args()


dataset = 'MSRP'
dataset_path = f'data/datasets/{dataset}/'
num_labels = 2 

# Specification of BERT model and save path.
model_name = 'distilbert-base-multilingual-cased'  # change to multilingual
model_save_path = f'output/training_class_{dataset}_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

starttime = time.time()
iter_outer = dict() # one dictionary for all iterations

for i in range(args.num_iterations):

    if i < 10:
        i = str(i).zfill(2)

    if dataset == 'MSRP':
        train_samples = util.msrp_base(f'{dataset_path}msr_paraphrase_train-new.csv')  # Important
        dev_samples = util.msrp_base(f'{dataset_path}msr_paraphrase_dev.csv')
        test_samples = util.msrp_base(f'{dataset_path}msr_paraphrase_test.csv')
        vector = False        

    else:
        train_samples = util.vector_base(f'{dataset_path}vector_trainfile.csv')
        dev_samples = util.vector_base(f'{dataset_path}vector_devfile.csv')
        test_samples = util.vector_base(f'{dataset_path}vector_testfile.csv')
        vector = True

    logging.info(f"ITERATION {i}")

    model = modifiedSentenceTransformer(model_name)

    iter_inner = {'epoch':[], 'global_steps':[]}  # inner dictionary for each iteration

    logging.info(f"Augment {dataset} dataset - Train")  # NOTE augmentation not saved - okay?
    # PLACEHOLDER FOR DATA AUGMENTATION FUNCTION
    
    if args.eda:
        train_samples = eda_train(train=train_samples, num_aug=3)
    elif args.tp:
        train_samples = tp_train(corpus='msrp', train=train_samples, nr_aug=3)
    elif args.we:
        train_samples = we_train(train=train_samples, nr_aug=3)
    elif args.ca:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_samples = ca_train(train=train_samples, batch_size=38, 
                        nr_augs=3, device=DEVICE)
    else:
        train_samples = util.zeroaugment(train_samples=train_samples)


    logging.info(f"Load {dataset} dataset, initialize loss and evaluators.")
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)
    len_datapoints = len(train_dataloader)
    dev_dataloader = DataLoader(dev_samples, shuffle=False, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_samples, shuffle=False, batch_size=args.batch_size)

    # Loss & evaluators
    train_loss = modifiedSoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=num_labels)
    dev_evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss, name=f'{dataset}-dev')
    
    # Configure warm-up steps
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * args.warmup_percent)
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=args.num_epochs,
            evaluation_steps=len_datapoints // args.eval_loops + 1,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            iter_dict=iter_inner,
            iteration=i)
    
    # Needed for Class: Load the model incl. loss.
    model = modifiedSentenceTransformer(model_save_path)
    bestloss = modifiedSoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=num_labels)
    bestloss.classifier.load_state_dict(torch.load(model_save_path + '/bestmodel_classifier.pth'))
    test_evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=bestloss, name=f'{dataset}-test')
    test_evaluator(model, output_path=model_save_path, iter_dict=iter_inner, i=i)

    iter_inner['epoch'].remove(-1)
    iter_inner['global_steps'].remove(-1)
    iter_outer[int(i)] = iter_inner

outputfile = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(f'{dataset}_{outputfile}.json', 'w') as f:
    json.dump(iter_outer, f, indent=2)
util.sum_best_dev_class(f'{dataset}_{outputfile}')
util.add_statistics(f'{dataset}_{outputfile}.csv')

    
endtime = time.time()  
elapsed = endtime - starttime
logging.info(f"Time elapsed: {util.hours_minutes_seconds(elapsed)}")
