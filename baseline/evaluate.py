from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
from sentence_transformers.readers import InputExample
from sentence_transformers.util import batch_to_device
from scipy.stats import pearsonr, spearmanr
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances 
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, classification_report
import numpy as np
from typing import List
import json
import logging
import os
import csv


logger = logging.getLogger(__name__)


class EmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.
    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float], batch_size: int = 16, 
                main_similarity: SimilarityFunction = None, name: str = '', 
                show_progress_bar: bool = False, write_csv: bool = True):
        """
        Constructs an evaluator based for the dataset
        The labels need to indicate the similarity between the sentences.
        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: Similarity score between sentences1[i] and sentences2[i]
        :param write_csv: Write results to a CSV file
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.write_csv = write_csv

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.main_similarity = main_similarity
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "global_steps", "cosine_pearson", "cosine_spearman", "euclidean_pearson", 
                            "euclidean_spearman", "manhattan_pearson", "manhattan_spearman", "dot_pearson", "dot_spearman", 
                            "MSE", "MAE"] # new additions


    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)


    def __call__(self, model, iter_dict, i, output_path: str = None, epoch: int = -1, global_steps: int = -1) -> float:

        out_txt = " in epoch {} after {} global steps:".format(epoch, global_steps)
        
        if 'dev' in self.name:
            evaltype = 'dev'
        elif 'test' in self.name:
            evaltype = 'test'
        else:
            evaltype = 'train'

        logger.info("EmbeddingSimilarityEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        iter_dict['epoch'].append(epoch)
        iter_dict['global_steps'].append(global_steps)

        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        labels = self.scores  # true

        # PRED
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))  # Similarity
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)  # Distance
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)  # Distance
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)] # Similarity


        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)  # Choose this
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)
        eval_mse_cosine = mean_squared_error(labels, cosine_scores)
        eval_mae_cosine = mean_absolute_error(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)  # Or this
        eval_spearman_dot, _ = spearmanr(labels, dot_products)
        eval_mse_dot = mean_squared_error(labels, dot_products)
        eval_mae_dot = mean_absolute_error(labels, dot_products)

        metric_names = ['spearman_cosine', 'pearson_cosine', 'mse_cosine', 'mae_cosine']
        metrics = [eval_spearman_cosine, eval_pearson_cosine, eval_mse_cosine, eval_mae_cosine]

        if evaltype not in iter_dict:
            iter_dict[evaltype] = dict()
            for n in range(len(metric_names)):
                iter_dict[evaltype][metric_names[n]] = [metrics[n]]
        else:
            for n in range(len(metric_names)):
                iter_dict[evaltype][metric_names[n]].append(metrics[n])
        
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, global_steps, eval_pearson_cosine, eval_spearman_cosine, eval_pearson_euclidean,
                                 eval_spearman_euclidean, eval_pearson_manhattan, eval_spearman_manhattan, 
                                 eval_pearson_dot, eval_spearman_dot])  # ADD

        
        return eval_spearman_cosine
        

class LabelAccuracyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset
    This requires a model with LossFunction.SOFTMAX
    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model = None, write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset
        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        if name:
            name = "_"+name

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "global_steps", "accuracy", "macro precision", "macro recall", "macro F-score", "micro precision", "micro recall", "micro F-score"]

    def __call__(self, model, iter_dict, i, output_path: str = None, epoch: int = -1, global_steps: int = -1) -> float:
        model.eval()
        total, correct = 0, 0
        pred, true = list(), list()

        out_txt = " in epoch {} after {} global steps:".format(epoch, global_steps)
        
        if 'dev' in self.name:
            evaltype = 'dev'
        elif 'test' in self.name:
            evaltype = 'test'
        else:
            evaltype = 'train'
        
        iter_dict['epoch'].append(epoch)
        iter_dict['global_steps'].append(global_steps)

        logger.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            true.extend(label_ids.cpu().numpy())
            
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)  # to self device: bug
            
            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()

            pred.extend(torch.argmax(prediction, dim=1).cpu().numpy())
            
        accuracy = correct/total
        a_precision, a_recall, a_fscore, _ = precision_recall_fscore_support(true, pred, average='macro', zero_division=1)
        i_precision, i_recall, i_fscore, _ = precision_recall_fscore_support(true, pred, average='micro', zero_division=1)

        metric_names = ['accuracy', 'macro_precision', 'macro_recall', 'macro_F1', 'micro_precision', 'micro_recall', 'micro_F1']
        metrics = [accuracy, a_precision, a_recall, a_fscore, i_precision, i_recall, i_fscore]
        
    
        """
            On micro - macro scores:
            https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/#:~:text=One%20that%20has%20no%20false,the%20higher%20is%20its%20precision.&text=So%20this%20already%20explains%20why,and%20recall%20are%20the%20same.
        """
        
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, global_steps, accuracy, a_precision, a_recall, a_fscore, i_precision, i_recall, i_fscore])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, global_steps, accuracy, a_precision, a_recall, a_fscore, i_precision, i_recall, i_fscore])
        
        if evaltype not in iter_dict:
            iter_dict[evaltype] = dict()
            for i in range(len(metric_names)):
                iter_dict[evaltype][metric_names[i]] = [metrics[i]]
        else:
            for i in range(len(metric_names)):
                iter_dict[evaltype][metric_names[i]].append(metrics[i])

        with open(os.path.join(output_path, 'eval_epoch_' + str(epoch) + '.json'), 'w') as f:
            json.dump(classification_report(true, pred, output_dict=True), f, indent=2)

        return accuracy
