from scipy.stats import pearsonr, spearmanr
from sentence_transformers.losses import CosineSimilarityLoss, SoftmaxLoss
import torch
from torch import nn, Tensor
from typing import Iterable, Dict
import logging
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error


class modifiedCosineSimilarityLoss(CosineSimilarityLoss):

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        output_list = output.cpu().tolist() # list of single outputs
        labels_list = labels.cpu().tolist() # list of single labels
        pearson_cosine, _ = pearsonr(labels_list, output_list) 
        spearman_cosine, _ = spearmanr(labels_list, output_list)
        mse_cosine = mean_squared_error(labels_list, output_list)
        mae_cosine = mean_absolute_error(labels_list, output_list)

        return self.loss_fct(output, labels.view(-1)), {"pearson_cosine":[pearson_cosine], 
                                                        "spearman_cosine":[spearman_cosine],
                                                        "mse_cosine":[mse_cosine], "mae_cosine":[mae_cosine]}


logger = logging.getLogger(__name__)


class modifiedSoftmaxLoss(SoftmaxLoss):

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]  # sentence representations.
        rep_a, rep_b = reps

        vectors_concat = [] 
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1) 

        output = self.classifier(features)  # für jede Klasse ein Neuron
        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            total, correct = 0, 0
            predictions = torch.argmax(output, dim=1).cpu().numpy() 
            cpu_labels = labels.cpu().numpy()
            total += len(predictions)
            for i in range(len(predictions)):
                if predictions[i] == cpu_labels[i]:
                    correct += 1
            accuracy = correct/total
            a_precision, a_recall, a_fscore, _ = precision_recall_fscore_support(predictions, cpu_labels, average='macro', zero_division=1)
            i_precision, i_recall, i_fscore, _ = precision_recall_fscore_support(predictions, cpu_labels, average='micro', zero_division=1)
            return loss, {"accuracy":[accuracy], "macro_precision":[a_precision], "macro_recall":[a_recall], 'macro_f1':[a_fscore],
                            "micro_precision":[i_precision], "micro_recall":[i_recall], 'micro_f1':[i_fscore]}
        else:
            return reps, output
