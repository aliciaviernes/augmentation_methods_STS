from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator

import os
# from collections import OrderedDict
# from typing import Union, List
from typing import Dict, Tuple, Iterable, Type, Callable
import numpy as np
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange


class modifiedSentenceTransformer(SentenceTransformer):

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            iter_dict: dict,  
            iteration = int,  
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        """

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        self.iter_dict = iter_dict
        self.iteration = iteration

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)


        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        train_loss_step_wise = []
        
        list_of_list_contains_metric = {}

        
        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            
            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)


                    features, labels = data


                    if use_amp:
                        with autocast():
                            loss_value, outputs_as_dict = loss_model(features, labels)
                            

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        skip_scheduler = scaler.get_scale() != scale_before_step
                        loss_of_batch = loss_value.tolist()
                        train_loss_step_wise.append(loss_of_batch)  
                        if len(list_of_list_contains_metric.keys()) == 0: 
                            list_of_list_contains_metric = outputs_as_dict
                        else:
                            for k in outputs_as_dict.keys():
                                list_of_list_contains_metric[k].extend(outputs_as_dict[k])
                    else:
                        loss_value, outputs_as_dict = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()
                        loss_of_batch = loss_value.tolist() # float since it is a scalar
                        train_loss_step_wise.append(loss_of_batch)

                        if len(list_of_list_contains_metric.keys()) == 0:
                            list_of_list_contains_metric = outputs_as_dict
                        else:
                            for k in outputs_as_dict.keys():
                                list_of_list_contains_metric[k].extend(outputs_as_dict[k])

                    optimizer.zero_grad()
                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0 and steps_per_epoch != training_steps:
                    train_info = (train_loss_step_wise, list_of_list_contains_metric)
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch,
                                               global_step, callback, loss_model, train_info)
                    list_of_list_contains_metric = {}
                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()
            train_info = (train_loss_step_wise, list_of_list_contains_metric)
            self._eval_during_training(evaluator, output_path, save_best_model, epoch, global_step, callback, loss_model, train_info)
            list_of_list_contains_metric = {}

        if evaluator is None and output_path is not None: 
            self.save(output_path)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, global_step, callback, loss_model, train_info=None):
        """Runs evaluation during the training"""  # loss object 
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, global_steps=global_step, iter_dict=self.iter_dict, i=self.iteration)  # dev
            if callback is not None:
                callback(score, epoch, global_step)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)
                    if 'Softmax' in str(type(loss_model)):
                        torch.save(loss_model.classifier.state_dict(), output_path + '/bestmodel_classifier.pth') 

        if train_info is not None:
            eval_iteration_train_loss = np.mean(train_info[0])
            list_of_list_contains_metric = train_info[1]

            for key in list_of_list_contains_metric.keys():
                if 'train' not in self.iter_dict:  # train
                    self.iter_dict['train'] = {}
                    self.iter_dict['train']['loss'] = [eval_iteration_train_loss]
                    for k in list_of_list_contains_metric:
                        self.iter_dict['train'][k] = [np.mean(list_of_list_contains_metric[key])]
                else:
                    self.iter_dict['train']['loss'].append(eval_iteration_train_loss)
                    for k in list_of_list_contains_metric:
                        self.iter_dict['train'][k].append(np.mean(list_of_list_contains_metric[key]))
 