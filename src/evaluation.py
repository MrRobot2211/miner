from abc import ABC, abstractmethod
import functools
import operator
import os
from typing import List
import pickle as pk

import numpy as np
from sklearn.metrics import roc_auc_score, top_k_accuracy_score
import torch
from torch import Tensor
import torch.nn.functional as torch_f

from src.entities import Dataset


class BaseEvaluator(ABC):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.prob_predictions = []
        self.targets = []
        self._convert_targets()

    @abstractmethod
    def _convert_targets(self):
        pass

    @abstractmethod
    def _convert_pred(self):
        pass

    @abstractmethod
    def eval_batch(self, logits: Tensor, impression_ids: Tensor):
        pass

    def compute_scores(self, metrics: List[str], save_result: bool, path: str = None):
        self._convert_pred()
        print(len(self.targets))
        print(self.targets[1])
        print(self.prob_predictions[1])





        assert len(self.targets) == len(self.prob_predictions)
       
        targets = flatten(self.targets)
        prob_predictions = flatten(self.prob_predictions)
        scores = {}

        for metric in metrics:
            if metric == 'auc':
                score = roc_auc_score(y_true=targets, y_score=prob_predictions)
                scores['auc'] = score
            elif metric == 'group_auc':
                list_score = [roc_auc_score(y_true=target, y_score=prob_prediction)
                              for target, prob_prediction in zip(self.targets, self.prob_predictions)]
                scores['group_auc'] = np.nanmean(list_score)
                if save_result:
                    save_scores(os.path.join(path, 'group_auc.txt'), list_score)
            elif metric == 'mrr':
                list_score = [compute_mrr_score(np.array(target), np.array(prob_prediction))
                              for target, prob_prediction in zip(self.targets, self.prob_predictions)]
                scores['mrr'] = np.nanmean(list_score)
                if save_result:
                    save_scores(os.path.join(path, 'mrr.txt'), list_score)
            elif metric.startswith('ndcg'):
                k = int(metric.split('@')[1])
                list_score = [compute_ndcg_score(y_true=np.array(target), y_score=np.array(prob_prediction), k=k)
                              for target, prob_prediction in zip(self.targets, self.prob_predictions)]
                scores[f'ndcg@{k}'] = np.nanmean(list_score)
                if save_result:
                    save_scores(os.path.join(path, f'ndcg{k}.txt'), list_score)
            
            elif metric.startswith('hit'):
                k = int(metric.split('@')[1])
                list_score = [is_hit(y_true=np.array(target), y_score=np.array(prob_prediction), k=k)
                              for target, prob_prediction in zip(self.targets, self.prob_predictions)]
                scores[f'hit@{k}'] = np.nanmean(list_score)
                if save_result:
                    save_scores(os.path.join(path, f'hit{k}.txt'), list_score)

        return scores


class FastEvaluator(BaseEvaluator):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def _convert_targets(self):
        for sample in self.dataset.samples:
            self.targets.append(sample.impression.label)

    def _convert_pred(self):
        pass

    def eval_batch(self, logits: Tensor, impression_ids: Tensor):
        r"""
        Evaluation a batch

        Args:
            logits: tensor of shape ``(batch_size, npratio + 1)``.
            impression_ids: tensor of shape ``(batch_size, npratio + 1)``.

        Returns:
            None
        """
        probs = torch_f.softmax(logits, dim=1)
        self.prob_predictions.extend(probs.tolist())


class SlowEvaluator(BaseEvaluator):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.impression_ids = []

    def _convert_targets(self):
        group_labels = {}
        for sample in self.dataset.samples:
            impression_id = sample.impression.impression_id
            group_labels[impression_id] = group_labels.get(impression_id, []) + sample.impression.label
            
        group_labels = sorted(group_labels.items())
        # print(impression_id)
        # print(group_labels)
        
        #
       # self.group_labels = group_labels 
        #
        self.targets = [i[1] for i in group_labels]
        #print(group_labels[impression_id])
        print(len(self.targets))

    def _convert_pred(self):
        group_predictions = {}
        for prob_prediction, impression_id in zip(self.prob_predictions, self.impression_ids):
            if not isinstance(prob_prediction,list):
                prob_prediction =[prob_prediction]

            group_predictions[impression_id] = group_predictions.get(impression_id, []) + prob_prediction #[prob_prediction]

        group_predictions = sorted(group_predictions.items())
        
        #
        #self.group_predictions = group_predictions 
        #

        self.prob_predictions = [i[1] for i in group_predictions]

    def eval_batch(self, logits: Tensor, impression_ids: Tensor):
        r"""
        Evaluation a batch

        Args:
            logits: tensor of shape ``(batch_size, 1)``.
            impression_ids: tensor of shape ``(batch_size, 1)``.

        Returns:
            None
        """
        #unbert forced me to change
        
        
        probs = torch.sigmoid(logits)
        #probs = torch.softmax(logits,dim=0)
        #self.prob_predictions.extend(probs[:,1].tolist())
        
        self.prob_predictions.extend(probs.tolist())
        self.impression_ids.extend(impression_ids.tolist())

    
    def save_predictions(self,path:str):
        pred_dict = {'pred':self.prob_predictions, 'impression_id':self.impression_ids}
        pk.dump(pred_dict,open(os.path.join(path, 'preds.pkl'),'wb'))

def compute_mrr_score(y_true: np.ndarray, y_score: np.ndarray):
    r"""
    Calculate the MRR score

    Args:
        y_true: ground-truth labels.
        y_score: predicted score.

    Returns:
        MRR Score
    """
    rank = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, rank)
    rr_score = y_true / (np.arange(len(y_true)) + 1)

    return np.sum(rr_score) / np.sum(y_true)


def compute_dcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int):
    r"""
    Calculate the DCG@k score

    Args:
        y_true: ground-truth labels.
        y_score: predicted score.
        k: only consider the highest ``k`` scores in the ranking.

    Returns:
        DCG@k score
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)

    return np.sum(gains / discounts)


def compute_ndcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int):
    r"""
    Calculate the nDCG@k score

    Args:
        y_true: ground-truth labels.
        y_score: predicted score.
        k: only consider the highest ``k`` scores in the ranking.

    Returns:
        nDCG@k score
    """
    best = compute_dcg_score(y_true, y_true, k)
    actual = compute_dcg_score(y_true, y_score, k)

    return actual / best


def save_scores(path, scores):
    with open(path, mode='w', encoding='utf-8') as f:
        for score in scores:
            f.write(str(score))
            f.write('\n')



def flatten(lists: List[List]) -> List:
    return functools.reduce(operator.iconcat, lists, [])

def is_hit(y_true,y_score,k):
    
    ordered_pred = sorted(zip(y_score,y_true),key=lambda x:x[0],reverse=True)
    hit_num = sum([label  for _,label in ordered_pred[:k] ])
    return int(hit_num > 0)