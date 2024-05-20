from abc import ABC, abstractmethod

from torch import Tensor
import torch.nn.functional as torch_f

from src.utils import pairwise_cosine_similarity


class AbstractLoss(ABC):
    @abstractmethod
    def compute(self, *args, **kwargs):
        pass


class Loss(AbstractLoss):
    def __init__(self, criterion):
        self._criterion = criterion

    def compute_vanilla(self, logits: Tensor, labels: Tensor):
        if labels.dim()>1:
            targets = labels.argmax(dim=1)
        else:
            targets=labels
        rank_loss = self._criterion(logits, targets)
        return rank_loss

    def compute(self, poly_attn: Tensor, logits: Tensor, labels: Tensor):
        r"""
        Compute batch loss

        Args:
            poly_attn: tensor of shape ``(batch_size, num_context_codes, embed_dim)``.
            logits: tensor of shape ``(batch_size, npratio + 1)``.
            labels: a one-hot tensor of shape ``(batch_size, npratio + 1)``.

        Returns:
            Loss value
        """
        disagreement_loss = pairwise_cosine_similarity(poly_attn, poly_attn, zero_diagonal=True).mean()
        targets = labels.argmax(dim=1)
        rank_loss = self._criterion(logits, targets)
        total_loss = disagreement_loss + rank_loss

        return total_loss
    
   
    def compute_vanilla_eval_loss(self,logits: Tensor, labels: Tensor):
        """
        Compute loss for evaluation phase

        Args:
            poly_attn: tensor of shape ``(batch_size, num_context_codes, embed_dim)``.
            logits: tensor of shape ``(batch_size, 1)``.
            labels: a binary tensor of shape ``(batch_size, 1)``.

        Returns:
            Loss value
        """
        if labels.dim()==1:
            rank_loss = self._criterion(logits, labels)
        else:    
            rank_loss = -(torch_f.logsigmoid(logits) * labels).sum()
        total_loss =  rank_loss

        return total_loss.item()
    

    @staticmethod
    def compute_eval_loss(poly_attn: Tensor, logits: Tensor, labels: Tensor):
        """
        Compute loss for evaluation phase

        Args:
            poly_attn: tensor of shape ``(batch_size, num_context_codes, embed_dim)``.
            logits: tensor of shape ``(batch_size, 1)``.
            labels: a binary tensor of shape ``(batch_size, 1)``.

        Returns:
            Loss value
        """
        disagreement_loss = pairwise_cosine_similarity(poly_attn, poly_attn, zero_diagonal=True).mean()
        rank_loss = -(torch_f.logsigmoid(logits) * labels).sum()
        total_loss = disagreement_loss + rank_loss

        return total_loss.item()

    def compute_pretrain(self,embs):
        
        positive = embs[:,0,:].unsqueeze(1) 
        augmentations = embs[:,1:4,:]
        negatives = embs[:,4:,:]
        

        main_distance = pairwise_cosine_similarity(positive,negatives).sum()
        aug_distance = pairwise_cosine_similarity(positive,augmentations).sum()

        #print(main_distance)
        return - (main_distance + 0.001*aug_distance)