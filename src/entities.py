from collections import OrderedDict
from typing import List
import random
import numpy as np



import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer

from src import utils


class News:
    def __init__(self, news_id: str, title: List[int], sapo: List[int], category: int):
        self._news_id = news_id
        self._title = title
        self._sapo = sapo
        self._category = category

    @property
    def news_id(self) -> int:
        return self._news_id

    @property
    def title(self) -> List[int]:
        return self._title

    @property
    def sapo(self) -> List[int]:
        return self._sapo

    @property
    def category(self) -> int:
        return self._category


class NewsAug:
    def __init__(self, news_id: str, title: List[int], sapo: List[int], category: int):
        self._news_id = news_id
        self._title = title
        self._sapo = sapo
        self._category = category
        self._aug_title_sapo

    @property
    def news_id(self) -> int:
        return self._news_id

    @property
    def title(self) -> List[int]:
        return self._title

    @property
    def sapo(self) -> List[int]:
        return self._sapo

    @property
    def category(self) -> int:
        return self._category
    
    @property
    def category(self) -> List[int]:
        return self._aug_title_sapo



class Impression:
    def __init__(self, impression_id: int, user_id: int, news: List[News], label: List[int]):
        self._impression_id = impression_id
        self._user_id = user_id
        self._news = news
        self._label = label

    @property
    def impression_id(self):
        return self._impression_id

    @property
    def user_id(self) -> int:
        return self._user_id

    @property
    def news(self) -> List[News]:
        return self._news

    @property
    def label(self):
        return self._label

class PreImpression:
    def __init__(self, impression_id: int, user_id: int, pos_news: dict, neg_news :List[News],label :List[int],npratio:int):
        self._impression_id = impression_id
        self._user_id = user_id
        self._pos_news = pos_news
        self._neg_news = neg_news
        self._label = label
        self._npratio = npratio

    @property
    def impression_id(self):
        return self._impression_id

    @property
    def user_id(self) -> int:
        return self._user_id

    @property
    def pos_news(self) -> dict:
        return self._pos_news
    @property
    def neg_news(self) -> List[News]:
        return self._neg_news

    @property
    def label(self):
        return self._label
    
    @property
    def npratio(self):
        return self._npratio


class Sample:
    def __init__(self, sample_id: int, user_id: int, clicked_news: List[News], impression: Impression):
        self._sample_id = sample_id
        self._user_id = user_id
        self._clicked_news = clicked_news
        self._impression = impression

    @property
    def user_id(self) -> int:
        return self._user_id

    @property
    def clicked_news(self) -> List[News]:
        return self._clicked_news

    @property
    def impression(self) -> Impression:
        return self._impression

class PreSample:
    def __init__(self, sample_id: int, user_id: int, clicked_news: List[News], pos_news: dict, neg_news :List[News],npratio:int,impression_id:int):
        self._sample_id = sample_id
        self._user_id = user_id
        self._clicked_news = clicked_news
        self._pos_news = pos_news
        self._neg_news = neg_news
        self._npratio = npratio
        self._impression_id = impression_id

    @property
    def user_id(self) -> int:
        return self._user_id

    @property
    def clicked_news(self) -> List[News]:
        return self._clicked_news

    @property
    def impression(self) -> Impression:
        return self._impression
    @property
    def pos_news(self) -> dict:
        return self._pos_news
    @property
    def neg_news(self) -> List[News]:
        return self._neg_news
    @property
    def impression_id(self):
        return self._impression_id
    
    @property
    def npratio(self):
        return self._npratio



class Dataset(TorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, data_name: str, tokenizer: PreTrainedTokenizer, category2id: dict):
        super().__init__()
        self._name = data_name
        self._samples = OrderedDict()
        self._mode = Dataset.TRAIN_MODE
        self._tokenizer = tokenizer
        self._category2id = category2id

        self._news_id = 0
        self._id = 0

    def set_mode(self, mode: str):
        self._mode = mode

    def create_news(self, title: List[int], sapo: List[int], category: int) -> News:
        news = News(self._news_id, title, sapo, category)
        self._news_id += 1

        return news

    @staticmethod
    def create_impression(impression_id: int, user_id: int, news: List[News], label: List[int]) -> Impression:
        impression = Impression(impression_id, user_id, news, label)

        return impression

    def add_sample(self, user_id: int, clicked_news: List[News], impression: Impression):
        sample = Sample(self._id, user_id, clicked_news, impression)
        self._samples[self._id] = sample
        self._id += 1

    @property
    def samples(self) -> List[Sample]:
        return list(self._samples.values())

    @property
    def news_count(self) -> int:
        return self._news_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        sample = self.samples[i]

        if self._mode == Dataset.TRAIN_MODE:
            return create_train_sample(sample, self._tokenizer, self._category2id)
        else:
            return create_eval_sample(sample, self._tokenizer, self._category2id)
        
class DatasetOnline(Dataset):
    def __init__(self, data_name: str, tokenizer: PreTrainedTokenizer, category2id: dict,aug_type='base',pad_id:int=1000):
        super().__init__(data_name, tokenizer, category2id)
        self.aug_type = aug_type
        self.pad_id=pad_id   
    # @staticmethod
    # def create_pre_impression(impression_id: int, user_id: int, pos_news: dict,neg_news:List[int]) -> Impression:
        
        
    #     impression = PreImpression(impression_id, user_id, pos_news,neg_news)

    #     return impression

    def add_sample(self, user_id: int, clicked_news: List[News],  pos_news: dict,neg_news:List[int],npratio:int,impression_id:int):
        sample = PreSample(self._id, user_id, clicked_news, pos_news,neg_news,npratio,impression_id)
        self._samples[self._id] = sample
        self._id += 1
   
    def _get_train_line(self,pos_news,neg_news,npratio,user_id,clicked_news,augmentations,impression_id):

        for  i in range(len(pos_news['vanilla'])):
            label = [1] + [0] * npratio

            list_news = [pos_news[np.random.choice(augmentations)][i]] + sample_news(neg_news, npratio, self.pad_id)

            impression_news = list(zip(list_news, label))
            np.random.shuffle(impression_news)
            list_news, label = zip(*impression_news)
            
        
            impression = self.create_impression(impression_id, user_id, list_news, label)
            self._id += 1
            
            sample = Sample(self._id, user_id, clicked_news, impression)
            return sample
    
    def _get_train_line_hard(self,pos_news,neg_news,npratio,user_id,clicked_news,augmentations,impression_id):

        for i in range(len(pos_news['vanilla'])):
            label = [1] + [0] * npratio

            num_to_pick = np.random.randint(1,min(len(augmentations), npratio))
            picks = np.random.choice(np.arange(len(augmentations)),num_to_pick,replace=False)
            picks = np.sort(picks)

            news = [pos_news[augmentations[pick]][i] for pick in picks]
            #print(news)
            list_news = news + sample_news(neg_news, npratio+1 - num_to_pick , self.pad_id)
           # print(list_news)
            assert len(list_news) == len(label)
            impression_news = list(zip(list_news, label))
            random.shuffle(impression_news)
            list_news, label = zip(*impression_news)
            
            impression = self.create_impression(impression_id, user_id, list_news, label)
            self._id += 1
            
            sample = Sample(self._id, user_id, clicked_news, impression)
            return sample
         
    
    #@TODO maybe do smth diff here
    def __getitem__(self, i: int):
        sample = self.samples[i]

        impression_id = sample.impression_id
        user_id = sample.user_id
        clicked_news =sample.clicked_news

        augmentations = [ aug for aug in sample.pos_news.keys()]
        pos_news = sample.pos_news

        neg_news = sample.neg_news
        npratio = sample.npratio

        if self.aug_type=='base':
            sample = self._get_train_line(pos_news,neg_news,npratio,user_id,clicked_news,augmentations,impression_id)
        
        elif self.aug_type=='hard':
            sample = self._get_train_line_hard(pos_news,neg_news,npratio,user_id,clicked_news,augmentations,impression_id)

            

        if self._mode == Dataset.TRAIN_MODE:
            return create_train_sample(sample, self._tokenizer, self._category2id)
        else:
            return create_eval_sample(sample, self._tokenizer, self._category2id)


#TODO:change this to incorporate augmentations, do another one of this that owhen called actually samples from the possible augmentation set in some way. 
        #1- just grab an augmented version at random
        #2- output a preferred augmented version as a negative and a new augmendted version (drop a negative at random)
        #3- create new class that just outputs news values
def _create_sample(sample: Sample, tokenizer: PreTrainedTokenizer, category2id: dict,aug=False) -> dict:
    title_clicked_news_encoding = [news.title for news in sample.clicked_news]
    sapo_clicked_news_encoding = [news.sapo for news in sample.clicked_news]
    category_clicked_news_encoding = [news.category for news in sample.clicked_news]
    
    if not aug:
        title_impression_encoding = [news.title for news in sample.impression.news]
        sapo_impression_encoding = [news.sapo for news in sample.impression.news]
        category_impression_encoding = [news.category for news in sample.impression.news]
    else:
        #sample from available alternatives
        pass
    # Create tensor
    impression_id = torch.tensor(sample.impression.impression_id)
    title_clicked_news_encoding = utils.padded_stack(title_clicked_news_encoding, padding=tokenizer.pad_token_id)
    sapo_clicked_news_encoding = utils.padded_stack(sapo_clicked_news_encoding, padding=tokenizer.pad_token_id)
    category_clicked_news_encoding = torch.tensor(category_clicked_news_encoding)
    his_mask = (category_clicked_news_encoding != category2id['pad'])
    his_title_mask = (title_clicked_news_encoding != tokenizer.pad_token_id)
    his_sapo_mask = (sapo_clicked_news_encoding != tokenizer.pad_token_id)

    title_impression_encoding = utils.padded_stack(title_impression_encoding, padding=tokenizer.pad_token_id)
    sapo_impression_encoding = utils.padded_stack(sapo_impression_encoding, padding=tokenizer.pad_token_id)
    category_impression_encoding = torch.tensor(category_impression_encoding)
    title_mask = (title_impression_encoding != tokenizer.pad_token_id)
    sapo_mask = (sapo_impression_encoding != tokenizer.pad_token_id)

    label = torch.tensor(sample.impression.label)

    return dict(his_title=title_clicked_news_encoding, his_title_mask=his_title_mask,
                his_sapo=sapo_clicked_news_encoding, his_sapo_mask=his_sapo_mask,
                his_category=category_clicked_news_encoding, his_mask=his_mask, title=title_impression_encoding,
                title_mask=title_mask, sapo=sapo_impression_encoding, sapo_mask=sapo_mask,
                category=category_impression_encoding, impression_id=impression_id, label=label)






def create_train_sample(sample: Sample, tokenizer: PreTrainedTokenizer, num_category: int) -> dict:
    return _create_sample(sample, tokenizer, num_category)


def create_eval_sample(sample: Sample, tokenizer: PreTrainedTokenizer, num_category: int) -> dict:
    return _create_sample(sample, tokenizer, num_category)


def sample_news(list_news: List[News], num_news: int, pad: News) -> List:
    if len(list_news) >= num_news:
        return random.sample(list_news, k=num_news)
    else:
        return list_news + [pad] * (num_news - len(list_news))