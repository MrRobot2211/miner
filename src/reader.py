import csv
import random
from typing import List, Tuple
import re

from transformers import PreTrainedTokenizer

import numpy as np

from src import constants
from src.entities import Dataset, News,DatasetOnline,MindDataset,MindEvalDataset


class Reader:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_title_length: int, max_sapo_length: int, user2id: dict,
                 category2id: dict, max_his_click: int, npratio: int = None):
        self._tokenizer = tokenizer
        self._max_title_length = max_title_length
        self._max_sapo_length = max_sapo_length
        self._user2id = user2id
        self._category2id = category2id
        self._max_his_click = max_his_click
        self._npratio = npratio

    def read_train_dataset(self, data_name: str, news_path: str, behaviors_path: str,augmentations=None,aug_mode='base',online=False) -> Dataset:
        dataset, news_dataset = self._read(data_name, news_path,augmentations,online=online,gen_mode=aug_mode)
        with open(behaviors_path, mode='r', encoding='utf-8', newline='') as f:
            behaviors_tsv = csv.reader(f, delimiter='\t')
            for i, line in enumerate(behaviors_tsv):
                if not online:
                    if aug_mode =='base':
                        self._parse_train_line(i, line, news_dataset, dataset,augmentations)
                    elif aug_mode=='hard':
                        self._parse_train_line_w_hard_examples(i, line, news_dataset, dataset,augmentations)
                else:
                    self._parse_train_line_online(i, line, news_dataset, dataset,augmentations)
        
        
        return dataset

    def read_eval_dataset(self, data_name: str, news_path: str, behaviors_path: str,aug_mode='base',online=False) -> Dataset:
        if aug_mode == 'unbert':
            dataset, news_dataset = self._read(data_name, news_path,gen_mode='val_'+aug_mode,online=online)
        else:
            dataset, news_dataset = self._read(data_name, news_path,online=online)
        
        with open(behaviors_path, mode='r', encoding='utf-8', newline='') as f:
            behaviors_tsv = csv.reader(f, delimiter='\t')
            for i, line in enumerate(behaviors_tsv):
                if not online:
                    self._parse_eval_line(i, line, news_dataset, dataset)
                else:
                    self._parse_eval_line(i, line, news_dataset, dataset)


        return dataset

    def _read(self, data_name: str, news_path: str,augmentations=None,online=False,gen_mode='base') -> Tuple[Dataset, dict]:
        #@TODO allow class tto create to be defined on the fly
        
        
        
        dataset = Dataset(data_name, self._tokenizer, self._category2id)
        news_dataset = self._read_news_info(news_path, dataset)
        if online:
            if gen_mode =='unbert':
                dataset = MindDataset(data_name, self._tokenizer, self._category2id,aug_type=gen_mode,pad_id=news_dataset['pad'])
            else:
                dataset = DatasetOnline(data_name, self._tokenizer, self._category2id,aug_type=gen_mode,pad_id=news_dataset['pad'])
        
        if gen_mode =='val_unbert':
            dataset = MindEvalDataset(data_name, self._tokenizer, self._category2id,pad_id=news_dataset['pad'])
        
        #dataset = AugDataset(data_name, self._tokenizer, self._category2id)
        
        print(augmentations)
        if augmentations is not None:
            news_datasets={'vanilla':news_dataset}
            
            for aug in augmentations:
                print(f'reading {aug}')

                news_datasets[aug] = self._read_news_info(re.sub('news.tsv',aug +'_'+'news.tsv',news_path), dataset)
            
            return dataset, news_datasets

        return dataset, {'vanilla':news_dataset}

    def _read_news_info(self, news_path: str, dataset: Dataset) -> dict:
        r"""
        Read news information

        Args:
            news_path: path to TSV file containing all news information.
            dataset: Dataset object.

        Returns:
            A dictionary
        """
        
        if self._tokenizer.eos_token_id is not None:
            pad_news_obj = dataset.create_news(
                [self._tokenizer.cls_token_id, self._tokenizer.eos_token_id],
                [self._tokenizer.cls_token_id, self._tokenizer.eos_token_id], self._category2id['pad'])
        else:
            pad_news_obj = dataset.create_news(
                [self._tokenizer.cls_token_id, self._tokenizer.pad_token_id],
                [self._tokenizer.cls_token_id, self._tokenizer.pad_token_id], self._category2id['pad'])
        #print( [self._tokenizer.cls_token_id, self._tokenizer.eos_token_id])
        news_dataset = {'pad': pad_news_obj}
        with open(news_path, mode='r', encoding='utf-8', newline='') as f:
            news_tsv = csv.reader(f, delimiter='\t')
            for line in news_tsv:
                title_encoding = self._tokenizer.encode(line[constants.TITLE], add_special_tokens=True, truncation=True,
                                                        max_length=self._max_title_length)
                category_id = self._category2id.get(line[constants.CATEGORY], self._category2id['unk'])
                sapo_encoding = self._tokenizer.encode(line[constants.SAPO], add_special_tokens=True, truncation=True,
                                                       max_length=self._max_sapo_length)
                
                #TODO
                # if isinstance(dataset, DatasetAug):
                #     aug_encoding = self._tokenizer.encode(line[constants.SAPO], add_special_tokens=True, truncation=True,
                #                                        max_length=self._max_sapo_length)
                #     news = dataset.create_news(title_encoding, sapo_encoding, category_id, aug_encoding)
                # else:
                
                news = dataset.create_news(title_encoding, sapo_encoding, category_id)
                #


                news_dataset[line[constants.NEWS_ID]] = news

        return news_dataset
#TODO: Modify this to accept augmentations
    def _parse_train_line(self, impression_id, line, news_dataset, dataset,augmentations=None):
        r"""
        Parse a line of the training dataset

        Args:
            impression_id: ID of the impression.
            line: information about the impression ``(ID - User ID - Time - History - Behavior)``.
            news_dataset: a dictionary contains information about all the news ``(News ID - News object)``.
            dataset: Dataset object.

        Returns:
            None
        """
        user_id = self._user2id.get(line[constants.USER_ID], self._user2id['unk'])



        #['vanilla']
        history_clicked = [news_dataset['vanilla'][news_id] for news_id in line[constants.HISTORY].split()]
        history_clicked = [news_dataset['vanilla']['pad']] * (self._max_his_click - len(history_clicked)) + history_clicked[
                                                                                                 :self._max_his_click]
        


        if augmentations is None:
        
            pos_news = [news_dataset['vanilla'][news_id] for news_id, label in
                        [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '1']
            
        else:
            augmentations = ["vanilla"] +augmentations
            pos_news = [news_dataset[np.random.choice(augmentations)][news_id] for news_id, label in
                        [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '1']
        
        neg_news = [news_dataset['vanilla'][news_id] for news_id, label in
                    [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '0']
        if len(neg_news) ==0:
            return
        for news in pos_news:
            label = [1] + [0] * self._npratio
            list_news = [news] + sample_news(neg_news, self._npratio, news_dataset['vanilla']['pad'])

            impression_news = list(zip(list_news, label))
            random.shuffle(impression_news)
            list_news, label = zip(*impression_news)
            impression = dataset.create_impression(impression_id, user_id, list_news, label)
            dataset.add_sample(user_id, history_clicked, impression)
            

        
    def _parse_train_line_w_hard_examples(self, impression_id, line, news_dataset, dataset,augmentations=None):
        r"""
        Parse a line of the training dataset

        Args:
            impression_id: ID of the impression.
            line: information about the impression ``(ID - User ID - Time - History - Behavior)``.
            news_dataset: a dictionary contains information about all the news ``(News ID - News object)``.
            dataset: Dataset object.

        Returns:
            None
        """
        user_id = self._user2id.get(line[constants.USER_ID], self._user2id['unk'])



        #['vanilla']
        history_clicked = [news_dataset['vanilla'][news_id] for news_id in line[constants.HISTORY].split()]
        history_clicked = [news_dataset['vanilla']['pad']] * (self._max_his_click - len(history_clicked)) + history_clicked[
                                                                                                 :self._max_his_click]
        
        # choose number of samples
        # pick number of samples in augmentations order, either pick indexes in any order and then sort
        #complete wiith true negatives

        pos_news = [news_id for news_id, label in
                        [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '1']

        # if augmentations is None:
        
        #     pos_news = [news_dataset['vanilla'][news_id] for news_id, label in
        #                 [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '1']
            
        # else:
           

        #     pos_news = [news_dataset[np.random.choice(augmentations,1)][news_id] for news_id, label in
        #                 [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '1']
        
        neg_news = [news_dataset['vanilla'][news_id] for news_id, label in
                    [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '0']
        

        augmentations = ["vanilla"] +augmentations
        for news in pos_news:
            label = [1] + [0] * self._npratio
            #TODO:use something similar to sample for pretraining
            num_to_pick = np.random.randint(0,min(len(augmentations), self._npratio))
            picks = np.random.choice(np.arange(len(augmentations)),num_to_pick,replace=False)
            picks = np.sort(picks)

            news = [news_dataset[augmentations[pick]] for pick in picks]
            
            list_news = news + sample_news(neg_news, self._npratio+1 - num_to_pick , news_dataset['vanilla']['pad'])

            assert len(list_news) == len(label)
            impression_news = list(zip(list_news, label))
            random.shuffle(impression_news)
            list_news, label = zip(*impression_news)
            impression = dataset.create_impression(impression_id, user_id, list_news, label)
            dataset.add_sample(user_id, history_clicked, impression)


    def _parse_train_line_online(self, impression_id, line, news_dataset, dataset,augmentations=None):
        r"""
        Parse a line of the training dataset

        Args:
            impression_id: ID of the impression.
            line: information about the impression ``(ID - User ID - Time - History - Behavior)``.
            news_dataset: a dictionary contains information about all the news ``(News ID - News object)``.
            dataset: Dataset object.

        Returns:
            None
        """
        user_id = self._user2id.get(line[constants.USER_ID], self._user2id['unk'])



        #['vanilla']
        history_clicked = [news_dataset['vanilla'][news_id] for news_id in line[constants.HISTORY].split()]
        history_clicked = [news_dataset['vanilla']['pad']] * (self._max_his_click - len(history_clicked)) + history_clicked[
                                                                                                 :self._max_his_click]
        

        if augmentations is None:
        
            augmentations = ["vanilla"] 
            pos_news = [{ aug:news_dataset[aug][news_id] for aug in augmentations} for news_id, label in
                        [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '1'] 
            
        else:
            augmentations = ["vanilla"] +augmentations
            pos_news = [{ aug:news_dataset[aug][news_id] for aug in augmentations} for news_id, label in
                        [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '1'] 
        
        neg_news = [news_dataset['vanilla'][news_id] for news_id, label in
                    [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '0']
        
        
        #print(isinstance(dataset,MindDataset))
        if isinstance(dataset,MindDataset) and ((len(pos_news)>0) or (len(neg_news)>0)):
            dataset.add_sample(user_id, history_clicked,pos_news,neg_news,  self._npratio, impression_id)
        else:
            if (len(pos_news)>0) and (len(neg_news)>0):
                for pos_new in pos_news:
                    #print(pos_new)
                    #print(neg_news)
                    dataset.add_sample(user_id, history_clicked,pos_new,neg_news,  self._npratio, impression_id)


    def _parse_PREtrain_line_online(self, impression_id, line, news_dataset, dataset,augmentations=None):
        r"""
        Parse a line of the training dataset

        Args:
            impression_id: ID of the impression.
            line: information about the impression ``(ID - User ID - Time - History - Behavior)``.
            news_dataset: a dictionary contains information about all the news ``(News ID - News object)``.
            dataset: Dataset object.

        Returns:
            None
        """
        user_id = self._user2id.get(line[constants.USER_ID], self._user2id['unk'])



        #['vanilla']
        history_clicked = [news_dataset['vanilla'][news_id] for news_id in line[constants.HISTORY].split()]
        history_clicked = [news_dataset['vanilla']['pad']] * (self._max_his_click - len(history_clicked)) + history_clicked[
                                                                                                 :self._max_his_click]
        


        if augmentations is None:
        
            augmentations = ["vanilla"] 
            pos_news = { aug:[news_dataset[aug][news_id] for news_id, label in
                        [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '1'] for aug in augmentations}
            
        else:
            augmentations = ["vanilla"] +augmentations
            pos_news = { aug:[news_dataset[aug][news_id] for news_id, label in
                        [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '1'] for aug in augmentations}
        
        neg_news = [news_dataset['vanilla'][news_id] for news_id, label in
                    [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '0']
        
        
        
        if isinstance(dataset,MindDataset) and (len(pos_news['vanilla'])>0) or (len(neg_news)>0):
            dataset.add_sample(user_id, history_clicked,pos_news,neg_news,  self._npratio, impression_id)
        else:
            if (len(pos_news['vanilla'])>0) and (len(neg_news)>0):
                dataset.add_sample(user_id, history_clicked,pos_news,neg_news,  self._npratio, impression_id)






    def _parse_eval_line(self, impression_id, line, news_dataset, dataset):
        r"""
        Parse a line of the evaluation dataset

        Args:
            impression_id: ID of the impression.
            line: information about the impression ``(ID - User ID - Time - History - Behavior)``.
            news_dataset: a dictionary contains information about all the news ``(News ID - News object)``.
            dataset: Dataset object.

        Returns:
            None
        """
        user_id = self._user2id.get(line[constants.USER_ID], self._user2id['unk'])

        news_dataset = news_dataset['vanilla']

        history_clicked = [news_dataset[news_id] for news_id in line[constants.HISTORY].split()]
        history_clicked = [news_dataset['pad']] * (self._max_his_click - len(history_clicked)) + history_clicked[
                                                                                                 :self._max_his_click]
        for behavior in line[constants.BEHAVIOR].split():
            news_id, label = behavior.split('-')
            impression = dataset.create_impression(impression_id, user_id, [news_dataset[news_id]], [int(label)])
            dataset.add_sample(user_id, history_clicked, impression)

#GENERATE LISTS OF POSITIVE AND OF NEGATIVE IMPRESSIONS
    
def _parse_train_line_online_unbert(self, impression_id, line, news_dataset, dataset,augmentations=None):
        r"""
        Parse a line of the training dataset

        Args:
            impression_id: ID of the impression.
            line: information about the impression ``(ID - User ID - Time - History - Behavior)``.
            news_dataset: a dictionary contains information about all the news ``(News ID - News object)``.
            dataset: Dataset object.

        Returns:
            None
        """
        user_id = self._user2id.get(line[constants.USER_ID], self._user2id['unk'])



        #['vanilla']
        history_clicked = [news_dataset['vanilla'][news_id] for news_id in line[constants.HISTORY].split()]
        history_clicked = [news_dataset['vanilla']['pad']] * (self._max_his_click - len(history_clicked)) + history_clicked[
                                                                                                 :self._max_his_click]
        


        if augmentations is None:
        
            augmentations = ["vanilla"] 
            pos_news = { aug:[news_dataset[aug][news_id] for news_id, label in
                        [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '1'] for aug in augmentations}
            
        else:
            augmentations = ["vanilla"] +augmentations
            pos_news = { aug:[news_dataset[aug][news_id] for news_id, label in
                        [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '1'] for aug in augmentations}
        
        neg_news = [news_dataset['vanilla'][news_id] for news_id, label in
                    [behavior.split('-') for behavior in line[constants.BEHAVIOR].split()] if label == '0']
        
        if (len(pos_news['vanilla'])>0) and (len(neg_news)>0):
            dataset.add_sample(user_id, history_clicked,pos_news,neg_news,  self._npratio, impression_id)

        for behavior in line[constants.BEHAVIOR].split():
            news_id, label = behavior.split('-')
            impression = dataset.create_impression(impression_id, user_id, [news_dataset[news_id]], [int(label)])
            dataset.add_sample(user_id, history_clicked, impression)






def sample_news(list_news: List[News], num_news: int, pad: News) -> List:
    if len(list_news) >= num_news:
        return random.sample(list_news, k=num_news)
    else:
        return list_news + [pad] * (num_news - len(list_news))
