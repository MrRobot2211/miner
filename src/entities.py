from collections import OrderedDict
from typing import List, Tuple, Dict, Any
import random
import numpy as np



import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer, AutoTokenizer

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

    def __init__(self, data_name: str, tokenizer: PreTrainedTokenizer, category2id: dict,combine_type:str=None   ):
        super().__init__()
        self._name = data_name
        self._samples = OrderedDict()
        self._mode = Dataset.TRAIN_MODE
        self._tokenizer = tokenizer
        self._category2id = category2id
        self.combine_type = combine_type   

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
            return create_train_sample(sample, self._tokenizer, self._category2id,self.combine_type)
        else:
            return create_eval_sample(sample, self._tokenizer, self._category2id,self.combine_type)
        
class DatasetOnline(Dataset):
    def __init__(self, data_name: str, tokenizer: PreTrainedTokenizer, category2id: dict,aug_type='base',pad_id:int=1000,combine_type:str=None):
        super().__init__(data_name, tokenizer, category2id)
        self.aug_type = aug_type
        self.pad_id=pad_id
        self.combine_type = combine_type   
    # @staticmethod
    # def create_pre_impression(impression_id: int, user_id: int, pos_news: dict,neg_news:List[int]) -> Impression:
        
        
    #     impression = PreImpression(impression_id, user_id, pos_news,neg_news)

    #     return impression

    def add_sample(self, user_id: int, clicked_news: List[News],  pos_news: dict,neg_news:List[int],npratio:int,impression_id:int):
        sample = PreSample(self._id, user_id, clicked_news, pos_news,neg_news,npratio,impression_id)
        self._samples[self._id] = sample
        self._id += 1
   
    def _get_train_line(self,pos_news,neg_news,npratio,user_id,clicked_news,augmentations,impression_id):

        #for  i in range(len(pos_news['vanilla'])):
        label = [1] + [0] * npratio

        list_news = [pos_news[np.random.choice(augmentations)]] + sample_news(neg_news, npratio, self.pad_id)

        impression_news = list(zip(list_news, label))
        np.random.shuffle(impression_news)
        list_news, label = zip(*impression_news)
        
    
        impression = self.create_impression(impression_id, user_id, list_news, label)
        self._id += 1
        
        sample = Sample(self._id, user_id, clicked_news, impression)
        return sample
    
    def _get_train_line_hard(self,pos_news,neg_news,npratio,user_id,clicked_news,augmentations,impression_id):

        #for i in range(len(pos_news['vanilla'])):
        label = [1] + [0] * npratio

        num_to_pick = np.random.randint(1,min(len(augmentations), npratio))
        picks = np.random.choice(np.arange(len(augmentations)),num_to_pick,replace=False)
        picks = np.sort(picks)

        news = [pos_news[augmentations[pick]] for pick in picks]
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
        #print(sample.pos_news)
        augmentations = [ aug for aug in sample.pos_news.keys()]
        pos_news = sample.pos_news

        neg_news = sample.neg_news
        npratio = sample.npratio

        if self.aug_type=='base':
            sample = self._get_train_line(pos_news,neg_news,npratio,user_id,clicked_news,augmentations,impression_id)
        
        elif self.aug_type=='hard':
            sample = self._get_train_line_hard(pos_news,neg_news,npratio,user_id,clicked_news,augmentations,impression_id)

            

        if self._mode == Dataset.TRAIN_MODE:
            return create_train_sample(sample, self._tokenizer, self._category2id,self.combine_type)
        else:
            return create_eval_sample(sample, self._tokenizer, self._category2id,self.combine_type)


#TODO:change this to incorporate augmentations, do another one of this that owhen called actually samples from the possible augmentation set in some way. 
        #1- just grab an augmented version at random
        #2- output a preferred augmented version as a negative and a new augmendted version (drop a negative at random)
        #3- create new class that just outputs news values
def _create_sample(sample: Sample, tokenizer: PreTrainedTokenizer, category2id: dict,combine_type=None) -> dict:
    title_clicked_news_encoding = [news.title for news in sample.clicked_news]
    sapo_clicked_news_encoding = [news.sapo for news in sample.clicked_news]
    
    category_clicked_news_encoding = [news.category for news in sample.clicked_news]
    
    #if not aug:
    title_impression_encoding = [news.title for news in sample.impression.news]
    sapo_impression_encoding = [news.sapo for news in sample.impression.news]
    if combine_type =='pre-concat':
        title_clicked_news_encoding = [news.title + news.sapo[1:] for news in sample.clicked_news]
        title_impression_encoding = [ news.title+ news.sapo[1:] for news in sample.impression.news]
    
    category_impression_encoding = [news.category for news in sample.impression.news]
    
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






def create_train_sample(sample: Sample, tokenizer: PreTrainedTokenizer, num_category: int,combine_type: str=None) -> dict:
    return _create_sample(sample, tokenizer, num_category,combine_type)


def create_eval_sample(sample: Sample, tokenizer: PreTrainedTokenizer, num_category: int, combine_type: str=None) -> dict:
    return _create_sample(sample, tokenizer, num_category, combine_type)


def sample_news(list_news: List[News], num_news: int, pad: News) -> List:
    if len(list_news) >= num_news:
        return random.sample(list_news, k=num_news)
    else:
        return list_news + [pad] * (num_news - len(list_news))
    



# stop_words = set(stopwords.words('english'))
# word_tokenizer = RegexpTokenizer(r'\w+')

def remove_stopword(sentence):
    return ' '.join([word for word in word_tokenizer.tokenize(sentence) if word not in stop_words])

def sampling(imps, ratio=4):
    pos = []
    neg = []
    for imp in imps.split():
        if imp[-1] == '1':
            pos.append(imp)
        else:
            neg.append(imp)
    n_neg = ratio * len(pos)
    if n_neg <= len(neg):
        neg = random.sample(neg, n_neg)
    else:
        neg = random.sample(neg * (n_neg // len(neg) + 1), n_neg)
    random.shuffle(neg)
    res = pos + neg
    random.shuffle(res)
    return ' '.join(res)

class MindDataset(DatasetOnline):
    def __init__(self, data_name: str, tokenizer: PreTrainedTokenizer, 
                 category2id: dict,aug_type='base',pad_id:int=1000,combine_type:str=None,
                   mode: str = 'train',
            news_max_len: int = 20,
            hist_max_len: int = 20,
            seq_max_len: int = 300
            ) -> None:
        super(MindDataset, self).__init__(data_name, tokenizer, category2id,pad_id=pad_id,combine_type=combine_type)
       
        self._mode = mode
        
        
        self._tokenizer = tokenizer
        self._mode = mode
        self._news_max_len = news_max_len
        self._hist_max_len = hist_max_len
        self._seq_max_len = seq_max_len

       # self._examples = self.get_examples(negative_sampling=4)
        #print(self._examples.head())
        #self._news = self.process_news()
    
    def get_examples(self, 
            negative_sampling: bool = None
            ) -> Any:
        behavior_file = os.path.join(self.data_path, self._mode, 'behaviors.tsv')
        if self._split == 'small':
            df = pd.read_csv(behavior_file, sep='\t', header=None, 
                    names=['user_id', 'time', 'news_history', 'impressions'])
            df['impression_id'] = list(range(len(df)))
        else:
            df = pd.read_csv(behavior_file, sep='\t', header=None, 
                    names=['impression_id', 'user_id', 'time', 'news_history', 'impressions'])
        if self._mode == 'train':
            df = df.dropna(subset=['news_history'])
        df['news_history'] = df['news_history'].fillna('')

        if self._mode == 'train' and negative_sampling is not None:
            df['impressions'] = df['impressions'].apply(lambda x: sampling(
                    x, ratio=negative_sampling))
        df = df.drop('impressions', axis=1).join(df['impressions'].str.split(' ', 
                expand=True).stack().reset_index(level=1, drop=True).rename('impression'))
        if self._mode == 'test':
            df['news_id'] = df['impression']
            df['click'] = [-1] * len(df)
        else:
            df[['news_id', 'click']] = df['impression'].str.split('-', expand=True)
        df['click'] = df['click'].astype(int)
        return df

    def process_news(self) -> Dict[str, Any]:
        filepath = os.path.join(self.data_path, 'news_dict.pkl')
        if os.path.exists(filepath):
            print('Loading news info from', filepath)
            with open(filepath, 'rb') as fin: news = pickle.load(fin)
            return news
        news = dict()
        news = self.read_news(news, os.path.join(self.data_path, 'train'))
        news = self.read_news(news, os.path.join(self.data_path, 'dev'))
        if self._split == 'large':
            news = self.read_news(news, os.path.join(self.data_path, 'test'))

        print('Saving news info from', filepath)
        with open(filepath, 'wb') as fout: pickle.dump(news, fout)
        return news

    def read_news(self, 
            news: Dict[str, Any], 
            filepath: str,
            drop_stopword: bool = True,
            ) -> Dict[str, Any]:
        with open(os.path.join(filepath, 'news.tsv'), encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            info = dict()
            splitted = line.strip('\n').split('\t')
            news_id = splitted[0]
            if news_id in news:
                continue
            title = splitted[3].lower()
            abstract = splitted[4].lower()
            if drop_stopword:
                title = remove_stopword(title)
                abstract = remove_stopword(abstract)
            news[news_id] = dict()
            title_words = self._tokenizer.tokenize(title)
            news[news_id]['title'] = self._tokenizer.convert_tokens_to_ids(title_words)
            abstract_words = self._tokenizer.tokenize(abstract)
            news[news_id]['abstract'] = self._tokenizer.convert_tokens_to_ids(abstract_words)
        return news

    def collate(self, batch: Dict[str, Any]):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        segment_ids = torch.tensor([item['segment_ids'] for item in batch])
        input_mask = torch.tensor([item['input_mask'] for item in batch])
        news_segment_ids = torch.tensor([item['news_segment_ids'] for item in batch])
        sentence_ids = torch.tensor([item['sentence_ids'] for item in batch])
        sentence_mask = torch.tensor([item['sentence_mask'] for item in batch])
        sentence_segment_ids = torch.tensor([item['sentence_segment_ids'] for item in batch])
        inputs = {'input_ids': input_ids, 
                  'segment_ids': segment_ids, 
                  'input_mask': input_mask, 
                  'news_segment_ids': news_segment_ids, 
                  'sentence_ids': sentence_ids, 
                  'sentence_mask': sentence_mask, 
                  'sentence_segment_ids': sentence_segment_ids, 
                  }
        if self._mode == 'train':
            inputs['label'] = torch.tensor([item['label'] for item in batch],dtype=torch.float)
            return inputs 
        elif self._mode == 'dev':
            inputs['impression_id'] = [item['impression_id'] for item in batch]
            inputs['label'] = torch.tensor([item['label'] for item in batch],dtype=torch.float)
            return inputs 
        elif self._mode == 'test':
            inputs['impression_id'] = [item['impression_id'] for item in batch]
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')
        
#  title_clicked_news_encoding = [news.title for news in sample.clicked_news]
#     sapo_clicked_news_encoding = [news.sapo for news in sample.clicked_news]
    
#     category_clicked_news_encoding = [news.category for news in sample.clicked_news]
    
#     #if not aug:
#     title_impression_encoding = [news.title for news in sample.impression.news]
#     sapo_impression_encoding = [news.sapo for news in sample.impression.news]
    
    def pack_bert_features(self, example: Any):
        #curr_news = self._news[example['news_id']]['title'][:self._news_max_len]
        candidate_idx = random.randint(0,len(example.impression.news)-1)

        curr_news =example.impression.news[candidate_idx].title[:self._news_max_len]
        label = example.impression.label[candidate_idx]


        news_segment_ids = []
        hist_news = []
        sentence_ids = [0, 1, 2]
        for i, ns in enumerate(example.clicked_news[:self._hist_max_len]):
            ids = ns.title[:self._news_max_len]
            hist_news += ids
            news_segment_ids += [i + 2] * len(ids)
            sentence_ids.append(sentence_ids[-1] + 1)
        
        tmp_hist_len = self._seq_max_len-len(curr_news)-3
        hist_news = hist_news[:tmp_hist_len]
        input_ids = [self._tokenizer.cls_token_id] + curr_news + [self._tokenizer.sep_token_id] \
                    + hist_news + [self._tokenizer.sep_token_id]
        news_segment_ids = [0] + [1] * len(curr_news) + [0] + news_segment_ids[:tmp_hist_len] + [0]
        segment_ids = [0] * (len(curr_news) + 2) + [1] * (len(hist_news) + 1)
        input_mask = [1] * len(input_ids)

        padding_len = self._seq_max_len - len(input_ids)
        input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
        input_mask = input_mask + [0] * padding_len
        segment_ids = segment_ids + [0] * padding_len
        news_segment_ids = news_segment_ids + [0] * padding_len

        sentence_segment_ids = [0] * 3 + [1] * (len(sentence_ids) - 3)
        sentence_mask = [1] * len(sentence_ids)

        sentence_max_len = 3 + self._hist_max_len
        sentence_mask = [1] * len(sentence_ids)
        padding_len = sentence_max_len - len(sentence_ids)
        sentence_ids = sentence_ids + [0] * padding_len
        sentence_mask = sentence_mask + [0] * padding_len
        sentence_segment_ids = sentence_segment_ids + [0] * padding_len


        assert len(input_ids) == self._seq_max_len
        assert len(input_mask) == self._seq_max_len
        assert len(segment_ids) == self._seq_max_len
        assert len(news_segment_ids) == self._seq_max_len

        assert len(sentence_ids) == sentence_max_len
        assert len(sentence_mask) == sentence_max_len
        assert len(sentence_segment_ids) == sentence_max_len

        return input_ids, input_mask, segment_ids, news_segment_ids, \
                sentence_ids, sentence_mask, sentence_segment_ids, label

    def __getitem__(self, index: int) -> Dict[str, Any]:
        #example = self._examples.iloc[index]
        #we need to open up impressions in some way (maybe just do it at random)
        
        sample = self.samples[index//5]

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


        input_ids, input_mask, segment_ids, news_segment_ids, \
            sentence_ids, sentence_mask, sentence_segment_ids, label = self.pack_bert_features(sample)
        inputs = {'input_ids': input_ids, 
                  'segment_ids': segment_ids, 
                  'input_mask': input_mask, 
                  'news_segment_ids': news_segment_ids, 
                  'sentence_ids': sentence_ids, 
                  'sentence_mask': sentence_mask, 
                  'sentence_segment_ids': sentence_segment_ids,
                  }
        if self._mode == 'train':
            inputs['label'] = label
            return inputs 
        elif self._mode == 'eval':
            inputs['impression_id'] = impression_id
            inputs['label'] = label
            return inputs 
        elif self._mode == 'test':
            inputs['impression_id'] = impression_id
            return inputs 
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def __len__(self) -> int:
        return 5*len(self.samples)
    

class MindEvalDataset(Dataset):
    def __init__(self, data_name: str, tokenizer: PreTrainedTokenizer, 
                 category2id: dict,pad_id:int=1000,combine_type:str=None,
                   mode: str = 'train',
            news_max_len: int = 20,
            hist_max_len: int = 20,
            seq_max_len: int = 300
            ) -> None:
        super(MindEvalDataset, self).__init__(data_name, tokenizer, category2id)
       
        self._mode = mode
        
        
        self._tokenizer = tokenizer
        self._mode = mode
        self._news_max_len = news_max_len
        self._hist_max_len = hist_max_len
        self._seq_max_len = seq_max_len

       # self._examples = self.get_examples(negative_sampling=4)
        #print(self._examples.head())
        #self._news = self.process_news()
    
        self._name = data_name
        self._samples = OrderedDict()
        self._mode = Dataset.TRAIN_MODE
        self._tokenizer = tokenizer
        self._category2id = category2id
        self.combine_type = combine_type   

        self._news_id = 0
        self._id = 0


    def collate(self, batch: Dict[str, Any]):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        segment_ids = torch.tensor([item['segment_ids'] for item in batch])
        input_mask = torch.tensor([item['input_mask'] for item in batch])
        news_segment_ids = torch.tensor([item['news_segment_ids'] for item in batch])
        sentence_ids = torch.tensor([item['sentence_ids'] for item in batch])
        sentence_mask = torch.tensor([item['sentence_mask'] for item in batch])
        sentence_segment_ids = torch.tensor([item['sentence_segment_ids'] for item in batch])
        inputs = {'input_ids': input_ids, 
                  'segment_ids': segment_ids, 
                  'input_mask': input_mask, 
                  'news_segment_ids': news_segment_ids, 
                  'sentence_ids': sentence_ids, 
                  'sentence_mask': sentence_mask, 
                  'sentence_segment_ids': sentence_segment_ids, 
                  }
        # if self._mode == 'train':
        #     inputs['label'] = torch.tensor([item['label'] for item in batch])
        #     return inputs 
        #elif self._mode == 'dev':
        inputs['impression_id'] = torch.tensor([item['impression_id'] for item in batch],dtype=torch.int)
        inputs['label'] = torch.tensor([item['label'] for item in batch],dtype=torch.float)
        return inputs 
       # elif self._mode == 'test':
        #     inputs['impression_id'] = [item['impression_id'] for item in batch]
        # else:
        #     raise ValueError('Mode must be `train`, `dev` or `test`.')
        
#  title_clicked_news_encoding = [news.title for news in sample.clicked_news]
#     sapo_clicked_news_encoding = [news.sapo for news in sample.clicked_news]
    
#     category_clicked_news_encoding = [news.category for news in sample.clicked_news]
    
#     #if not aug:
#     title_impression_encoding = [news.title for news in sample.impression.news]
#     sapo_impression_encoding = [news.sapo for news in sample.impression.news]
    
    def pack_bert_features(self, example: Any):
        #curr_news = self._news[example['news_id']]['title'][:self._news_max_len]
        #candidate_idx = random.randint(0,len(example.impression.news)-1)

        curr_news =example.impression.news[0].title[:self._news_max_len]
        label = example.impression.label[0]


        news_segment_ids = []
        hist_news = []
        sentence_ids = [0, 1, 2]
        for i, ns in enumerate(example.clicked_news[:self._hist_max_len]):
            ids = ns.title[:self._news_max_len]
            hist_news += ids
            news_segment_ids += [i + 2] * len(ids)
            sentence_ids.append(sentence_ids[-1] + 1)
        
        tmp_hist_len = self._seq_max_len-len(curr_news)-3
        hist_news = hist_news[:tmp_hist_len]
        input_ids = [self._tokenizer.cls_token_id] + curr_news + [self._tokenizer.sep_token_id] \
                    + hist_news + [self._tokenizer.sep_token_id]
        news_segment_ids = [0] + [1] * len(curr_news) + [0] + news_segment_ids[:tmp_hist_len] + [0]
        segment_ids = [0] * (len(curr_news) + 2) + [1] * (len(hist_news) + 1)
        input_mask = [1] * len(input_ids)

        padding_len = self._seq_max_len - len(input_ids)
        input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
        input_mask = input_mask + [0] * padding_len
        segment_ids = segment_ids + [0] * padding_len
        news_segment_ids = news_segment_ids + [0] * padding_len

        sentence_segment_ids = [0] * 3 + [1] * (len(sentence_ids) - 3)
        sentence_mask = [1] * len(sentence_ids)

        sentence_max_len = 3 + self._hist_max_len
        sentence_mask = [1] * len(sentence_ids)
        padding_len = sentence_max_len - len(sentence_ids)
        sentence_ids = sentence_ids + [0] * padding_len
        sentence_mask = sentence_mask + [0] * padding_len
        sentence_segment_ids = sentence_segment_ids + [0] * padding_len


        assert len(input_ids) == self._seq_max_len
        assert len(input_mask) == self._seq_max_len
        assert len(segment_ids) == self._seq_max_len
        assert len(news_segment_ids) == self._seq_max_len

        assert len(sentence_ids) == sentence_max_len
        assert len(sentence_mask) == sentence_max_len
        assert len(sentence_segment_ids) == sentence_max_len

        return input_ids, input_mask, segment_ids, news_segment_ids, \
                sentence_ids, sentence_mask, sentence_segment_ids, label



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        #sample = self.samples[i]

        sample = self.samples[i]

        impression_id = sample.impression.impression_id
        user_id = sample.user_id
        clicked_news =sample.clicked_news

    


        input_ids, input_mask, segment_ids, news_segment_ids, \
            sentence_ids, sentence_mask, sentence_segment_ids, label = self.pack_bert_features(sample)
        

        inputs = {'input_ids': input_ids, 
                  'segment_ids': segment_ids, 
                  'input_mask': input_mask, 
                  'news_segment_ids': news_segment_ids, 
                  'sentence_ids': sentence_ids, 
                  'sentence_mask': sentence_mask, 
                  'sentence_segment_ids': sentence_segment_ids,
                  }
        
        inputs['impression_id'] = impression_id
        inputs['label'] = label
        return inputs 
        # elif self._mode == 'test':
        #     inputs['impression_id'] = impression_id
        #     return inputs 
        # else:
        #     raise ValueError('Mode must be `train`, `dev` or `test`.')

        
        #return create_eval_sample(sample, self._tokenizer, self._category2id,self.combine_type)