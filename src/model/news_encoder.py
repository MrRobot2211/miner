from abc import ABC
from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel, AutoModelForCausalLM , BertModel,BertConfig ,BertPreTrainedModel
import torch.functional as F
import torch.nn.functional as fn

class NewsEncoder(ABC, RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig, apply_reduce_dim: bool, use_sapo: bool, dropout: float,
                 freeze_transformer: bool, word_embed_dim: Union[int, None] = None,
                 combine_type: Union[str, None] = None, lstm_num_layers: Union[int, None] = None,
                 lstm_dropout: Union[float, None] = None):
        r"""
        Initialization

        Args:
            config: the configuration of a ``RobertaModel``.
            apply_reduce_dim: whether to reduce the dimension of Roberta's embedding or not.
            use_sapo: whether to use sapo embedding or not.
            dropout: dropout value.
            freeze_transformer: whether to freeze Roberta weight or not.
            word_embed_dim: size of each word embedding vector if ``apply_reduce_dim``.
            combine_type: method to combine news information.
            lstm_num_layers: number of recurrent layers in LSTM.
            lstm_dropout: dropout value in LSTM.
        """
        super().__init__(config)
        self.roberta = RobertaModel(config)
        if freeze_transformer:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.apply_reduce_dim = apply_reduce_dim

        if self.apply_reduce_dim:
            assert word_embed_dim is not None
            self.reduce_dim = nn.Linear(in_features=config.hidden_size, out_features=word_embed_dim)
            self.word_embed_dropout = nn.Dropout(dropout)
            self._embed_dim = word_embed_dim
        else:
            self._embed_dim = config.hidden_size

        self.use_sapo = use_sapo
        if self.use_sapo:
            assert combine_type is not None
            self.combine_type = combine_type
            if self.combine_type == 'linear':
                self.linear_combine = nn.Linear(in_features=self._embed_dim * 2, out_features=self._embed_dim)
            elif self.combine_type == 'lstm':
                self.lstm = nn.LSTM(input_size=self._embed_dim * 2, hidden_size=self._embed_dim // 2,
                                    num_layers=lstm_num_layers, batch_first=True, dropout=lstm_dropout,
                                    bidirectional=True)
                self._embed_dim = (self._embed_dim // 2) * 2

        self.init_weights()

    def forward(self, title_encoding: Tensor, title_attn_mask: Tensor, sapo_encoding: Union[Tensor, None] = None,
                sapo_attn_mask: Union[Tensor, None] = None):
        r"""
        Forward propagation

        Args:
            title_encoding: tensor of shape ``(batch_size, title_length)``.
            title_attn_mask: tensor of shape ``(batch_size, title_length)``.
            sapo_encoding: tensor of shape ``(batch_size, sapo_length)``.
            sapo_attn_mask: tensor of shape ``(batch_size, sapo_length)``.

        Returns:
            Tensor of shape ``(batch_size, embed_dim)``
        """
        news_info = []
        # Title encoder
        title_word_embed = self.roberta(input_ids=title_encoding, attention_mask=title_attn_mask)[0]
        title_repr = title_word_embed[:, 0, :]
        if self.apply_reduce_dim:
            title_repr = self.reduce_dim(title_repr)
            title_repr = self.word_embed_dropout(title_repr)
        news_info.append(title_repr)

        #@TODO2 does not make sense to embed abstract and title separately



        # Sapo encoder
        if self.use_sapo:
            sapo_word_embed = self.roberta(input_ids=sapo_encoding, attention_mask=sapo_attn_mask)[0]
            sapo_repr = sapo_word_embed[:, 0, :]
            if self.apply_reduce_dim:
                sapo_repr = self.reduce_dim(sapo_repr)
                sapo_repr = self.word_embed_dropout(sapo_repr)
            news_info.append(sapo_repr)

            if self.combine_type == 'linear':
                news_info = torch.cat(news_info, dim=1)

                return self.linear_combine(news_info)
            elif self.combine_type == 'lstm':
                news_info = torch.cat(news_info, dim=1)
                news_repr, _ = self.lstm(news_info)

                return news_repr
        else:
            return title_repr

    @property
    def embed_dim(self):
        return self._embed_dim




class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((fn.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = fn.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)
    


class NewsEncoderMoe(ABC, BertPreTrainedModel):
    def __init__(self, config: BertConfig, apply_reduce_dim: bool, use_sapo: bool, dropout: float,
                 freeze_transformer: bool, word_embed_dim: Union[int, None] = None,
                 combine_type: Union[str, None] = None, lstm_num_layers: Union[int, None] = None,
                 lstm_dropout: Union[float, None] = None,pretrained_moe_path:str =None):
        r"""
        Initialization

        Args:
            config: the configuration of a ``RobertaModel``.
            apply_reduce_dim: whether to reduce the dimension of Roberta's embedding or not.
            use_sapo: whether to use sapo embedding or not.
            dropout: dropout value.
            freeze_transformer: whether to freeze Roberta weight or not.
            word_embed_dim: size of each word embedding vector if ``apply_reduce_dim``.
            combine_type: method to combine news information.
            lstm_num_layers: number of recurrent layers in LSTM.
            lstm_dropout: dropout value in LSTM.
        """
        super().__init__(config)
        self.roberta = BertModel(config)


        # assert self.train_stage in [
        #     'pretrain', 'inductive_ft', 'transductive_ft'
        # ], f'Unknown train stage: [{self.train_stage}]'

        # if self.train_stage in ['pretrain', 'inductive_ft']:
        #     self.item_embedding = None
        #     # for `transductive_ft`, `item_embedding` is defined in SASRec base model
        # if self.train_stage in ['inductive_ft', 'transductive_ft']:
        #     # `plm_embedding` in pre-train stage will be carried via dataloader
        #     self.plm_embedding = copy.deepcopy(dataset.plm_embedding)

        # plm_suffix: feat1CLS
        # plm_suffix_aug: feat2CLS
        # train_stage: transductive_ft  # pretrain / inductive_ft / transductive_ft
        # plm_size: 768
        # adaptor_dropout_prob: 0.2
        # adaptor_layers: [768,300]
        # temperature: 0.07
        # n_exps: 8
        cfig = {  "adaptor_layers": [768,300],
         "temperature": 0.07,
         "n_exps": 8,
         "adaptor_dropout_prob": 0.2
         }
        
        self.moe_adaptor = MoEAdaptorLayer(
            cfig['n_exps'],
            cfig['adaptor_layers'],
            cfig['adaptor_dropout_prob']
        )

        # if pretrained_moe_path is not None:
        #     self.moe_adaptor.load_state_dict(torch.load(pretrained_moe_path)) 
        


        if freeze_transformer:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.apply_reduce_dim = False

        if self.apply_reduce_dim:
            assert word_embed_dim is not None
            self.reduce_dim = nn.Linear(in_features=config.hidden_size, out_features=word_embed_dim)
            self.word_embed_dropout = nn.Dropout(dropout)
            self._embed_dim = word_embed_dim
        else:
            self._embed_dim = config.hidden_size


        self._embed_dim = cfig['adaptor_layers'][1]

        self.use_sapo = use_sapo
        if self.use_sapo:
            assert combine_type is not None
            self.combine_type = combine_type
            if self.combine_type == 'linear':
                self.linear_combine = nn.Linear(in_features=self._embed_dim * 2, out_features=self._embed_dim)
            elif self.combine_type == 'lstm':
                self.lstm = nn.LSTM(input_size=self._embed_dim * 2, hidden_size=self._embed_dim // 2,
                                    num_layers=lstm_num_layers, batch_first=True, dropout=lstm_dropout,
                                    bidirectional=True)
                self._embed_dim = (self._embed_dim // 2) * 2

        self.init_weights()
        # state_dict= torch.load('unisrec_pretrained_weights/unisrec_pretained_state_dict.pth',map_location='cpu')
        # self.load_state_dict(state_dict=state_dict,strict=False)
        # print("loaded")
        
    def forward(self, title_encoding: Tensor, title_attn_mask: Tensor, sapo_encoding: Union[Tensor, None] = None,
                sapo_attn_mask: Union[Tensor, None] = None):
        r"""
        Forward propagation

        Args:
            title_encoding: tensor of shape ``(batch_size, title_length)``.
            title_attn_mask: tensor of shape ``(batch_size, title_length)``.
            sapo_encoding: tensor of shape ``(batch_size, sapo_length)``.
            sapo_attn_mask: tensor of shape ``(batch_size, sapo_length)``.

        Returns:
            Tensor of shape ``(batch_size, embed_dim)``
        """
        news_info = []
        # Title encoder
       # if self.combine_type == "pre-concat":

        
        title_word_embed = self.roberta(input_ids=title_encoding, attention_mask=title_attn_mask)[0]
        title_repr = self.moe_adaptor(title_word_embed[:, 0, :])
        if self.apply_reduce_dim:
            title_repr = self.reduce_dim(title_repr)
            title_repr = self.word_embed_dropout(title_repr)
        news_info.append(title_repr)
        
        if self.combine_type== "pre-concat":
            return title_repr

        #@TODO2 does not make sense to embed abstract and title separately

        # TODO create another class or a function to support cases in which embeddings are pre-computed

        # Sapo encoder
        if self.use_sapo:
            sapo_word_embed = self.roberta(input_ids=sapo_encoding, attention_mask=sapo_attn_mask)[0]
            sapo_repr = self.moe_adaptor(sapo_word_embed[:, 0, :])
            if self.apply_reduce_dim:
                sapo_repr = self.reduce_dim(sapo_repr)
                sapo_repr = self.word_embed_dropout(sapo_repr)
            news_info.append(sapo_repr)

            if self.combine_type == 'linear':
                news_info = torch.cat(news_info, dim=1)

                return self.linear_combine(news_info)
            elif self.combine_type == 'lstm':
                news_info = torch.cat(news_info, dim=1)
                news_repr, _ = self.lstm(news_info)

                return news_repr
        else:
            return title_repr
        


    @property
    def embed_dim(self):
        return self._embed_dim







#TODO newsencoder for Fastformer should have an attention layer on top, instead of choosing CLS

# class NewsEncoderUni(ABC, ):
#     def __init__(self, config: RobertaConfig, apply_reduce_dim: bool, use_sapo: bool, dropout: float,
#                  freeze_transformer: bool, word_embed_dim: Union[int, None] = None,
#                  combine_type: Union[str, None] = None, lstm_num_layers: Union[int, None] = None,
#                  lstm_dropout: Union[float, None] = None):
#         r"""
#         Initialization

#         Args:
#             config: the configuration of a ``RobertaModel``.
#             apply_reduce_dim: whether to reduce the dimension of Roberta's embedding or not.
#             use_sapo: whether to use sapo embedding or not.
#             dropout: dropout value.
#             freeze_transformer: whether to freeze Roberta weight or not.
#             word_embed_dim: size of each word embedding vector if ``apply_reduce_dim``.
#             combine_type: method to combine news information.
#             lstm_num_layers: number of recurrent layers in LSTM.
#             lstm_dropout: dropout value in LSTM.
#         """
#         super().__init__(config)
#         self.roberta = RobertaModel(config)

        

#         self.model = AutoModelForCausalLM.from_pretrained("microsoft/unilm-base-cased")

#         if freeze_transformer:
#             for param in self.roberta.parameters():
#                 param.requires_grad = False

#         self.apply_reduce_dim = apply_reduce_dim

#         if self.apply_reduce_dim:
#             assert word_embed_dim is not None
#             self.reduce_dim = nn.Linear(in_features=config.hidden_size, out_features=word_embed_dim)
#             self.word_embed_dropout = nn.Dropout(dropout)
#             self._embed_dim = word_embed_dim
#         else:
#             self._embed_dim = config.hidden_size

#         self.use_sapo = use_sapo
#         if self.use_sapo:
#             assert combine_type is not None
#             self.combine_type = combine_type
#             if self.combine_type == 'linear':
#                 self.linear_combine = nn.Linear(in_features=self._embed_dim * 2, out_features=self._embed_dim)
#             elif self.combine_type == 'lstm':
#                 self.lstm = nn.LSTM(input_size=self._embed_dim * 2, hidden_size=self._embed_dim // 2,
#                                     num_layers=lstm_num_layers, batch_first=True, dropout=lstm_dropout,
#                                     bidirectional=True)
#                 self._embed_dim = (self._embed_dim // 2) * 2

#         self.init_weights()

#     def forward(self, title_encoding: Tensor, title_attn_mask: Tensor, sapo_encoding: Union[Tensor, None] = None,
#                 sapo_attn_mask: Union[Tensor, None] = None):
#         r"""
#         Forward propagation

#         Args:
#             title_encoding: tensor of shape ``(batch_size, title_length)``.
#             title_attn_mask: tensor of shape ``(batch_size, title_length)``.
#             sapo_encoding: tensor of shape ``(batch_size, sapo_length)``.
#             sapo_attn_mask: tensor of shape ``(batch_size, sapo_length)``.

#         Returns:
#             Tensor of shape ``(batch_size, embed_dim)``
#         """
#         news_info = []
#         # Title encoder
#         title_word_embed = self.roberta(input_ids=title_encoding, attention_mask=title_attn_mask)[0]
#         title_repr = title_word_embed[:, 0, :]
#         if self.apply_reduce_dim:
#             title_repr = self.reduce_dim(title_repr)
#             title_repr = self.word_embed_dropout(title_repr)
#         news_info.append(title_repr)

#         #@TODO2 does not make sense to embed abstract and title separately



#         # Sapo encoder
#         if self.use_sapo:
#             sapo_word_embed = self.roberta(input_ids=sapo_encoding, attention_mask=sapo_attn_mask)[0]
#             sapo_repr = sapo_word_embed[:, 0, :]
#             if self.apply_reduce_dim:
#                 sapo_repr = self.reduce_dim(sapo_repr)
#                 sapo_repr = self.word_embed_dropout(sapo_repr)
#             news_info.append(sapo_repr)

#             if self.combine_type == 'linear':
#                 news_info = torch.cat(news_info, dim=1)

#                 return self.linear_combine(news_info)
#             elif self.combine_type == 'lstm':
#                 news_info = torch.cat(news_info, dim=1)
#                 news_repr, _ = self.lstm(news_info)

#                 return news_repr
#         else:
#             return title_repr

#     @property
#     def embed_dim(self):
#         return self._embed_dim
