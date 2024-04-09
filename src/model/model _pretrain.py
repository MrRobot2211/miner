from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as torch_f

from src.model.news_encoder import NewsEncoder
from src.utils import pairwise_cosine_similarity


#TODO: change this to add augmented training, and by enhancement/topic change finetuning
class Miner(nn.Module):
    r"""
    Implementation of Multi-interest matching network for news recommendation. Please see the paper in
    https://aclanthology.org/2022.findings-acl.29.pdf.
    """
    def __init__(self, news_encoder: NewsEncoder, use_category_bias: bool, num_context_codes: int,
                 context_code_dim: int, score_type: str, dropout: float, num_category: Union[int, None] = None,
                 category_embed_dim: Union[int, None] = None, category_pad_token_id: Union[int, None] = None,
                 category_embed: Union[Tensor, None] = None):
        r"""
        Initialization

        Args:
            news_encoder: NewsEncoder object.
            use_category_bias: whether to use Category-aware attention weighting.
            num_context_codes: the number of attention vectors ``K``.
            context_code_dim: the number of features in a context code.
            score_type: the ways to aggregate the ``K`` matching scores as a final user click score ('max', 'mean' or
                'weighted').
            dropout: dropout value.
            num_category: the size of the dictionary of categories.
            category_embed_dim: the size of each category embedding vector.
            category_pad_token_id: ID of the padding token type in the category vocabulary.
            category_embed: pre-trained category embedding.
        """
        super().__init__()
        self.news_encoder = news_encoder
        self.news_embed_dim = self.news_encoder.embed_dim
        self.use_category_bias = use_category_bias
        if self.use_category_bias:
            self.category_dropout = nn.Dropout(dropout)
            if category_embed is not None:
                self.category_embedding = nn.Embedding.from_pretrained(category_embed, freeze=False,
                                                                       padding_idx=category_pad_token_id)
                self.category_embed_dim = category_embed.shape[1]
            else:
                assert num_category is not None
                self.category_embedding = nn.Embedding(num_embeddings=num_category, embedding_dim=category_embed_dim,
                                                       padding_idx=category_pad_token_id)
                self.category_embed_dim = category_embed_dim

        self.poly_attn = PolyAttention(in_embed_dim=self.news_embed_dim, num_context_codes=num_context_codes,
                                       context_code_dim=context_code_dim)
        self.score_type = score_type
        if self.score_type == 'weighted':
            self.target_aware_attn = TargetAwareAttention(self.news_embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, title: Tensor, title_mask: Tensor, his_title: Tensor, his_title_mask: Tensor,
                his_mask: Tensor, sapo: Union[Tensor, None] = None, sapo_mask: Union[Tensor, None] = None,
                his_sapo: Union[Tensor, None] = None, his_sapo_mask: Union[Tensor, None] = None,
                category: Union[Tensor, None] = None, his_category: Union[Tensor, None] = None):
        r"""
        Forward propagation

        Args:
            title: tensor of shape ``(batch_size, num_candidates, title_length)``.
            title_mask: tensor of shape ``(batch_size, num_candidates, title_length)``.
            his_title: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            his_title_mask: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            his_mask: tensor of shape ``(batch_size, num_clicked_news)``.
            sapo: tensor of shape ``(batch_size, num_candidates, sapo_length)``.
            sapo_mask: tensor of shape ``(batch_size, num_candidates, sapo_length)``.
            his_sapo: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            his_sapo_mask: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            category: tensor of shape ``(batch_size, num_candidates)``.
            his_category: tensor of shape ``(batch_size, num_clicked_news)``.

        Returns:
            tuple
                - multi_user_interest: tensor of shape ``(batch_size, num_context_codes, embed_dim)``
                - matching_scores: tensor of shape ``(batch_size, num_candidates)``
        """
        batch_size = title.shape[0]
        num_candidates = title.shape[1]
        his_length = his_title.shape[1]

        # Representation of candidate news
        title = title.view(batch_size * num_candidates, -1)
        title_mask = title_mask.view(batch_size * num_candidates, -1)
        sapo = sapo.view(batch_size * num_candidates, -1)
        sapo_mask = sapo_mask.view(batch_size * num_candidates, -1)

        candidate_repr = self.news_encoder(title_encoding=title, title_attn_mask=title_mask, sapo_encoding=sapo,
                                           sapo_attn_mask=sapo_mask)
        candidate_repr = candidate_repr.view(batch_size, num_candidates, -1)
        
        #@TODO: in order to do pretraining take candidate repr and force separations


        # Representation of history clicked news
        his_title = his_title.view(batch_size * his_length, -1)
        his_title_mask = his_title_mask.view(batch_size * his_length, -1)
        his_sapo = his_sapo.view(batch_size * his_length, -1)
        his_sapo_mask = his_sapo_mask.view(batch_size * his_length, -1)

        history_repr = self.news_encoder(title_encoding=his_title, title_attn_mask=his_title_mask,
                                         sapo_encoding=his_sapo, sapo_attn_mask=his_sapo_mask)
        history_repr = history_repr.view(batch_size, his_length, -1)

        if self.use_category_bias:
            his_category_embed = self.category_embedding(his_category)
            his_category_embed = self.category_dropout(his_category_embed)
            candidate_category_embed = self.category_embedding(category)
            candidate_category_embed = self.category_dropout(candidate_category_embed)

            #may have to alter this 
            category_bias = pairwise_cosine_similarity(his_category_embed, candidate_category_embed)

            multi_user_interest = self.poly_attn(embeddings=history_repr, attn_mask=his_mask, bias=category_bias)
        else:
            multi_user_interest = self.poly_attn(embeddings=history_repr, attn_mask=his_mask, bias=None)

        # Click predictor
        matching_scores = torch.matmul(candidate_repr, multi_user_interest.permute(0, 2, 1))
        if self.score_type == 'max':
            matching_scores = matching_scores.max(dim=2)[0]
        elif self.score_type == 'mean':
            matching_scores = matching_scores.mean(dim=2)
        elif self.score_type == 'weighted':
            matching_scores = self.target_aware_attn(query=multi_user_interest, key=candidate_repr,
                                                     value=matching_scores)
        else:
            raise ValueError('Invalid method of aggregating matching score')

        return multi_user_interest, matching_scores


class PolyAttention(nn.Module):
    r"""
    Implementation of Poly attention scheme that extracts `K` attention vectors through `K` additive attentions
    """
    def __init__(self, in_embed_dim: int, num_context_codes: int, context_code_dim: int):
        r"""
        Initialization

        Args:
            in_embed_dim: The number of expected features in the input ``embeddings``
            num_context_codes: The number of attention vectors ``K``
            context_code_dim: The number of features in a context code
        """
        super().__init__()
        self.linear = nn.Linear(in_features=in_embed_dim, out_features=context_code_dim, bias=False)
        self.context_codes = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_context_codes, context_code_dim),
                                                                  gain=nn.init.calculate_gain('tanh')))

    def forward(self, embeddings: Tensor, attn_mask: Tensor, bias: Tensor = None):
        r"""
        Forward propagation

        Args:
            embeddings: tensor of shape ``(batch_size, his_length, embed_dim)``
            attn_mask: tensor of shape ``(batch_size, his_length)``
            bias: tensor of shape ``(batch_size, his_length, num_candidates)``

        Returns:
            A tensor of shape ``(batch_size, num_context_codes, embed_dim)``
        """
        proj = torch.tanh(self.linear(embeddings))
        if bias is None:
            weights = torch.matmul(proj, self.context_codes.T)
        else:
            bias = bias.mean(dim=2).unsqueeze(dim=2)
            weights = torch.matmul(proj, self.context_codes.T) + bias
        weights = weights.permute(0, 2, 1)
        weights = weights.masked_fill_(~attn_mask.unsqueeze(dim=1), 1e-30)
        weights = torch_f.softmax(weights, dim=2)
        poly_repr = torch.matmul(weights, embeddings)

        return poly_repr


class TargetAwareAttention(nn.Module):
    """Implementation of target-aware attention network"""
    def __init__(self, embed_dim: int):
        r"""
        Initialization

        Args:
            embed_dim: The number of features in query and key vectors
        """
        super().__init__()
        self.linear = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        r"""
        Forward propagation

        Args:
            query: tensor of shape ``(batch_size, num_context_codes, embed_dim)``
            key: tensor of shape ``(batch_size, num_candidates, embed_dim)``
            value: tensor of shape ``(batch_size, num_candidates, num_context_codes)``

        Returns:
            tensor of shape ``(batch_size, num_candidates)``
        """
        proj = torch_f.gelu(self.linear(query))
        weights = torch_f.softmax(torch.matmul(key, proj.permute(0, 2, 1)), dim=2)
        outputs = torch.mul(weights, value).sum(dim=2)

        return outputs




class FastFormer(nn.Module):
    r" FastrFormer  user encoder model implementation"

     def __init__(self, news_encoder: NewsEncoder, use_category_bias: bool, num_context_codes: int,
                 context_code_dim: int, score_type: str, dropout: float, num_category: Union[int, None] = None,
                 category_embed_dim: Union[int, None] = None, category_pad_token_id: Union[int, None] = None,
                 category_embed: Union[Tensor, None] = None):
        r"""
        Initialization

        Args:
            news_encoder: NewsEncoder object.
            use_category_bias: whether to use Category-aware attention weighting.
            num_context_codes: the number of attention vectors ``K``.
            context_code_dim: the number of features in a context code.
            score_type: the ways to aggregate the ``K`` matching scores as a final user click score ('max', 'mean' or
                'weighted').
            dropout: dropout value.
            num_category: the size of the dictionary of categories.
            category_embed_dim: the size of each category embedding vector.
            category_pad_token_id: ID of the padding token type in the category vocabulary.
            category_embed: pre-trained category embedding.
        """
        super().__init__()
        self.news_encoder = news_encoder
        self.news_embed_dim = self.news_encoder.embed_dim
        self.use_category_bias = use_category_bias
        if self.use_category_bias:
            self.category_dropout = nn.Dropout(dropout)
            if category_embed is not None:
                self.category_embedding = nn.Embedding.from_pretrained(category_embed, freeze=False,
                                                                       padding_idx=category_pad_token_id)
                self.category_embed_dim = category_embed.shape[1]
            else:
                assert num_category is not None
                self.category_embedding = nn.Embedding(num_embeddings=num_category, embedding_dim=category_embed_dim,
                                                       padding_idx=category_pad_token_id)
                self.category_embed_dim = category_embed_dim

        self.fast_attn = FastformerEncoder(fast_config)
        self.score_type = score_type
        if self.score_type == 'weighted':
            self.target_aware_attn = TargetAwareAttention(self.news_embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, title: Tensor, title_mask: Tensor, his_title: Tensor, his_title_mask: Tensor,
                his_mask: Tensor, sapo: Union[Tensor, None] = None, sapo_mask: Union[Tensor, None] = None,
                his_sapo: Union[Tensor, None] = None, his_sapo_mask: Union[Tensor, None] = None,
                category: Union[Tensor, None] = None, his_category: Union[Tensor, None] = None):
        r"""
        Forward propagation

        Args:
            title: tensor of shape ``(batch_size, num_candidates, title_length)``.
            title_mask: tensor of shape ``(batch_size, num_candidates, title_length)``.
            his_title: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            his_title_mask: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            his_mask: tensor of shape ``(batch_size, num_clicked_news)``.
            sapo: tensor of shape ``(batch_size, num_candidates, sapo_length)``.
            sapo_mask: tensor of shape ``(batch_size, num_candidates, sapo_length)``.
            his_sapo: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            his_sapo_mask: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            category: tensor of shape ``(batch_size, num_candidates)``.
            his_category: tensor of shape ``(batch_size, num_clicked_news)``.

        Returns:
            tuple
                - multi_user_interest: tensor of shape ``(batch_size, num_context_codes, embed_dim)``
                - matching_scores: tensor of shape ``(batch_size, num_candidates)``
        """
        batch_size = title.shape[0]
        num_candidates = title.shape[1]
        his_length = his_title.shape[1]

        # Representation of candidate news
        title = title.view(batch_size * num_candidates, -1)
        title_mask = title_mask.view(batch_size * num_candidates, -1)
        sapo = sapo.view(batch_size * num_candidates, -1)
        sapo_mask = sapo_mask.view(batch_size * num_candidates, -1)

        candidate_repr = self.news_encoder(title_encoding=title, title_attn_mask=title_mask, sapo_encoding=sapo,
                                           sapo_attn_mask=sapo_mask)
        candidate_repr = candidate_repr.view(batch_size, num_candidates, -1)
        
        #@TODO: in order to do pretraining take candidate repr and force separations


        # Representation of history clicked news
        his_title = his_title.view(batch_size * his_length, -1)
        his_title_mask = his_title_mask.view(batch_size * his_length, -1)
        his_sapo = his_sapo.view(batch_size * his_length, -1)
        his_sapo_mask = his_sapo_mask.view(batch_size * his_length, -1)

        history_repr = self.news_encoder(title_encoding=his_title, title_attn_mask=his_title_mask,
                                         sapo_encoding=his_sapo, sapo_attn_mask=his_sapo_mask)
        history_repr = history_repr.view(batch_size, his_length, -1)
        
        multi_user_interest = self.fast_attn(embeddings=history_repr, attn_mask=his_mask)

        # Click predictor
        matching_scores = torch.matmul(candidate_repr, multi_user_interest.permute(0, 2, 1))
        if self.score_type == 'max':
            matching_scores = matching_scores.max(dim=2)[0]
        elif self.score_type == 'mean':
            matching_scores = matching_scores.mean(dim=2)
        elif self.score_type == 'weighted':
            matching_scores = self.target_aware_attn(query=multi_user_interest, key=candidate_repr,
                                                     value=matching_scores)
        else:
            raise ValueError('Invalid method of aggregating matching score')

        return multi_user_interest, matching_scores



class AttentionPooling(nn.Module):
    def __init__(self, config):
        self.config = config
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.att_fc2 = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
                
    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  
        return x

class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super(FastSelfAttention, self).__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.attention_head_size = int(config.hidden_size /config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim= config.hidden_size
        
        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
                
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask):
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5


       # print(query_for_score.shape)
       # print(attention_mask.shape)
      # print(hidden_states.shape)
        # add attention mask
        query_for_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads*self.attention_head_size)
        pooled_query_repeat= pooled_query.repeat(1, seq_len,1)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer=mixed_key_layer* pooled_query_repeat
        
        query_key_score=(self.key_att(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)
        
        # add attention mask
        query_key_score +=attention_mask

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        #query = value
        weighted_value =(pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer
      
        return weighted_value
    

class FastAttention(nn.Module):
    def __init__(self, config):
        super(FastAttention, self).__init__()
        self.self = FastSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class FastformerLayer(nn.Module):
    def __init__(self, config):
        super(FastformerLayer, self).__init__()
        self.attention = FastAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
class FastformerEncoder(nn.Module):
    def __init__(self, config, pooler_count=1):
        super(FastformerEncoder, self).__init__()
        self.config = config
        self.encoders = nn.ModuleList([FastformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # support multiple different poolers with shared bert encoder.
        self.poolers = nn.ModuleList()
        if config.pooler_type == 'weightpooler':
            for _ in range(pooler_count):
                self.poolers.append(AttentionPooling(config))
        logging.info(f"This model has {len(self.poolers)} poolers.")

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, 
                input_embs, 
                attention_mask, 
                pooler_index=0):
        #input_embs: batch_size, seq_len, emb_dim
        #attention_mask: batch_size, seq_len, emb_dim

        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        batch_size, seq_length, emb_dim = input_embs.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embs.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embs + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        #print(embeddings.size())
        
        all_hidden_states = [embeddings]

        for i, layer_module in enumerate(self.encoders):
            layer_outputs = layer_module(all_hidden_states[-1], extended_attention_mask)
            all_hidden_states.append(layer_outputs)
        assert len(self.poolers) > pooler_index
        output = self.poolers[pooler_index](all_hidden_states[-1], attention_mask)
        
      #  print(embeddings.size())

        return output 


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec


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
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class UniSRec(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']

        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None
            # for `transductive_ft`, `item_embedding` is defined in SASRec base model
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            # `plm_embedding` in pre-train stage will be carried via dataloader
            self.plm_embedding = copy.deepcopy(dataset.plm_embedding)

        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob']
        )

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        if self.train_stage == 'transductive_ft':
            input_emb = input_emb + self.item_embedding(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction):
        pos_items_emb = self.moe_adaptor(interaction['pos_item_emb'])
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def seq_seq_contrastive_task(self, seq_output, same_pos_id, interaction):
        item_seq_aug = interaction[self.ITEM_SEQ + '_aug']
        item_seq_len_aug = interaction[self.ITEM_SEQ_LEN + '_aug']
        item_emb_list_aug = self.moe_adaptor(interaction['item_emb_list_aug'])
        seq_output_aug = self.forward(item_seq_aug, item_emb_list_aug, item_seq_len_aug)
        seq_output_aug = F.normalize(seq_output_aug, dim=1)

        pos_logits = (seq_output * seq_output_aug).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, seq_output_aug.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def pretrain(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(interaction['item_emb_list'])
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)

        # Remove sequences with the same next item
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))

        loss_seq_item = self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)
        loss_seq_seq = self.seq_seq_contrastive_task(seq_output, same_pos_id, interaction)
        loss = loss_seq_item + self.lam * loss_seq_seq
        return loss

    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)

        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_item_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_item_emb = test_item_emb + self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(test_item_emb, dim=1)

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_items_emb = test_items_emb + self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores