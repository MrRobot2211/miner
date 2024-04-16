
from abc import ABC
from typing import Union
import copy 
import math

import torch
from torch import Tensor
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel, AutoModelForCausalLM  
import torch.functional as F
import torch.nn.functional as fn

from src.model.news_encoder import NewsEncoder
from src.utils import pairwise_cosine_similarity

# class SequentialRecommender(AbstractRecommender):
#     """
#     This is a abstract sequential recommender. All the sequential model should implement This class.
#     """

#     type = ModelType.SEQUENTIAL

#     def __init__(self, config, dataset):
#         super(SequentialRecommender, self).__init__()

#         # load dataset info
#         self.USER_ID = config["USER_ID_FIELD"]
#         self.ITEM_ID = config["ITEM_ID_FIELD"]
#         self.ITEM_SEQ = self.ITEM_ID + config["LIST_SUFFIX"]
#         self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
#         self.POS_ITEM_ID = self.ITEM_ID
#         self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
#         self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
#         self.n_items = dataset.num(self.ITEM_ID)

#         # load parameters info
#         self.device = config["device"]

#     def gather_indexes(self, output, gather_index):
#         """Gathers the vectors at the specific positions over a minibatch"""
#         gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
#         output_tensor = output.gather(dim=1, index=gather_index)
#         return output_tensor.squeeze(1)

#     def get_attention_mask(self, item_seq, bidirectional=False):
#         """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
#         attention_mask = item_seq != 0
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
#         if not bidirectional:
#             extended_attention_mask = torch.tril(
#                 extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
#             )
#         extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
#         return extended_attention_mask
    


class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    


class TransformerEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class UniSRec(nn.Module): #SASRec
    def __init__(self, news_encoder: NewsEncoder,args ):
        super().__init__()
        cfig =dict(
        n_layers = 2
        ,n_heads = 2
        ,hidden_size = 300
        ,inner_size = 256
        ,hidden_dropout_prob = 0.5
        ,attn_dropout_prob = 0.5
        ,hidden_act = 'gelu'
        ,layer_norm_eps = 1e-12
        ,initializer_range = 0.02
        ,loss_type = "CE"
        ,train_stage = "pretrain"
        ,transform = "ptm_emb"
        ,ptm_size = 768
        ,adaptor_dropout_prob = 0.2
        ,adaptor_layers = [768, 300]
        ,temperature = 0.07
        ,n_exps = 8
        ,pretrain_epochs = 300
        ,save_step = 50
        ,ddp = True
        ,rank = 0
        ,world_size = 2
        # MODEL_INPUT_TYPE = InputType.POINTWISE
        # eval_type = EvaluatorType.RANKING
        # device = cuda
        ,max_his_len=50
        ,train_neg_sample_args = {'strategy': 'none'}
        ,eval_neg_sample_args = {'strategy': 'full', 'distribution': 'uniform'})
       
        #state_dict= torch.load('unisrec_pretrained_weights/unisrec_pretained_state_dict.pth',map_location='cpu')
        #self.load_state_dict(state_dict=state_dict,strict=False)

        self.train_stage = cfig['train_stage']
        self.temperature = cfig['temperature']
        #self.lam = config['lambda']
    

            #load parameters info
        self.n_layers = cfig["n_layers"]
        self.n_heads = cfig["n_heads"]
        self.hidden_size = cfig["hidden_size"]  # same as embedding_size
        self.inner_size = cfig[    "inner_size"]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = cfig["hidden_dropout_prob"]
        self.attn_dropout_prob = cfig["attn_dropout_prob"]
        self.hidden_act = cfig["hidden_act"]
        self.layer_norm_eps = cfig["layer_norm_eps"]

        self.initializer_range = cfig["initializer_range"]
        self.loss_type = cfig["loss_type"]

        # define layers and loss
        # self.item_embedding = nn.Embedding(
        #     self.n_items, self.hidden_size, padding_idx=0
        # )
        self.news_encoder = news_encoder
        self.position_embedding = nn.Embedding(cfig['max_his_len'], self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # parameters initialization
       # self.apply(self._init_weights)
        state_dict= torch.load('unisrec_pretrained_weights/unisrec_pretained_state_dict.pth',map_location='cpu')
        self.load_state_dict(state_dict=state_dict,strict=False)
        #if freeze_transformer:
        for n,param in self.named_parameters():
            if 'moe' not in n.lower():
                param.requires_grad = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()




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
        #print(history_repr.shape)

        
        position_ids = torch.arange(history_repr.size(1), dtype=torch.long, device=history_repr.device)
        position_ids = position_ids.unsqueeze(0) #.expand_as(history_repr)
        

        position_embedding = self.position_embedding(position_ids)

        # print(history_repr.shape)
        # print(position_ids.shape)
        # print(position_embedding.shape)

        history_repr = history_repr + position_embedding
        
        history_repr = self.LayerNorm(history_repr)
        history_repr = self.dropout(history_repr)
        #print(history_repr.shape)
        his_mask = self.get_attention_mask(his_mask)
        # print(history_repr.shape)
        # print(his_mask.shape)

        user_embeddings = self.trm_encoder(hidden_states=history_repr, attention_mask=his_mask, output_all_encoded_layers=False)

        #print(user_embeddings[0].shape)
        output = user_embeddings[0][:,0,:]
        #output = self.gather_indexes(output, his_length - 1)




       




        # Click predictor
        #TODO revise if prior to this we were not supposed to use the CLS token which would then limit the dimensions after this (taking max over interests dos not make sense)
        matching_scores = torch.matmul(candidate_repr, output.unsqueeze(-1)).squeeze(-1)
        # if self.score_type == 'max':
        #     matching_scores = matching_scores.max(dim=2)[0]
        # elif self.score_type == 'mean':
        #     matching_scores = matching_scores.mean(dim=2)
        # elif self.score_type == 'weighted':
        #     matching_scores = self.target_aware_attn(query=multi_user_interest, key=candidate_repr,
        #                                              value=matching_scores)
        # else:
        #     raise ValueError('Invalid method of aggregating matching score')
       # print(matching_scores.shape)
        return  matching_scores
    

    # def forward(self, item_seq, item_emb, item_seq_len):
        
        
        
        # position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        # position_embedding = self.position_embedding(position_ids)

        # input_emb = item_emb + position_embedding
        # if self.train_stage == 'transductive_ft':
        #     input_emb = input_emb + self.item_embedding(item_seq)
        # input_emb = self.LayerNorm(input_emb)
        # input_emb = self.dropout(input_emb)

        # extended_attention_mask = self.get_attention_mask(item_seq)

        # trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        # output = trm_output[-1]
        # output = self.gather_indexes(output, item_seq_len - 1)
        # return output  # [B H]

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask
    
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