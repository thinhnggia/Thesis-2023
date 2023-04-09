import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertConfig, BertModel

from .cts import CTS
from .prompt_tuning import BertPrefixForSequenceClassification

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(7)


class CTSBertDocumentToken(BertPreTrainedModel, CTS):
    """
    CTS model combine with Document and Token level
    """
    def __init__(self, bert_model_config: BertConfig, **kwargs):
        super(CTSBertDocumentToken, self).__init__(bert_model_config, **kwargs)
        self.bert = BertModel(bert_model_config)
        self.drop_out = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.mlp = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size * 2, 186),
        )
        self.bert_batch_size = 1
        self.mlp.apply(init_weights)

    def forward(self, pos: torch.Tensor, linguistic: torch.Tensor, readability: torch.Tensor, document_batch: torch.Tensor, device='cpu'):
        # CTS part
        pos_x = self.pos_embedding(pos)
        pos_x_maskedout = self.zero_masked_entries(pos_x)
        pos_drop_x = self.drop_out(pos_x_maskedout)
        pos_resh_W = pos_drop_x.view(-1, self.maxum, self.maxlen, self.pos_embedding_dim)
        pos_resh_W = torch.transpose(pos_resh_W, 2, 3) # Swap the length and embedding dimension
        pos_zcnn = self.time_distributed_conv(pos_resh_W)
        pos_zcnn = torch.transpose(pos_zcnn, 2, 3) # Swap the length and embedding dimension
        pos_avg_zcnn = self.time_distributed_att(pos_zcnn)
        pos_hz_lstm_list = [self.trait_lstm[index](pos_avg_zcnn)[0] for index in range(self.output_dim)]
        pos_avg_hz_lstm_list = [self.trait_att_pool[index](pos_hz_lstm) for index, pos_hz_lstm in enumerate(pos_hz_lstm_list)]
        pos_avg_hz_lstm_feat_list = [torch.cat([pos_rep, linguistic, readability], dim=1) for pos_rep in pos_avg_hz_lstm_list]
        pos_avg_hz_lstm = torch.cat([pos_rep.reshape(-1, 1, self.final_doc_dim)
                             for pos_rep in pos_avg_hz_lstm_feat_list], dim=-2)

        final_preds = []
        for index in range(self.output_dim):
            mask = torch.tensor([True for _ in range(self.output_dim)])
            mask[index] = False
            non_target_rep = pos_avg_hz_lstm[:, mask, :]
            target_rep = pos_avg_hz_lstm[:, index: index+1, :]
            att_attention, _ = self.trait_att[index](target_rep, non_target_rep)
            attention_concat = torch.cat([target_rep, att_attention], dim=-1)
            attention_concat = attention_concat.view(-1, attention_concat.size(-1))
            final_pred = torch.sigmoid(
                self.trait_dense[index](attention_concat)
            )
            final_preds.append(final_pred)
        y = torch.cat([pred for pred in final_preds], dim=-1)

        # Bert part
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                        min(document_batch.shape[1], self.bert_batch_size),
                                        self.bert.config.hidden_size * 2),
                                  dtype=torch.float, device=device)
        for doc_id in range(document_batch.shape[0]):
            all_bert_output_info = self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                             token_type_ids=document_batch[doc_id][:self.bert_batch_size, 1],
                                             attention_mask=document_batch[doc_id][:self.bert_batch_size, 2])
            bert_token_max = torch.max(all_bert_output_info[0], 1)
            bert_output[doc_id][:self.bert_batch_size] = torch.cat((bert_token_max.values, all_bert_output_info[1]), 1)

        prediction = self.mlp(bert_output.view(bert_output.shape[0], -1))
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction
    

class CTSBertSegment(BertPreTrainedModel, CTS):
    """
    CTS model combine with Segment level
    """
    def __init__(self, bert_model_config: BertConfig, **kwargs):
        super(CTSBertSegment).__init__(bert_model_config, **kwargs)
        self.bert = BertModel(bert_model_config)
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.lstm = nn.LSTM(bert_model_config.hidden_size,bert_model_config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, 1)
        )
        self.w_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, bert_model_config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, bert_model_config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)
        self.mlp.apply(init_weights)

    def forward(self, pos: torch.Tensor, linguistic: torch.Tensor, readability: torch.Tensor, document_batch: torch.Tensor, device='cpu', bert_batch_size=0):
        # CTS part
        pos_x = self.pos_embedding(pos)
        pos_x_maskedout = self.zero_masked_entries(pos_x)
        pos_drop_x = self.drop_out(pos_x_maskedout)
        pos_resh_W = pos_drop_x.view(-1, self.maxum, self.maxlen, self.pos_embedding_dim)
        pos_resh_W = torch.transpose(pos_resh_W, 2, 3) # Swap the length and embedding dimension
        pos_zcnn = self.time_distributed_conv(pos_resh_W)
        pos_zcnn = torch.transpose(pos_zcnn, 2, 3) # Swap the length and embedding dimension
        pos_avg_zcnn = self.time_distributed_att(pos_zcnn)
        pos_hz_lstm_list = [self.trait_lstm[index](pos_avg_zcnn)[0] for index in range(self.output_dim)]
        pos_avg_hz_lstm_list = [self.trait_att_pool[index](pos_hz_lstm) for index, pos_hz_lstm in enumerate(pos_hz_lstm_list)]
        pos_avg_hz_lstm_feat_list = [torch.cat([pos_rep, linguistic, readability], dim=1) for pos_rep in pos_avg_hz_lstm_list]
        pos_avg_hz_lstm = torch.cat([pos_rep.reshape(-1, 1, self.final_doc_dim)
                             for pos_rep in pos_avg_hz_lstm_feat_list], dim=-2)

        final_preds = []
        for index in range(self.output_dim):
            mask = torch.tensor([True for _ in range(self.output_dim)])
            mask[index] = False
            non_target_rep = pos_avg_hz_lstm[:, mask, :]
            target_rep = pos_avg_hz_lstm[:, index: index+1, :]
            att_attention, _ = self.trait_att[index](target_rep, non_target_rep)
            attention_concat = torch.cat([target_rep, att_attention], dim=-1)
            attention_concat = attention_concat.view(-1, attention_concat.size(-1))
            final_pred = torch.sigmoid(
                self.trait_dense[index](attention_concat)
            )
            final_preds.append(final_pred)
        y = torch.cat([pred for pred in final_preds], dim=-1)

        # Bert part
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                        min(document_batch.shape[1],
                                            bert_batch_size),
                                        self.bert.config.hidden_size), dtype=torch.float, device=device)
        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:bert_batch_size,0],
                                                                           token_type_ids=document_batch[doc_id][:bert_batch_size, 1],
                                                                           attention_mask=document_batch[doc_id][:bert_batch_size, 2])[1])
        output, (_, _) = self.lstm(bert_output.permute(1, 0, 2))
        output = output.permute(1, 0, 2) # (batch_size, seq_len, num_hiddens)
        attention_w = torch.tanh(torch.matmul(output, self.w_omega) + self.b_omega)
        attention_u = torch.matmul(attention_w, self.u_omega)
        attention_score = F.softmax(attention_u, dim=1)
        attention_hidden = output * attention_score
        attention_hidden = torch.sum(attention_hidden, dim=1)
        prediction = self.mlp(attention_hidden)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction
    

class CTSBertDocument(BertPreTrainedModel, CTS):
    """
    CTS model combine with Document level
    """
    def __init__(self, bert_model_config: BertConfig, **kwargs):
        super(CTSBertDocumentToken, self).__init__(bert_model_config, **kwargs)
        self.bert = BertModel(bert_model_config)
        self.drop_out = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

    def forward(self, pos: torch.Tensor, linguistic: torch.Tensor, readability: torch.Tensor, document_batch: torch.Tensor, device='cpu'):
        # CTS part
        pos_x = self.pos_embedding(pos)
        pos_x_maskedout = self.zero_masked_entries(pos_x)
        pos_drop_x = self.drop_out(pos_x_maskedout)
        pos_resh_W = pos_drop_x.view(-1, self.maxum, self.maxlen, self.pos_embedding_dim)
        pos_resh_W = torch.transpose(pos_resh_W, 2, 3) # Swap the length and embedding dimension
        pos_zcnn = self.time_distributed_conv(pos_resh_W)
        pos_zcnn = torch.transpose(pos_zcnn, 2, 3) # Swap the length and embedding dimension
        pos_avg_zcnn = self.time_distributed_att(pos_zcnn)
        pos_hz_lstm_list = [self.trait_lstm[index](pos_avg_zcnn)[0] for index in range(self.output_dim)]
        pos_avg_hz_lstm_list = [self.trait_att_pool[index](pos_hz_lstm) for index, pos_hz_lstm in enumerate(pos_hz_lstm_list)]
        pos_avg_hz_lstm_feat_list = [torch.cat([pos_rep, linguistic, readability], dim=1) for pos_rep in pos_avg_hz_lstm_list]
        pos_avg_hz_lstm = torch.cat([pos_rep.reshape(-1, 1, self.final_doc_dim)
                             for pos_rep in pos_avg_hz_lstm_feat_list], dim=-2)

        final_preds = []
        for index in range(self.output_dim):
            mask = torch.tensor([True for _ in range(self.output_dim)])
            mask[index] = False
            non_target_rep = pos_avg_hz_lstm[:, mask, :]
            target_rep = pos_avg_hz_lstm[:, index: index+1, :]
            att_attention, _ = self.trait_att[index](target_rep, non_target_rep)
            attention_concat = torch.cat([target_rep, att_attention], dim=-1)
            attention_concat = attention_concat.view(-1, attention_concat.size(-1))
            final_pred = torch.sigmoid(
                self.trait_dense[index](attention_concat)
            )
            final_preds.append(final_pred)
        y = torch.cat([pred for pred in final_preds], dim=-1)

        # Bert part
        return
    

class CTSPrompt(BertPreTrainedModel, CTS):
    """
    CTS combine with Prompt-tuning V2
    """
    pass