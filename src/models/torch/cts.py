import torch

import numpy as np
import torch.nn as nn
import tensorflow.keras.layers as layers
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf

from tensorflow.keras import Model

from src.config.config import Configs
from .custom_layers import (
    ZeroMaskedEntries, 
    AttentionPool, 
    TimeDistributed, 
    DotProductAttention
)


class CTS(nn.Module):
    """
    CTS Pytorch
    """
    def __init__(self, pos_vocab_size, maxnum, maxlen, readability_count, linguistic_count, 
                 config: Configs, output_dim, **kwargs) -> None:
        super(CTS, self).__init__()
        dropout_prob = config.DROPOUT
        cnn_filters = config.CNN_FILTERS
        cnn_kernel_size = config.CNN_KERNEL_SIZE
        lstm_units = config.LSTM_UNITS
        self.maxnum = maxnum
        self.maxlen = maxlen
        self.pos_embedding_dim = config.EMBEDDING_DIM

        self.pos_embedding = nn.Embedding(pos_vocab_size, self.pos_embedding_dim, padding_idx=0)
        self.drop_out = nn.Dropout(dropout_prob)
        self.share_conv = nn.Conv1d(in_channels=self.pos_embedding_dim, out_channels=cnn_filters, kernel_size=cnn_kernel_size)
        self.time_distributed_conv = TimeDistributed(module=self.share_conv)
        self.att_pool_1 = AttentionPool(input_shape=(1, 46, 100)) # TODO: Hard code
        self.time_distributed_att = TimeDistributed(module=self.att_pool_1)
        self.zero_masked_entries = ZeroMaskedEntries()

        self.linguistic_linear = nn.Linear(linguistic_count, lstm_units)
        self.readability_linear = nn.Linear(readability_count, lstm_units)

        self.trait_lstm = nn.ModuleList([
            nn.LSTM(input_size=cnn_filters, hidden_size=lstm_units, bidirectional=False, batch_first=True)
            for _ in range(output_dim)
        ])
        self.trait_att_pool = nn.ModuleList([
            AttentionPool(input_shape=(1, 97, 100)) for _ in range(output_dim)
        ])
        self.trait_att = nn.ModuleList([
            DotProductAttention() for _ in range(output_dim)
        ])
        self.trait_dense = nn.ModuleList([
            nn.Linear(372, 1) for _ in range(output_dim) # Concat target_rep and att_attention. TODO: Hard code
        ])

        # Make global some properties
        self.output_dim = output_dim
        self.final_doc_dim = lstm_units + linguistic_count + readability_count
        
    def forward(self, pos, linguistic, readability):
        pos_x = self.pos_embedding(pos)
        pos_x_maskedout = self.zero_masked_entries(pos_x)
        pos_drop_x = self.drop_out(pos_x_maskedout)
        pos_resh_W = pos_drop_x.contiguous().view(-1, self.maxnum, self.maxlen, self.pos_embedding_dim)
        pos_resh_W = torch.swapaxes(pos_resh_W, 2, 3) # Swap the length and embedding dimension
        pos_zcnn = self.time_distributed_conv(pos_resh_W)
        pos_zcnn = torch.swapaxes(pos_zcnn, 2, 3).contiguous() # Swap the length and embedding dimension
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

        return y