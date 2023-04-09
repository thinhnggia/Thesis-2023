import numpy as np
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model
from .custom_layers import ZeroMaskedEntriesTF, AttentionTF


class CTSTF(tf.keras.Model):
    def __init__(self, pos_vocab_size, maxnum, maxlen, readability_feature_count,
              linguistic_feature_count, config, output_dim, **kwargs):
        super(CTSTF, self).__init__()
        pos_embedding_dim = config.EMBEDDING_DIM
        dropout_prob = config.DROPOUT
        cnn_filters = config.CNN_FILTERS
        cnn_kernel_size = config.CNN_KERNEL_SIZE
        lstm_units = config.LSTM_UNITS
        self.output_dim = output_dim

        self.embedding = layers.Embedding(output_dim=pos_embedding_dim, input_dim=pos_vocab_size, input_length=maxnum*maxlen,
                             weights=None, mask_zero=True, name='pos_x')
        self.zero_mask = ZeroMaskedEntriesTF(name='pos_x_maskedout')
        self.drop_out = layers.Dropout(dropout_prob, name='pos_drop_x')
        self.reshape = layers.Reshape((maxnum, maxlen, pos_embedding_dim), name='pos_resh_W')
        self.time_distributed_conv = layers.TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid'), name='pos_zcnn')
        self.time_distributed_att = layers.TimeDistributed(AttentionTF(), name='pos_avg_zcnn')

        self.trait_lstm = [layers.LSTM(lstm_units, return_sequences=True) for _ in range(output_dim)]
        self.trait_att_pool = [AttentionTF() for _ in range(output_dim)]
        self.trait_contcat = [layers.Concatenate() for _ in range(output_dim)]
        self.trait_reshape = [layers.Reshape((1, lstm_units + linguistic_feature_count + readability_feature_count)) for _ in range(output_dim)]
        self.trait_att = [layers.Attention() for _ in range(output_dim)]
        self.trait_flat = [layers.Flatten() for _ in range(output_dim)]
        self.trait_dense = [layers.Dense(units=1, activation='sigmoid') for _ in range(output_dim)]
        self.final_concat = layers.Concatenate()

    def call(self, inputs, training=False):
        pos, linguistic, readability = inputs
        pos_x = self.embedding(pos)
        pos_x_maskedout = self.zero_mask(pos_x)
        pos_drop_x = self.drop_out(pos_x_maskedout, training=training)
        pos_resh_W = self.reshape(pos_drop_x)
        pos_zcnn = self.time_distributed_conv(pos_resh_W)
        pos_avg_zcnn = self.time_distributed_att(pos_zcnn)

        pos_hz_lstm_list = [self.trait_lstm[i](pos_avg_zcnn) for i in range(self.output_dim)]
        pos_avg_hz_lstm_list = [self.trait_att_pool[i](pos_hz_lstm) for i, pos_hz_lstm in enumerate(pos_hz_lstm_list)]
        pos_avg_hz_lstm_feat_list = [self.trait_contcat[i]([pos_rep, linguistic, readability]) for i, pos_rep in enumerate(pos_avg_hz_lstm_list)]
        pos_avg_hz_lstm = tf.concat([self.trait_reshape[i](pos_rep) for i, pos_rep in enumerate(pos_avg_hz_lstm_feat_list)], axis=-2)

        final_preds = []
        for index in range(self.output_dim):
            mask = np.array([True for _ in range(self.output_dim)])
            mask[index] = False
            non_target_rep = tf.boolean_mask(pos_avg_hz_lstm, mask, axis=-2)
            target_rep = pos_avg_hz_lstm[:, index:index+1]
            att_attention = self.trait_att[index]([target_rep, non_target_rep])
            attention_concat = tf.concat([target_rep, att_attention], axis=-1)
            attention_concat = self.trait_flat[index](attention_concat)
            final_pred = self.trait_dense[index](attention_concat)
            final_preds.append(final_pred)
        
        y = self.final_concat([pred for pred in final_preds])
        return y