import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np

from transformers import TFBertPreTrainedModel, BertConfig, TFBertMainLayer
from transformers.modeling_tf_utils import TFSequenceClassificationLoss, get_initializer, TFModelInputType
from typing import Optional, Tuple, Union

from .cts import CTSTF


class CTSBertTF(CTSTF):
    def __init__(self, base_model, base_config, **kwargs):
        super().__init__(**kwargs)

        self.bert = base_model
        self.num_labels = base_config.num_labels
        classifier_dropout = (
            base_config.classifier_dropout if base_config.classifier_dropout is not None else base_config.hidden_dropout_prob
        )
        self.bert_dropout = layers.Dropout(rate=classifier_dropout)
        self.bert_dense = layers.Dense(
            units=372, # Currently hard code for the final dimension
            kernel_initializer=get_initializer(base_config.initializer_range),
            name="bert_classifier"
        )

    def call(
        self,
        inputs,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False
    ):
        pos, linguistic, readability, input_ids, attention_mask, token_type_ids = inputs

        # Bert part
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training
        )
        bert_pooled_outputs = bert_outputs[1]
        bert_pooled_outputs = self.bert_dropout(inputs=bert_pooled_outputs, training=training)
        bert_doc_embed = self.bert_dense(inputs=bert_pooled_outputs) # Output representation of a document with Bert

        # CTS part
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
            
            # final_rep = tf.add(bert_doc_embed, attention_concat) # Add two representation from bert and cts
            final_rep = tf.concat([bert_doc_embed, attention_concat], axis=-1)
            final_pred = self.trait_dense[index](final_rep)
            final_preds.append(final_pred)
        
        y = self.final_concat([pred for pred in final_preds])
        return y


class CTSBertSegmentTF(CTSTF):
    def __init__(self, base_model, base_config, **kwargs):
        super().__init__(**kwargs)
        
        # Bert base

        self.bert = base_model
        self.num_labels = base_config.num_labels
        classifier_dropout = (
            base_config.classifier_dropout if base_config.classifier_dropout is not None else base_config.hidden_dropout_prob
        )
        self.bert_dropout = layers.Dropout(rate=classifier_dropout, name="bert_dropout")
        self.lstm = layers.LSTM(units=base_config.hidden_size, return_sequences=True, name="bert_lstm")
        self.mlp = tf.keras.Sequential([
            layers.Dropout(rate=base_config.hidden_dropout_prob, name="mlp_dropout"),
            layers.Dense(
                units=372, # Currently hard code for the final dimension
                kernel_initializer="glorot_uniform",
                name="bert_classifier"
            )
        ], name="mlp")

        # Pooling layer
        self.w_omega = tf.Variable(tf.random.uniform([base_config.hidden_size, base_config.hidden_size], -0.1, 0.1), name="w_1")
        self.b_omega = tf.Variable(tf.random.uniform([1, base_config.hidden_size], -0.1, 0.1), name="b_1")
        self.u_omega = tf.Variable(tf.random.uniform([base_config.hidden_size, 1], -0.1, 0.1), name="u_1")

    def call(
        self,
        inputs,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False
    ):
        pos, linguistic, readability, segment_1, segment_2, segment_3, segment_4 = inputs

        # Bert part
        output_segment_list = []
        for segment in [segment_1, segment_2, segment_3, segment_4]:
            input_ids = segment[:, :, 0, :]
            token_type_ids = segment[:, :, 1, :]
            attention_mask = segment[:, :, 2, :]

            seq_num = tf.shape(input_ids)[1]
            seq_len = tf.shape(input_ids)[-1]

            # Reshape to shape (batch_size * seq_num, seq_len)
            reshaped_input_ids = tf.reshape(input_ids, [-1, seq_len])
            reshaped_token_type_ids = tf.reshape(token_type_ids, [-1, seq_len])
            reshaped_attention_mask = tf.reshape(attention_mask, [-1, seq_len])

            bert_outputs = self.bert_dropout(self.bert(
                                    reshaped_input_ids, 
                                    token_type_ids=reshaped_token_type_ids,
                                    attention_mask=reshaped_attention_mask,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    inputs_embeds=inputs_embeds,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict,
                                    training=training
                                )[1], training=training)
            # Reshape back to shape (batch_size, seq_num, bert_dim)
            bert_outputs = tf.reshape(bert_outputs, [-1, seq_num, bert_outputs.shape[-1]])

            output = self.lstm(bert_outputs)
            attention_w = tf.tanh(tf.matmul(output, self.w_omega) + self.b_omega)
            attention_u = tf.matmul(attention_w, self.u_omega) # (batch_size, seq_len, 1)
            attention_score = tf.nn.softmax(attention_u, axis=1) # (batch_size, seq_len, 1)
            attention_hidden = output * attention_score # (batch_size, seq_len, num_hiddens)
            attention_hidden = tf.reduce_sum(attention_hidden, axis=1) # 加权求和 (batch_size, num_hiddens)
            mlp_out = self.mlp(attention_hidden)
            output_segment_list.append(mlp_out)
        bert_segment_embed = tf.add_n(output_segment_list)

        # CTS part
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
            
            # final_rep = tf.add(bert_doc_embed, attention_concat) # Add two representation from bert and cts
            final_rep = tf.concat([bert_segment_embed, attention_concat], axis=-1)
            final_pred = self.trait_dense[index](final_rep)
            final_preds.append(final_pred)
        
        y = self.final_concat([pred for pred in final_preds])
        return y