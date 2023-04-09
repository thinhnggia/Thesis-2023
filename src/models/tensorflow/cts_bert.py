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