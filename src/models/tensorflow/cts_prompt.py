import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np

from transformers import TFBertModel


class PrefixEncoderTF(tf.keras.layers.Layer):
    """
    Prefix encoder
    """
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = tf.keras.layers.Embedding(config.pre_seq_len, config.hidden_size)
            self.dense_1 = tf.keras.layers.Dense(config.prefix_hidden_size, trainable=True)
            self.activation = tf.keras.layers.Activation('tanh')
            self.dense_2 = tf.keras.layers.Dense(config.num_hidden_layers * 2 * config.hidden_size, trainable=True)
        else:
            self.embedding = tf.keras.layers.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def call(self, prefix):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            out = self.dense_1(prefix_tokens)
            out = self.activation(out)
            past_key_values = self.dense_2(out)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class TFBertPrefix(tf.keras.Model):
    """
    Prompt-tuning V2
    """
    def __init__(self, config, freeze=True):
        super().__init__()
        # Bert 
        self.num_labels = config.num_labels
        self.config = config
        self.bert = TFBertModel.from_pretrained(config.model_name_or_path)
        # TODO: Currently hard code dropout rate
        self.dropout = layers.Dropout(rate=0.1)
        
        if freeze: # Freeze pretrained Bert
            for layer in self.bert.layers:
                layer.trainable = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        # Prefix
        self.prefix_tokens = tf.range(self.pre_seq_len, dtype=tf.int32)
        self.prefix_encoder = PrefixEncoderTF(config)

    def get_prompt(self, batch_size, training):
        prefix_tokens = tf.expand_dims(self.prefix_tokens, axis=0)
        prefix_tokens = tf.tile(prefix_tokens, [batch_size, 1])
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = tf.reshape(past_key_values, [batch_size, self.pre_seq_len, self.n_layer * 2, self.n_head, self.n_embd])
        past_key_values = self.dropout(past_key_values, training=training)
        past_key_values = tf.transpose(past_key_values, perm=[2, 0, 3, 1, 4])
        past_key_values = tf.split(past_key_values, num_or_size_splits=12, axis=0)
        return past_key_values
    
    def call(self, 
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            training=False
        ):        
        batch_size = tf.shape(input_ids)[0]
        past_key_values = self.get_prompt(batch_size=batch_size, training=training)
        prefix_attention_mask = tf.ones([batch_size, self.pre_seq_len], dtype=tf.int32)
        attention_mask = tf.concat([prefix_attention_mask, attention_mask], axis=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training
        )

        return outputs