import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer


# ------------------------------------------------------------------------
#                              Pytorch version
# ------------------------------------------------------------------------
class AttentionPool(nn.Module):
    """
    Custom Attention Pooling layer
    """
    def __init__(self, input_shape, op='attsum', activation='tanh', init_stdev=0.01):
        super(AttentionPool, self).__init__()
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev

        self.reset_parameters(input_shape=input_shape)

    def reset_parameters(self, input_shape):
        init_val_v = (torch.randn(input_shape[2]) * self.init_stdev).type(torch.float32)
        self.att_v = nn.Parameter(init_val_v)
        init_val_W = (torch.randn(input_shape[2], input_shape[2]) * self.init_stdev).type(torch.float32)
        self.att_W = nn.Parameter(init_val_W)

    def forward(self, x, mask=None):
        y = torch.matmul(x, self.att_W)

        if not self.activation:
            weights = torch.tensordot(self.att_v, y, dims=([0], [2]))
        elif self.activation == 'tanh':
            weights = torch.tensordot(self.att_v, torch.tanh(y), dims=([0], [2]))
        
        weights = torch.softmax(weights, dim=-1)
        out = x * torch.transpose(torch.repeat_interleave(weights.unsqueeze(1), x.shape[2], dim=1), 1, 2)
        if self.op == 'attsum':
            out = torch.sum(out, dim=1)
        elif self.op == 'attmean':
            out = out.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return out.float()
    

class ZeroMaskedEntries(nn.Module):
    """
    Custom zero mask entries
    """
    def __init__(self):
        super(ZeroMaskedEntries, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x
        mask = mask.float()
        mask = mask.unsqueeze(-1)
        return x * mask
    

class TimeDistributed(nn.Module):
    """
    Custom Time Distributed Module
    """
    def __init__(self, module, batch_first=True) -> None:
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
    
    def forward(self, x):
        # Assert has at least 3 dimension
        if len(x.size()) <= 2:
            return self.module(x)
        
        input_shape = x.shape
        bs, seq_len = input_shape[0], input_shape[1]
        x_reshape = x.contiguous().view(bs * seq_len, *x.shape[2:])
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, *y.shape[1:])
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y
    

class DotProductAttention(nn.Module):
    """
    Dot product Attention
    """
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query, value):
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn

# ------------------------------------------------------------------------
#                              Tensorflow version
# ------------------------------------------------------------------------
class ZeroMaskedEntriesTF(Layer):
    """
    Zero mask tensorflow
    """
    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntriesTF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, x, mask=None):
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return x * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None

class AttentionTF(Layer):
    """
    Attention tensorflow
    """
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(AttentionTF, self).__init__(**kwargs)

    def build(self, input_shape):
        init_val_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_v = K.variable(init_val_v, name='att_v')
        init_val_W = (np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_W = K.variable(init_val_W, name='att_W')
        self.trainable_weights.append(self.att_v)
        self.trainable_weights.append(self.att_W)
        self.built = True

    def call(self, x, mask=None):
        y = K.dot(x, self.att_W)
        if not self.activation:
            weights = tf.tensordot(self.att_v, y, axes=[[0], [2]])
        elif self.activation == 'tanh':
            weights = tf.tensordot(self.att_v, K.tanh(y), axes=[[0], [2]])

        weights = K.softmax(weights)
        out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
        if self.op == 'attsum':
            out = K.sum(out, axis=1)
        elif self.op == 'attmean':
            out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(AttentionTF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))