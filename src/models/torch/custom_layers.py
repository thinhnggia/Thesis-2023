import torch

import torch.nn as nn
import torch.nn.functional as F


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