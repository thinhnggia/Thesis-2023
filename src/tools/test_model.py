import torch

import torch.nn as nn
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from src.models.custom_layers import ZeroMaskedEntriesTF, ZeroMaskedEntries, AttentionTF, AttentionPool, TimeDistributed
from src.models.cts import CTS, build_CTS
from src.config.config import Configs
from pytorch_model_summary import summary

# Common
init_shape = (1, 4850)
readability_shape = (1, 35)
linguistic_shape = (1, 51)
pos_vocab_size = 36
maxnum = 97
maxlen = 50
readability_count = 35
linguistic_count = 51
config = Configs()
output_dim = 9

input_data = np.random.randint(36, size=init_shape)
input_readability = np.random.randn(*readability_shape)
input_linguistic = np.random.randn(*linguistic_shape)

# Pytorch
torch_input = torch.from_numpy(input_data)
torch_readability = torch.from_numpy(input_readability).float()
torch_linguistic = torch.from_numpy(input_linguistic).float()
torch_model = CTS(
    pos_vocab_size,
    maxnum,
    maxlen,
    readability_count,
    linguistic_count,
    config,
    output_dim
)
torch_output = torch_model(torch_input, torch_readability, torch_linguistic)
pytorch_total_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
model_stats = summary(torch_model, torch_input, torch_linguistic, torch_readability, show_input=True)
print("Pytorch model stats: ")
print(model_stats)

# Tensorflow
tf.random.set_seed(1)
input = layers.Input(shape=init_shape[1:])
model = build_CTS(
    pos_vocab_size,
    maxnum,
    maxlen,
    readability_count,
    linguistic_count,
    config,
    output_dim
)
tf_output = model.predict([input_data, input_linguistic, input_readability])

print("Input data shape: ", input_data.shape)
print("Tensorflow output shape: ", tf_output.shape)
print("Pytorch output shape: ", torch_output.shape)

# input_readability = np.random.randn(97, 50, 50)
# input = layers.Input(shape=(50, 50))
# output = layers.Conv1D(100, 3, padding="valid")(input)
# model = Model(inputs=input, outputs=output)
# tf_output = model(input_readability)
# import pdb; pdb.set_trace()

# input_data = np.random.randn(1, 97, 100)
# torch_data = torch.from_numpy(input_data).float()
# lstm = nn.LSTM(input_size=100, hidden_size=100, bidirectional=False, batch_first=True)
# lstm_output = lstm(torch_data)
# import pdb; pdb.set_trace()