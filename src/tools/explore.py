import torch

import torch.nn as nn
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from pytorch_model_summary import summary
from transformers import AutoModel, BertModel

from src.models.torch.cts import CTS
from src.models.tensorflow.cts import CTSTF
from src.models.tensorflow.cts_bert import CTSBertTF
from src.config.config import Configs

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

# # Pytorch
# torch_input = torch.from_numpy(input_data).long()
# torch_readability = torch.from_numpy(input_readability).float()
# torch_linguistic = torch.from_numpy(input_linguistic).float()
# torch_model = CTS(
#     pos_vocab_size,
#     maxnum,
#     maxlen,
#     readability_count,
#     linguistic_count,
#     config,
#     output_dim
# )
# torch_output = torch_model(torch_input, torch_readability, torch_linguistic)
# pytorch_total_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
# model_stats = summary(torch_model, torch_input, torch_linguistic, torch_readability, show_input=True)
# print("Pytorch model stats: ")
# print(model_stats)

# Tensorflow
# tf.random.set_seed(1)
# input_ids_shape = (1, 512)
# attention_masks_shape = (1, 512)
# input_ids = np.random.randint(4000, size=input_ids_shape)
# attention_masks = np.random.randint(1, size=attention_masks_shape)
# model = CTSTF(
#     pos_vocab_size,
#     maxnum,
#     maxlen,
#     readability_count,
#     linguistic_count,
#     config,
#     output_dim
# )
# model = CTSBertTF.from_pretrained(
#     "bert-base-uncased", 
#     pos_vocab_size=pos_vocab_size, 
#     maxnum=maxnum,
#     maxlen=maxlen,
#     readability_feature_count=readability_count,
#     linguistic_feature_count=linguistic_count,
#     configs=config,
    # output_dim=output_dim)
# tf_output = model(input_ids, attention_masks)