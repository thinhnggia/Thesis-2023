# import torch.nn.functional as F
import tensorflow.keras.backend as K

from tensorflow import keras


# def masked_loss_function(y_true, y_pred):
#     mask_value = -1
#     mask = (y_true != mask_value).float()
#     mse = F.mse_loss
#     return mse(y_true * mask, y_pred * mask)

def tf_masked_loss_function(y_true, y_pred):
    mask_value = -1
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    mse = keras.losses.MeanSquaredError()
    return mse(y_true * mask, y_pred * mask)