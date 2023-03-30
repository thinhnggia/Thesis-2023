import torch.nn.functional as F

def masked_loss_function(y_true, y_pred):
    mask_value = -1
    mask = (y_true != mask_value).float()
    mse = F.mse_loss
    return mse(y_true * mask, y_pred * mask)