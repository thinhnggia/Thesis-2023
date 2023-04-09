from .torch.cts import CTS
from .torch.cts_bert import CTSBertDocument, CTSBertDocumentToken, CTSBertSegment, CTSPrompt
from .tensorflow.cts import CTSTF
from .tensorflow.cts_bert import CTSBertTF
from .tensorflow.cts_prompt import TFBertPrefix
from transformers import TFBertModel, BertConfig
from tensorflow.keras.layers import Input
from tensorflow.keras import Model


def get_torch_model(model_name, **kwargs):
    """
    Get pytorch model
    """
    if model_name == 'cts':
        return CTS(**kwargs)
    elif model_name == 'cts_bert':
        return CTSBertDocument(**kwargs)
    elif model_name == 'cts_bert_token':
        return CTSBertDocumentToken(**kwargs)
    elif model_name == 'cts_bert_segment':
        return CTSBertSegment(**kwargs)
    elif model_name == 'cts_prompt':
        return CTSPrompt(**kwargs)
    else:
        raise ValueError(f'Model {model_name} not found')


def get_tf_model(model_name, pretrain_name="bert-base-uncased", freeze=True, omit_num=None, **kwargs):
    """
    Get tensorflow model
    """
    if model_name == 'cts':
        temp_model = CTSTF(**kwargs)
        dummy_inputs = [
            Input(shape=(4850), dtype="int64"), 
            Input(shape=(51)), 
            Input(shape=(35)), 
        ]
        outputs = temp_model(dummy_inputs)
        model = Model(inputs=dummy_inputs, outputs=outputs)
        model.summary(expand_nested=True)
        return model
    elif model_name == "cts_bert":
        base_model = TFBertModel.from_pretrained(pretrain_name)
        if freeze:
            freeze_layers = base_model.layers if not omit_num else base_model.layers[:-omit_num]
            for layer in freeze_layers:
                layer.trainable = False
        base_config = BertConfig.from_pretrained(pretrain_name)
        temp_model = CTSBertTF(base_model, base_config, **kwargs)
        dummy_inputs = [
            Input(shape=(4850), dtype="int32"), 
            Input(shape=(51)), 
            Input(shape=(35)), 
            Input(shape=(512), dtype="int32"), 
            Input(shape=(512), dtype="int32"), 
            Input(shape=(512), dtype="int32")]
        outputs = temp_model(dummy_inputs)
        model = Model(inputs=dummy_inputs, outputs=outputs)
        model.summary(expand_nested=True)
        return model
    elif model_name == "cts_bert_prompt":
        # Initialize configuration
        base_config = BertConfig.from_pretrained(pretrain_name)
        base_config.pre_seq_len = 20
        base_config.model_name_or_path = pretrain_name
        base_config.prefix_projection = False
        base_config.prefix_hidden_size = 512

        base_model = TFBertPrefix(base_config, freeze=freeze)
        temp_model = CTSBertTF(base_model, base_config, **kwargs)
        dummy_inputs = [
            Input(shape=(4850), dtype="int32"), 
            Input(shape=(51)), 
            Input(shape=(35)), 
            Input(shape=(512), dtype="int32"), 
            Input(shape=(512), dtype="int32"), 
            Input(shape=(512), dtype="int32")]
        outputs = temp_model(dummy_inputs)
        model = Model(inputs=dummy_inputs, outputs=outputs)
        model.summary(expand_nested=True)
        return model
    else:
        raise ValueError(f'Model {model_name} not supported')