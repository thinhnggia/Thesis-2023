import os

# from .torch.cts import CTS
# from .torch.cts_bert import CTSBertDocument, CTSBertDocumentToken, CTSBertSegment, CTSPrompt
from .tensorflow.cts import CTSTF
from .tensorflow.cts_bert import CTSBertTF, CTSBertSegmentTF
from .tensorflow.cts_prompt import TFBertPrefix
from .tensorflow.mod_bert import ModTFBertModel
from .tensorflow.custom_cts import ModCTSBertTF, ModSplitCTSBertTF
from transformers import TFBertModel, BertConfig
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model


# def get_torch_model(model_name, **kwargs):
#     """
#     Get pytorch model
#     """
#     if model_name == 'cts':
#         return CTS(**kwargs)
#     elif model_name == 'cts_bert':
#         return CTSBertDocument(**kwargs)
#     elif model_name == 'cts_bert_token':
#         return CTSBertDocumentToken(**kwargs)
#     elif model_name == 'cts_bert_segment':
#         return CTSBertSegment(**kwargs)
#     elif model_name == 'cts_prompt':
#         return CTSPrompt(**kwargs)
#     else:
#         raise ValueError(f'Model {model_name} not found')


def get_prompt_tuning_model(pretrain_name, freeze):
    base_config = BertConfig.from_pretrained(pretrain_name)
    base_config.pre_seq_len = 40
    base_config.model_name_or_path = pretrain_name
    base_config.prefix_projection = False
    base_config.prefix_hidden_size = 512

    base_model = TFBertPrefix(base_config, freeze=freeze)
    dummy_inputs = [
        Input(shape=(512), dtype="int32"), 
        Input(shape=(512), dtype="int32"), 
        Input(shape=(512), dtype="int32")]
    outputs = base_model(input_ids=dummy_inputs[0],
        attention_mask=dummy_inputs[1],
        token_type_ids=dummy_inputs[2])
    x = Dropout(rate=base_config.hidden_dropout_prob)(outputs[1])
    outputs = Dense(units=1, activation="sigmoid")(x)
    model = Model(inputs=dummy_inputs, outputs=outputs)
    return model


def get_tf_model(args, model_name, pretrain_name="bert-base-uncased", freeze=True, omit_num=None, **kwargs):
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
        base_config.pre_seq_len = 40
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
        
        config = kwargs.get("config")
        if config.PRETRAIN_BERT:
            pretrain_bert_path = os.path.join(config.OUTPUT_PATH, "prompt_tuning", f"current_model_prompt_{args.test_prompt_id}.h5")
            base_clone_model = get_prompt_tuning_model(pretrain_name, freeze)
            base_clone_model.load_weights(pretrain_bert_path)
            base_layer_dict = dict([(layer.name, layer) for layer in base_clone_model.layers])
            print("Base layer dict: ", base_layer_dict)

            print("Loaded layers:")
            print("===================")
            for layer in model.layers[-1].layers:
                layer_name = layer.name
                if layer_name in "tf_bert_prefix_1":
                    layer.set_weights(base_layer_dict["tf_bert_prefix_1"].get_weights())
                    print(layer_name)
            print("===================")
            print("Loaded weights from clone to new model: ", pretrain_bert_path)

        return model
    elif model_name == "cts_bert_segment":
        # Initialize configuration
        base_model = TFBertModel.from_pretrained(pretrain_name)
        if freeze:
            freeze_layers = base_model.layers if not omit_num else base_model.layers[:-omit_num]
            for layer in freeze_layers:
                layer.trainable = False
        base_config = BertConfig.from_pretrained(pretrain_name)
        temp_model = CTSBertSegmentTF(base_model, base_config, **kwargs)
        dummy_inputs = [
            Input(shape=(4850), dtype="int32"), 
            Input(shape=(51)), 
            Input(shape=(35)), 
            Input(shape=(None, 3, 90), dtype="int32", name="segment_1"), # Chunk size 1
            Input(shape=(None, 3, 30), dtype="int32", name="segment_2"),  # Chunk size 2
            Input(shape=(None, 3, 130), dtype="int32", name="segment_3"), # Chunk size 3
            Input(shape=(None, 3, 10), dtype="int32", name="segment_4"), # Chunk size 4
        ]
        outputs = temp_model(dummy_inputs)
        model = Model(inputs=dummy_inputs, outputs=outputs)
        model.summary(expand_nested=True)
        return model
    elif model_name == "prompt_tuning":
        return get_prompt_tuning_model(pretrain_name, freeze)
    elif model_name == "custom_cts_bert_prompt":
        # Initialize configuration
        base_config = BertConfig.from_pretrained(pretrain_name)
        base_config.pre_seq_len = 40
        base_config.model_name_or_path = pretrain_name
        base_config.prefix_projection = False
        base_config.prefix_hidden_size = 512
        base_config.pos_vocab_size = 39

        base_model = TFBertPrefix(base_config, use_custom=True, freeze=freeze)
        # base_model.layers[0].embeddings.trainable = False
        temp_model = ModCTSBertTF(base_model, base_config, 
                                  output_dim=kwargs.get("output_dim"),
                                  linguistic_feature_count=kwargs.get("linguistic_feature_count"),
                                  readability_feature_count=kwargs.get("readability_feature_count"), use_custom=True)
        dummy_inputs = [
            Input(shape=(512), dtype="int32"), 
            Input(shape=(51)), 
            Input(shape=(35)), 
            Input(shape=(512), dtype="int32"), 
            Input(shape=(512), dtype="int32"), 
            Input(shape=(512), dtype="int32")]
        outputs = temp_model(dummy_inputs)
        model = Model(inputs=dummy_inputs, outputs=outputs)
        model.summary(expand_nested=True)
        return model
    elif model_name == "custom_split_cts_bert_prompt":
        # Initialize configuration
        config = kwargs.get("config")
        base_config = BertConfig.from_pretrained(pretrain_name)
        base_config.pre_seq_len = 40
        base_config.model_name_or_path = pretrain_name
        base_config.prefix_projection = False
        base_config.prefix_hidden_size = 512
        base_config.pos_vocab_size = 39
        base_config.pos_embedding_dim = config.EMBEDDING_DIM
        base_config.pos_vocab_size = kwargs.get("pos_vocab_size")
        base_config.pos_dropout_prob = config.DROPOUT


        base_model = TFBertPrefix(base_config, use_custom=True, freeze=freeze)
        # temp_model = ModSplitCTSBertTF(base_model, base_config, 
        #                           output_dim=kwargs.get("output_dim"),
        #                           linguistic_feature_count=kwargs.get("linguistic_feature_count"),
        #                           readability_feature_count=kwargs.get("readability_feature_count"))
        temp_model = ModCTSBertTF(base_model, base_config, 
                                  output_dim=kwargs.get("output_dim"),
                                  linguistic_feature_count=kwargs.get("linguistic_feature_count"),
                                  readability_feature_count=kwargs.get("readability_feature_count"))

        dummy_inputs = [
            Input(shape=(512), dtype="int32"), 
            Input(shape=(51)), 
            Input(shape=(35)), 
            Input(shape=(512), dtype="int32"), 
            Input(shape=(512), dtype="int32"), 
            Input(shape=(512), dtype="int32")]
        outputs = temp_model(dummy_inputs)
        model = Model(inputs=dummy_inputs, outputs=outputs)
        model.summary(expand_nested=True)

        if config.PRETRAIN_BERT:
            pretrain_bert_path = os.path.join(config.OUTPUT_PATH, "prompt_tuning", f"current_model_prompt_{args.test_prompt_id}.h5")
            base_clone_model = get_prompt_tuning_model(pretrain_name, freeze)
            base_clone_model.load_weights(pretrain_bert_path)
            base_layer_dict = dict([(layer.name, layer) for layer in base_clone_model.layers])
            print("Base layer dict: ", base_layer_dict)

            print("Loaded layers:")
            print("===================")
            for layer in model.layers[-1].layers:
                layer_name = layer.name
                if layer_name in "tf_bert_prefix_1":
                    layer.set_weights(base_layer_dict["tf_bert_prefix_1"].get_weights())
                    print(layer_name)
            print("===================")
            print("Loaded weights from clone to new model: ", pretrain_bert_path)
        return model
    else:
        raise ValueError(f'Model {model_name} not supported')