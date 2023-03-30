from .cts import CTS
from .cts_bert import CTSBertDocument, CTSBertDocumentToken, CTSBertSegment, CTSPrompt


def get_model(model_name):
    if model_name == 'cts':
        return CTS
    elif model_name == 'cts_bert':
        return CTSBertDocument
    elif model_name == 'cts_bert_token':
        return CTSBertDocumentToken
    elif model_name == 'cts_bert_segment':
        return CTSBertSegment
    elif model_name == 'cts_prompt':
        return CTSPrompt
    else:
        raise ValueError(f'Model {model_name} not found')