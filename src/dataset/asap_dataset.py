import os

import numpy as np

from src.config.config import Configs
from src.utils.dataset import read_pos_vocab, read_essays, get_scaled_down_scores, pad_hierarchical_text_sequences
from datasets import Dataset
from transformers import BertTokenizer


def data_pipeline(data, max_sentlen, max_sentnum, bert_tokenizer: BertTokenizer = None):
    # Scaled down label score
    data['y_scaled'] = get_scaled_down_scores(data['data_y'], data['prompt_ids'])

    # Pad text sequence of postag
    X_pos = pad_hierarchical_text_sequences(data['pos_x'], max_sentnum, max_sentlen)
    X_pos = X_pos.reshape((X_pos.shape[0], X_pos.shape[1] * X_pos.shape[2]))

    # Get linguistic features
    X_linguistic_features = np.array(data['features_x'])

    # Get readability
    X_readability = np.array(data['readability_x'])

    # Get label score
    Y = np.array(data['y_scaled'])

    # Get bert tokenized 
    tokenized_sentences = bert_tokenizer(data['original_text'], padding='max_length', truncation=True, return_tensors="np")

    dataset = Dataset.from_dict({
        "pos": X_pos.astype(np.int32),
        "linguistic": X_linguistic_features,
        "readability": X_readability,
        "scores": Y,
        "prompt_ids": data['prompt_ids'],
        "input_ids": tokenized_sentences["input_ids"],
        "token_type_ids": tokenized_sentences["token_type_ids"],
        "attention_mask": tokenized_sentences["attention_mask"]
    })
    tf_dataset = Dataset.from_dict({
        "input_1": X_pos.astype(np.int32),
        "input_2": X_linguistic_features,
        "input_3": X_readability,
        "scores": Y,
        "prompt_ids": data['prompt_ids'],
        "input_4": tokenized_sentences["input_ids"],
        "input_5": tokenized_sentences["token_type_ids"],
        "input_6": tokenized_sentences["attention_mask"]
    })
    return dataset, tf_dataset


def get_dataset(config: Configs, args, bert_tokenizer: BertTokenizer = None) -> dict:
    read_configs = {
        "train_path": os.path.join(config.DATA_PATH, str(args.test_prompt_id), 'train.pkl'),
        "dev_path": os.path.join(config.DATA_PATH, str(args.test_prompt_id), 'dev.pkl'),
        "test_path": os.path.join(config.DATA_PATH, str(args.test_prompt_id), 'test.pkl'),
        "features_path": config.FEATURES_PATH,
        "readability_path": config.READABILITY_PATH
    }

    # Get postag dictionary
    pos_vocab = read_pos_vocab(read_configs)
    train_data, dev_data, test_data = read_essays(read_configs, pos_vocab)
    
    max_sentlen = max(
        train_data['max_sentlen'],
        dev_data['max_sentlen'],
        test_data['max_sentlen']
    )
    max_sentnum = max(
        train_data['max_sentnum'],
        dev_data['max_sentnum'],
        test_data['max_sentnum']
    )
    
    # Run data pipeline
    train_dataset, train_dataset_tf = data_pipeline(train_data, max_sentlen, max_sentnum, bert_tokenizer)
    dev_dataset, dev_dataset_tf = data_pipeline(dev_data, max_sentlen, max_sentnum, bert_tokenizer)
    test_dataset, test_dataset_tf = data_pipeline(test_data, max_sentlen, max_sentnum, bert_tokenizer)
    
    return {
        "datasets": [train_dataset, dev_dataset, test_dataset],
        "tf_datasets": [train_dataset_tf, dev_dataset_tf, test_dataset_tf],
        "pos_vocab": pos_vocab,
        "max_sentnum": max_sentnum,
        "max_sentlen": max_sentlen,
        "linguistic_feature_count": len(train_dataset["linguistic"][0]),
        "readability_feature_count": len(train_dataset["readability"][0]),
        "output_dim": len(train_dataset["scores"][0])
    }


