import os

import numpy as np

from src.config.config import Configs
from src.utils.dataset import read_pos_vocab, read_essays, get_scaled_down_scores, pad_hierarchical_text_sequences
from datasets import Dataset


def data_pipeline(data, max_sentlen, max_sentnum):
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

    dataset = Dataset.from_dict({
        "pos": X_pos,
        "linguistic": X_linguistic_features,
        "readability": X_readability,
        "scores": Y,
        "prompt_ids": data['prompt_ids']
    })
    return dataset


def get_dataset(config: Configs, args) -> dict:
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
    train_dataset = data_pipeline(train_data, max_sentlen, max_sentnum)
    dev_dataset = data_pipeline(dev_data, max_sentlen, max_sentnum)
    test_dataset = data_pipeline(test_data, max_sentlen, max_sentnum)
    
    return {
        "datasets": [train_dataset, dev_dataset, test_dataset],
        "pos_vocab": pos_vocab,
        "max_sentnum": max_sentnum,
        "max_sentlen": max_sentlen,
        "linguistic_feature_count": len(train_dataset["linguistic"][0]),
        "readability_feature_count": len(train_dataset["readability"][0]),
        "output_dim": len(train_dataset["scores"][0])
    }


