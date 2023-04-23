import os
import math

import numpy as np

from src.config.config import Configs
from src.utils.dataset import read_pos_vocab, read_essays, get_scaled_down_scores, pad_hierarchical_text_sequences, get_corrupt_func, load_discourse_indicators
from datasets import Dataset
from transformers import BertTokenizer


def data_pipeline(data, max_sentlen, max_sentnum, config, bert_tokenizer: BertTokenizer = None):
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

    dataset_dict = {
        "pos": X_pos.astype(np.int32),
        "linguistic": X_linguistic_features,
        "readability": X_readability,
        "scores": Y,
        "prompt_ids": data['prompt_ids']
    }
    tf_dataset_dict = {
        "input_1": X_pos.astype(np.int32),
        "input_2": X_linguistic_features,
        "input_3": X_readability,
        "scores": Y,
        "prompt_ids": data['prompt_ids'],
    }

    if bert_tokenizer:
        if config.MODE == "use_segment":
            chunk_sizes = [90, 30, 130,10] # TODO: Currently hard code, refactor to add segment size
        else:
            chunk_sizes = []
        res = bert_tokenize(data, bert_tokenizer, chunk_sizes=chunk_sizes, shuffle_type=config.SHUFFLE_TYPE)
        if config.SHUFFLE_TYPE:
            tokenized_documents, labels = res
            dataset_dict = {
                "prompt_ids": [0] * len(labels) # Dummy prompt ids
            }
            dataset_dict["cr_labels"] = np.array(labels) # Corrupted labels

            tf_dataset_dict = {
                "prompt_ids": [0] * len(labels) # Dummy prompt ids
            }
            tf_dataset_dict["cr_labels"] = np.array(labels) # Corrupted labels
        else:
            tokenized_documents = res

        if config.MODE == "use_bert" or config.MODE == "prompt_tuning":
            dataset_dict["input_ids"] = tokenized_documents["input_ids"]
            dataset_dict["token_type_ids"] = tokenized_documents["token_type_ids"]
            dataset_dict["attention_mask"] = tokenized_documents["attention_mask"]

            if config.SHUFFLE_TYPE:
                tf_dataset_dict["input_1"] = tokenized_documents["input_ids"]
                tf_dataset_dict["input_2"] = tokenized_documents["token_type_ids"]
                tf_dataset_dict["input_3"] = tokenized_documents["attention_mask"]
            else:
                tf_dataset_dict["input_4"] = tokenized_documents["input_ids"]
                tf_dataset_dict["input_5"] = tokenized_documents["token_type_ids"]
                tf_dataset_dict["input_6"] = tokenized_documents["attention_mask"]
        elif config.MODE == "use_segment":
            dataset_dict["segment_1"] = tokenized_documents[0]
            dataset_dict["segment_2"] = tokenized_documents[1]
            dataset_dict["segment_3"] = tokenized_documents[2]
            dataset_dict["segment_4"] = tokenized_documents[3]

            tf_dataset_dict["input_4"] = tokenized_documents[0]
            tf_dataset_dict["input_5"] = tokenized_documents[0]
            tf_dataset_dict["input_6"] = tokenized_documents[0]
            tf_dataset_dict["input_7"] = tokenized_documents[0]
    
    dataset = Dataset.from_dict(dataset_dict)
    tf_dataset = Dataset.from_dict(tf_dataset_dict)

    return dataset, tf_dataset


def bert_tokenize(data, tokenizer, chunk_sizes, shuffle_type):
    original_text = data['original_text']
    labels = None
    if shuffle_type:
        di_list = load_discourse_indicators("src/data/DI_wo_and.txt")
        original_text, labels = get_corrupt_func(shuffle_type)(data["original_text"], di_list, use_prob=False)

    if not chunk_sizes:
        tokenized_documents = tokenizer(original_text, padding='max_length', truncation=True, return_tensors="np")
        if labels:
            return tokenized_documents, labels
        else:
            return tokenized_documents
    else:
        document_representations_chunk_list = []
        for chunk_size in chunk_sizes:
            tokenized_documents = [tokenizer.tokenize(document) for document in original_text]
            max_sequences_per_document = math.ceil(max(len(x)/(chunk_size-2) for x in tokenized_documents))
            output = np.zeros(shape=(len(original_text), max_sequences_per_document, 3, chunk_size), dtype=np.int32)
            
            document_seq_lengths = []
            for doc_index, tokenized_document in enumerate(tokenized_documents):
                max_seq_index = 0
                for seq_index, i in enumerate(range(0, len(tokenized_document), (chunk_size-2))):
                    raw_tokens = tokenized_document[i:i+(chunk_size-2)]
                    tokens = []
                    input_type_ids = []
                    tokens.append("[CLS]")
                    input_type_ids.append(0)
                    for token in raw_tokens:
                        tokens.append(token)
                        input_type_ids.append(0)
                    tokens.append("[SEP]")
                    input_type_ids.append(0)

                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    attention_masks = [1] * len(input_ids)

                    while len(input_ids) < chunk_size:
                        input_ids.append(0)
                        input_type_ids.append(0)
                        attention_masks.append(0)
                    output[doc_index][seq_index] = np.concatenate((np.array(input_ids)[np.newaxis,:],
                                                            np.array(input_type_ids)[np.newaxis,:],
                                                            np.array(attention_masks)[np.newaxis,:]),
                                                            axis=0)
                    max_seq_index = seq_index
                document_seq_lengths.append(max_seq_index+1)
            
            document_representations_chunk_list.append(output)

        return document_representations_chunk_list


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
    train_dataset, train_dataset_tf = data_pipeline(train_data, max_sentlen, max_sentnum, config, bert_tokenizer)
    dev_dataset, dev_dataset_tf = data_pipeline(dev_data, max_sentlen, max_sentnum, config, bert_tokenizer)
    test_dataset, test_dataset_tf = data_pipeline(test_data, max_sentlen, max_sentnum, config, bert_tokenizer)
    
    if config.SHUFFLE_TYPE:
        return {
            "datasets": [train_dataset, dev_dataset, test_dataset],
            "tf_datasets": [train_dataset_tf, dev_dataset_tf, test_dataset_tf],
            "pos_vocab": [],
            "max_sentnum": 0,
            "max_sentlen": 0,
            "linguistic_feature_count": 0,
            "readability_feature_count": 0,
            "output_dim": 0
        }
    else:
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
