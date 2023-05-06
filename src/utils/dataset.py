import nltk
import pickle
import re
import random

import numpy as np
import pandas as pd

from sklearn import preprocessing


MAX_SENTLEN = 50
MAX_SENTNUM = 100
url_replacer = '<url>'


def replace_url(text):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens


def shorten_sentence(sent, max_sentlen):
    new_tokens = []
    sent = sent.strip()
    tokens = nltk.word_tokenize(sent)
    if len(tokens) > max_sentlen:
        split_keywords = ['because', 'but', 'so', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
        processed_tokens = []
        if not k_indexes:
            num = len(tokens) / max_sentlen
            num = int(round(num))
            k_indexes = [(i+1)*max_sentlen for i in range(num)]

        processed_tokens.append(tokens[0:k_indexes[0]])
        len_k = len(k_indexes)
        for j in range(len_k-1):
            processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
        processed_tokens.append(tokens[k_indexes[-1]:])

        for token in processed_tokens:
            if len(token) > max_sentlen:
                num = len(token) / max_sentlen
                num = int(np.ceil(num))
                s_indexes = [(i+1)*max_sentlen for i in range(num)]

                len_s = len(s_indexes)
                new_tokens.append(token[0:s_indexes[0]])
                for j in range(len_s-1):
                    new_tokens.append(token[s_indexes[j]:s_indexes[j+1]])
                new_tokens.append(token[s_indexes[-1]:])

            else:
                new_tokens.append(token)
    else:
        return [tokens]

    return new_tokens


def tokenize_to_sentences(text, max_sentlength, create_vocab_flag=False):
    sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)
    processed_sents = []
    for sent in sents:
        if re.search(r'(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent):
            s = re.split(r'(?=.{2,})(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent)
            ss = " ".join(s)
            ssL = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', ss)

            processed_sents.extend(ssL)
        else:
            processed_sents.append(sent)

    if create_vocab_flag:
        sent_tokens = [tokenize(sent) for sent in processed_sents]
        tokens = [w for sent in sent_tokens for w in sent]
        return tokens

    sent_tokens = []
    for sent in processed_sents:
        shorten_sents_tokens = shorten_sentence(sent, max_sentlength)
        sent_tokens.extend(shorten_sents_tokens)
    return sent_tokens


def text_tokenizer(text, tokenize_sent_flag=True, create_vocab_flag=False):
    tokens = tokenize(text)
    if tokenize_sent_flag:
        text = " ".join(tokens)
        sent_tokens = tokenize_to_sentences(text, MAX_SENTLEN, create_vocab_flag)
        return sent_tokens
    else:
        raise NotImplementedError
    
def preprocess_text(text, replace_url_flag=True):
    if replace_url_flag:
        text = replace_url(text)
    text = text.replace(u'"', u'')
    if "..." in text:
        text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
    if "??" in text:
        text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
    if "!!" in text:
        text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)
    return text


def read_pos_vocab(read_configs):
    file_path = read_configs['train_path']
    pos_tags_count = {}

    with open(file_path, 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    for _, essay in enumerate(train_essays_list[:16]):
        content = essay['content_text']
        proc_content = preprocess_text(content)
        content = text_tokenizer(proc_content, True, True)
        content = [w.lower() for w in content]
        tags = nltk.pos_tag(content)
        for tag in tags:
            tag = tag[1]
            try:
                pos_tags_count[tag] += 1
            except KeyError:
                pos_tags_count[tag] = 1

    pos_tags = {'<pad>': 0, '<unk>': 1}
    pos_len = len(pos_tags)
    pos_index = pos_len
    for pos in pos_tags_count.keys():
        pos_tags[pos] = pos_index
        pos_index += 1
    return pos_tags


def get_readability_features(readability_path):
    with open(readability_path, 'rb') as fp:
        readability_features = pickle.load(fp)
    return readability_features


def get_linguistic_features(linguistic_features_path):
    features_df = pd.read_csv(linguistic_features_path)
    return features_df


def get_normalized_features(features_df):
    column_names_not_to_normalize = ['item_id', 'prompt_id', 'score']
    column_names_to_normalize = list(features_df.columns.values)
    for col in column_names_not_to_normalize:
        column_names_to_normalize.remove(col)
    final_columns = ['item_id'] + column_names_to_normalize
    normalized_features_df = None
    for prompt_ in range(1, 9):
        is_prompt_id = features_df['prompt_id'] == prompt_
        prompt_id_df = features_df[is_prompt_id]
        x = prompt_id_df[column_names_to_normalize].values
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_pd1 = min_max_scaler.fit_transform(x)
        df_temp = pd.DataFrame(normalized_pd1, columns=column_names_to_normalize, index = prompt_id_df.index)
        prompt_id_df[column_names_to_normalize] = df_temp
        final_df = prompt_id_df[final_columns]
        if normalized_features_df is not None:
            normalized_features_df = pd.concat([normalized_features_df,final_df],ignore_index=True)
        else:
            normalized_features_df = final_df
    return normalized_features_df


def get_score_vector_positions():
    return {
        'score': 0,
        'content': 1,
        'organization': 2,
        'word_choice': 3,
        'sentence_fluency': 4,
        'conventions': 5,
        'prompt_adherence': 6,
        'language': 7,
        'narrativity': 8,
    }


def read_essay_sets(essay_list, readability_features, normalized_features_df, pos_vocab):
    out_data = {
        'essay_ids': [],
        'pos_x': [],
        'readability_x': [],
        'features_x': [],
        'data_y': [],
        'prompt_ids': [],
        "original_text": [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    for essay in essay_list:
        # Get general data
        essay_id = int(essay['essay_id'])
        essay_set = int(essay['prompt_id'])
        out_data['prompt_ids'].append(essay_set)
        out_data['essay_ids'].append(essay_id)

        # Get label vector
        scores_and_positions = get_score_vector_positions()
        y_vector = [-1] * len(scores_and_positions)
        for score in scores_and_positions.keys():
            if score in essay.keys():
                y_vector[scores_and_positions[score]] = int(essay[score])
        out_data['data_y'].append(y_vector)

        # Get readability features
        item_index = np.where(readability_features[:, :1] == essay_id)
        item_row_index = item_index[0][0]
        item_features = readability_features[item_row_index][1:]
        out_data['readability_x'].append(item_features)

        # Get handcrafted features
        feats_df = normalized_features_df[normalized_features_df.loc[:, 'item_id'] == essay_id]
        feats_list = feats_df.values.tolist()[0][1:]
        out_data['features_x'].append(feats_list)

        # Get pos tag
        content = essay['content_text']
        proc_content = preprocess_text(content)
        out_data['original_text'].append(proc_content)

        sent_tokens = text_tokenizer(proc_content, tokenize_sent_flag=True)
        sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

        sent_tag_indices = []
        tag_indices = []
        for sent in sent_tokens:
            length = len(sent)
            if length > 0:
                if out_data['max_sentlen'] < length:
                    out_data['max_sentlen'] = length
                tags = nltk.pos_tag(sent)
                for tag in tags:
                    if tag[1] in pos_vocab:
                        tag_indices.append(pos_vocab[tag[1]])
                    else:
                        tag_indices.append(pos_vocab['<unk>'])
                sent_tag_indices.append(tag_indices)
                tag_indices = []

        out_data['pos_x'].append(sent_tag_indices)
        if out_data['max_sentnum'] < len(sent_tag_indices):
            out_data['max_sentnum'] = len(sent_tag_indices)

    assert(len(out_data['pos_x']) == len(out_data['readability_x']))
    return out_data


def read_essays(read_configs, pos_vocab):
    readability_features = get_readability_features(read_configs['readability_path'])
    linguistic_features = get_linguistic_features(read_configs['features_path'])
    normalized_linguistic_features = get_normalized_features(linguistic_features)
    with open(read_configs['train_path'], 'rb') as train_file:
        train_essays_list = pickle.load(train_file)[:20]
    with open(read_configs['dev_path'], 'rb') as dev_file:
        dev_essays_list = pickle.load(dev_file)[:20]
    with open(read_configs['test_path'], 'rb') as test_file:
        test_essays_list = pickle.load(test_file)[:20]

    train_data = read_essay_sets(train_essays_list, readability_features, normalized_linguistic_features, pos_vocab)
    dev_data = read_essay_sets(dev_essays_list, readability_features, normalized_linguistic_features, pos_vocab)
    test_data = read_essay_sets(test_essays_list, readability_features, normalized_linguistic_features, pos_vocab)
    return train_data, dev_data, test_data


def get_min_max_scores():
    return {
        1: {'score': (2, 12), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        2: {'score': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        3: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        4: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        5: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        6: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        7: {'score': (0, 30), 'content': (0, 6), 'organization': (0, 6), 'conventions': (0, 6)},
        8: {'score': (0, 60), 'content': (2, 12), 'organization': (2, 12), 'word_choice': (2, 12),
            'sentence_fluency': (2, 12), 'conventions': (2, 12)}}


def get_scaled_down_scores(scores, prompts):
    score_positions = get_score_vector_positions()
    min_max_scores = get_min_max_scores()
    score_prompts = zip(scores, prompts)
    scaled_score_list = []
    for score_vector, prompt in score_prompts:
        rescaled_score_vector = [-1] * len(score_positions)
        for ind, att_val in enumerate(score_vector):
            if att_val != -1:
                attribute_name = list(score_positions.keys())[list(score_positions.values()).index(ind)]
                min_val = min_max_scores[prompt][attribute_name][0]
                max_val = min_max_scores[prompt][attribute_name][1]
                scaled_score = (att_val - min_val) / (max_val - min_val)
                rescaled_score_vector[ind] = scaled_score
        scaled_score_list.append(rescaled_score_vector)
    assert len(scaled_score_list) == len(scores)
    for scores in scaled_score_list:
        assert min(scores) >= -1
        assert max(scores) <= 1
    return scaled_score_list


def pad_hierarchical_text_sequences(index_sequences, max_sentnum, max_sentlen):
    X = np.empty([len(index_sequences), max_sentnum, max_sentlen], dtype=np.int32)

    for i in range(len(index_sequences)):
        sequence_ids = index_sequences[i]
        num = len(sequence_ids)

        for j in range(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            for k in range(length):
                wid = word_ids[k]
                X[i, j, k] = wid
            X[i, j, length:] = 0

        X[i, num:, :] = 0
    return X


def parse_dataset(dataset, mode="use_bert"):
    if mode == "use_bert":
        inputs = [
            np.array(dataset["pos"]),
            np.array(dataset["linguistic"]),
            np.array(dataset["readability"]),
            np.array(dataset["input_ids"]),
            np.array(dataset["attention_mask"]),
            np.array(dataset["token_type_ids"])
        ]
        Y = np.array(dataset["scores"])
    elif mode == "use_segment":
        inputs = [
            np.array(dataset["pos"]),
            np.array(dataset["linguistic"]),
            np.array(dataset["readability"]),
            np.array(dataset["segment_1"]),
            np.array(dataset["segment_2"]),
            np.array(dataset["segment_3"]),
            np.array(dataset["segment_4"]),
        ]
        Y = np.array(dataset["scores"])
    elif mode == "prompt_tuning":
        inputs = [
            np.array(dataset["input_ids"]),
            np.array(dataset["attention_mask"]),
            np.array(dataset["token_type_ids"])
        ]
        Y = np.array(dataset["cr_labels"])
    elif mode == "use_custom":
        inputs = [
            np.array(dataset["pos_tags"]),
            np.array(dataset["linguistic"]),
            np.array(dataset["readability"]),
            np.array(dataset["input_ids"]),
            np.array(dataset["attention_mask"]),
            np.array(dataset["token_type_ids"])
        ]
        Y = np.array(dataset["scores"])
    else:
        inputs = [
            np.array(dataset["pos"]),
            np.array(dataset["linguistic"]),
            np.array(dataset["readability"])
        ]
        Y = np.array(dataset["scores"])

    prompt_ids = dataset["prompt_ids"]
    return inputs, prompt_ids, Y 


# -------------------------------------------------------------------------
#                          Discourse corruption
# -------------------------------------------------------------------------
def create_training_data_for_di_shuffled_essays(refined_essay, di_list, use_prob=False):
    essay_orig, essay_shf = di_shuffled_essay(refined_essay, di_list)
    total_essay = essay_orig + essay_shf

    scores = [1] * len(essay_orig) + [0] * len(essay_shf)
    return total_essay, scores


def di_shuffled_essay(essay_list, di_list):
    shuffle_essay = []
    segmented_essay = []
    
    for s in essay_list:
        # Replace discourse indicators with the predefined list.
        i_di = find_replace_di(s, di_list)

        # Shuffle the discourse indicators in the essay.
        i_di_shuf, is_corrupt = di_change(i_di)

        segmented_essay.append(s)
        if is_corrupt:
            shuffle_essay.append(i_di_shuf)

    return segmented_essay, shuffle_essay


def find_replace_di(essay, di_list):
    
    # Query will look like: "because|in order to"
    regex_di = "|".join([" ".join(di) for di in di_list])
    
    def _rep(m):
        return r" DI_{}".format(m.group(2).replace(" ", "_"))
        
    essay = re.sub(
        fr"(^| )({regex_di})(\b)",
        _rep,
        essay)

    return essay


def di_func1(token, dis):
    """
    Replace with random DI
    """
    return dis.pop()[3:].replace("_", " ")


def di_func2(token, dis):
    """
    Delete token
    """
    return 


def di_func3(token, dis):
    """
    Return unchange token
    """
    return token


def di_rand_func():
    functions = [di_func1, di_func2, di_func3]
    chose_func = random.choices(functions, weights=[60, 20, 20], k=1)[0]
    return chose_func


def di_change(essay):
    tks = [tk for tk in essay.split(" ")]
    dis = [tk for tk in tks if tk.startswith("DI_")]
    random.shuffle(dis)

    new_tks = []
    is_corrupt = False
    for tk in tks:
        if tk.startswith("DI_"):
            chose_func = di_rand_func()
            new_tk = chose_func(tk, dis)
            if new_tk:
                new_tks.append(new_tk)
                is_corrupt = True
        else:
            new_tks.append(tk)

    new_tks = [tk.replace("_", " ") for tk in new_tks]
    return " ".join(new_tks), is_corrupt


def load_discourse_indicators(fn_di):
    file = open(fn_di)
    data = file.read()
    data = data.splitlines()

    lowered_list = [i.lower() for i in data]
    di_list = [i.split() for i in lowered_list]
    
    return sorted(di_list, key=len, reverse=True)


def create_training_data_for_shuffled_essays(refined_essay, di_list, use_prob=False):
    essay_orig, essay_shf = shuffled_essay(refined_essay, di_list)
    total_essay = essay_orig + essay_shf
    
    scores = [1] * len(essay_orig) + [0] * len(essay_shf)    
    return total_essay, scores


def shuffled_essay(essay_list, di_list):
    shuffle_essay = []
    segmented_essay = []

    for i in essay_list:
        s = [x for x in nltk.sent_tokenize(i)]
        segmented_essay+= [" ".join(s)]

        np.random.shuffle(s)
        nn = " ".join(s)
        shuffle_essay+=[nn]

    return segmented_essay, shuffle_essay


def create_corrupt_data(essays, di_list, use_prob=False):
    if use_prob:
        shufftle_types = [shuffled_essay, di_shuffled_essay]
        rand_shuf_type = random.choices(shufftle_types, weights=[50, 50], k=1)[0]

        essay_orig, essay_shf = rand_shuf_type(essays, di_list)
    else:
        # Create all version of corrupted
        essay_orig, essay_shf_1 = di_shuffled_essay(essays, di_list)
        essay_orig, essay_shf_2 = shuffled_essay(essays, di_list)
        essay_shf = essay_shf_1 + essay_shf_2

    total_essay = essay_orig + essay_shf
    
    scores = [1] * len(essay_orig) + [0] * len(essay_shf)    
    return total_essay, scores


def get_corrupt_func(shuffle_type):
    if shuffle_type == "all":
        return create_corrupt_data
    elif shuffle_type == "di":
        return create_training_data_for_di_shuffled_essays
    elif shuffle_type == "sentence":
        return create_training_data_for_shuffled_essays
    else:
        raise ValueError("Not supported shuffle type: {}".format(shuffle_type))
    

def get_pos_tags_by_tokenizer(max_length, original_text, tokenizer, pos_vocab):
    list_pos_tags = []
    for text in original_text:
        tokens = tokenizer.tokenize(text)
        pos_tags = nltk.pos_tag(tokens=tokens)
        
        tag_indices = []
        for tag in pos_tags:
            if tag[1] in pos_vocab:
                tag_indices.append(pos_vocab[tag[1]])
            else:
                tag_indices.append(pos_vocab['<unk>'])
        tag_indices = [pos_vocab['<unk>']] + tag_indices + [pos_vocab['<unk>']]
        if len(tag_indices) < max_length:
            padding = max_length - len(tag_indices)
            padding_indices = tag_indices + [pos_vocab['<pad>']] * padding
        else:
            padding_indices = tag_indices[:max_length]

        assert len(padding_indices) == max_length
        list_pos_tags.append(padding_indices)
    return list_pos_tags
