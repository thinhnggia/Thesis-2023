import os
import argparse

from tqdm import tqdm
from src.utils.common import save_file


def combine_all_prompt_essays(file_list, essays, prompt):
    if prompt < 3:
        attribute_score_indices = {
            'score': 3,
            'content': 4,
            'organization': 5,
            'word_choice': 6,
            'sentence_fluency': 7,
            'conventions': 8
        }
    else:
        attribute_score_indices = {
            'score': 3,
            'content': 4,
            'prompt_adherence': 5,
            'language': 6,
            'narrativity': 7
        }
    for file_ in file_list:
        input_file = open(file_, 'r')
        lines = input_file.readlines()
        for line in lines[1:]:
            tokens = line.strip().split('\t')
            essay = {
                'essay_id': tokens[0],
                'prompt_id': tokens[1]
            }
            for key in attribute_score_indices.keys():
                essay[key] = tokens[attribute_score_indices[key]]
            essays.append(essay)
    return essays


def combine_for_prompt_seven_eight(filepath, essays, prompt):
    if prompt == 7:
        attribute_score_indices = {
            'content': (10, 16),
            'organization': (11, 17),
            'style': (12, 18),
            'conventions': (13, 19)
        }
    elif prompt == 8:
        attribute_score_indices = {
            'content': (10, 16, 22),
            'organization': (11, 17, 23),
            'voice': (12, 18, 24),
            'word_choice': (13, 19, 25),
            'sentence_fluency': (14, 20, 26),
            'conventions': (15, 21, 27)
        }

    with open(filepath, 'r', encoding='latin-1') as input_file:
        for index, line in enumerate(input_file):
            tokens = line.strip().split('\t')
            if index == 0:
                pass
            else:
                essay = {
                    'essay_id': tokens[0],
                    'prompt_id': tokens[1],
                    'score': tokens[6]
                }
                if prompt == 7:
                    if int(tokens[1]) == prompt:
                        for key in attribute_score_indices.keys():
                            rater1 = int(tokens[attribute_score_indices[key][0]])
                            rater2 = int(tokens[attribute_score_indices[key][1]])
                            resolved_score = rater1 + rater2
                            essay[key] = resolved_score
                        essays.append(essay)
                elif prompt == 8:
                    if int(tokens[1]) == prompt:
                        for key in attribute_score_indices.keys():
                            rater1 = int(tokens[attribute_score_indices[key][0]])
                            rater2 = int(tokens[attribute_score_indices[key][1]])
                            attribute_tokens = tokens[10:28]
                            if len(attribute_tokens) == 12:
                                resolved_score = rater1 + rater2
                            elif len(attribute_tokens) == 18:
                                resolved_score = int(tokens[attribute_score_indices[key][2]]) * 2
                            else:
                                raise NotImplementedError
                            essay[key] = resolved_score
                        essays.append(essay)
    return essays


def find_matches(list_of_essay_dicts, tsv_path):
    matched_essays = []
    input_file = open(tsv_path, 'r')
    lines = input_file.readlines()
    for line in lines[1:]:
        tokens = line.strip().split('\t')
        essay_id = tokens[0]
        content = tokens[2]
        for essay_dict in list_of_essay_dicts:
            if essay_dict['essay_id'] == essay_id:
                essay_dict['content_text'] = content
                matched_essays.append(essay_dict)
                break
    return matched_essays


def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    all_combined_attribute_essays = []
    for i in range(1, 7):
        prompt_attribute_essays_path = os.path.join(args.traits_path, str(i))
        prompt_attribute_train_path = os.path.join(prompt_attribute_essays_path, 'train.tsv')
        prompt_attribute_dev_path = os.path.join(prompt_attribute_essays_path, 'dev.tsv') 
        prompt_attribute_test_path = os.path.join(prompt_attribute_essays_path, 'test.tsv')
        prompt_attribute_paths = [
            prompt_attribute_train_path, prompt_attribute_dev_path, prompt_attribute_test_path]

        all_combined_attribute_essays = combine_all_prompt_essays(
            prompt_attribute_paths, all_combined_attribute_essays, i)

    for i in range(7, 9):
        all_combined_attribute_essays = combine_for_prompt_seven_eight(
            args.orig_path, all_combined_attribute_essays, i)
    
    tbar = tqdm(range(1, 9), total=8)
    for i in tbar:
        tbar.set_description(f'Matching prompt: {i}')
        
        output_path = os.path.join(args.output_path, str(i))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        fold_path = os.path.join(args.paes_path, str(i))
        train_path = os.path.join(fold_path, 'train.tsv')
        dev_path = os.path.join(fold_path, 'dev.tsv')
        test_path = os.path.join(fold_path, 'test.tsv')

        cross_prompt_train = find_matches(all_combined_attribute_essays, train_path)
        cross_prompt_dev = find_matches(all_combined_attribute_essays, dev_path)
        cross_prompt_test = find_matches(all_combined_attribute_essays, test_path)

        save_file(os.path.join(output_path, 'train.pkl'), cross_prompt_train)
        save_file(os.path.join(output_path, 'dev.pkl'), cross_prompt_dev)
        save_file(os.path.join(output_path, 'test.pkl'), cross_prompt_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tool to prepare dataset")
    parser.add_argument("--traits_path", type=str, help="Path to traits score", default="src/data/ASAP/traits")
    parser.add_argument("--paes_path", type=str, help="Path to data fold of PAES paper", default="src/data/ASAP/PAES")
    parser.add_argument("--orig_path", type=str, help="Original data path", default="src/data/ASAP/training_set_rel3.tsv")
    parser.add_argument("--output_path", type=str, help="Output to save prepared data", default="src/data/ASAP/final_data")
    args = parser.parse_args()

    main(args)
