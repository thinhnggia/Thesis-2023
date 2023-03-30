import numpy as np
from src.utils.dataset import get_min_max_scores, get_score_vector_positions


def separate_attributes_for_scoring(scores, set_ids):
    score_vector_positions = get_score_vector_positions()
    min_max_scores = get_min_max_scores()
    individual_att_scores_dict = {att: [] for att in score_vector_positions.keys()}
    score_set_comb = list(zip(scores, set_ids))
    for att_scores, set_id in score_set_comb:
        for relevant_attribute in min_max_scores[set_id].keys():
            att_position = score_vector_positions[relevant_attribute]
            individual_att_scores_dict[relevant_attribute].append(att_scores[att_position])
    return individual_att_scores_dict


def separate_and_rescale_attributes_for_scoring(scores, set_ids):
    score_vector_positions = get_score_vector_positions()
    min_max_scores = get_min_max_scores()
    individual_att_scores_dict = {}
    score_set_comb = list(zip(scores, set_ids))
    for att_scores, set_id in score_set_comb:
        for relevant_attribute in min_max_scores[set_id].keys():
            min_score = min_max_scores[set_id][relevant_attribute][0]
            max_score = min_max_scores[set_id][relevant_attribute][1]
            att_position = score_vector_positions[relevant_attribute]
            att_score = att_scores[att_position]
            rescaled_score = att_score * (max_score - min_score) + min_score
            try:
                individual_att_scores_dict[relevant_attribute].append(np.around(rescaled_score).astype(int))
            except KeyError:
                individual_att_scores_dict[relevant_attribute] = [np.around(rescaled_score).astype(int)]
    return individual_att_scores_dict