from collections import Counter
import json
import sys


def f1_score(prediction, ground_truth):
    # prediction and ground truth are strings
    if prediction in ['yes', 'no', 'unknown'] and prediction != ground_truth:
        return 0, 0, 0
    if ground_truth in ['yes', 'no', 'noanswer'] and prediction != ground_truth:
        return 0, 0, 0

    common_tokens = Counter(prediction) & Counter(ground_truth)
    same_token_amount = sum(common_tokens.values())
    if same_token_amount == 0:
        return 0, 0, 0
    precision = 1.0 * same_token_amount / len(prediction)
    recall = 1.0 * same_token_amount / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def if_exact_match(prediction, ground_truth):
    # if the answer is exactly the same as the ground truth
    if prediction == ground_truth:
        return 1.0
    return 0.0


def update_answer_metrics(metrics, prediction, ground_truth):
    # update the f1 score and exact match score per sample on answers
    exact_match = if_exact_match(prediction, ground_truth)
    metrics['em'] += exact_match

    precision, recall, f1 = f1_score(prediction, ground_truth)
    metrics['f1'] += f1
    metrics['precision'] += precision
    metrics['recall'] += recall

    return exact_match, precision, recall, f1


def update_sp_metrics(metrics, prediction, ground_truth):
    # update the f1 score and exact match score per sample on support facts
    support_predict = set(map(tuple, prediction))  # {(title of doc, sentence local id), ...}
    support_ground_truth = set(map(tuple, ground_truth))

    # update count for calculating metrics
    true_pos, false_pos, false_neg = 0, 0, 0
    # calculate true positive and false positive predictions
    for item in support_predict:
        if item in support_ground_truth:
            true_pos += 1
        else:
            false_pos += 1
    # calculate false negative predictions
    for item in support_ground_truth:
        if item not in support_predict:
            false_neg += 1

    # calculate precision, recall, f1 and exact match
    precision = 1.0 * true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0.0
    recall = 1.0 * true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    exact_match = 1.0 if false_pos + false_neg == 0 else 0.0

    # update the evaluation metrics
    metrics['sp_em'] += exact_match
    metrics['sp_f1'] += f1
    metrics['sp_precision'] += precision
    metrics['sp_recall'] += recall

    return exact_match, precision, recall, f1


def eval_performance(prediction_file_path, ground_truth_file_path):
    # read the files in
    with open(prediction_file_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    with open(ground_truth_file_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    # initialize the evaluation metrics
    metrics = {'em': 0, 'f1': 0, 'precision': 0, 'recall': 0,
               'sp_em': 0, 'sp_f1': 0, 'sp_precision': 0, 'sp_recall': 0,
               'joint_em': 0, 'joint_f1': 0, 'joint_precision': 0, 'joint_recall': 0}

    for case in ground_truth:
        # step through each document(case) in the ground truth
        doc_id = str(case['_id'])
        if_joint_eval = True  # if evaluate answer and support facts jointly

        # safe check
        # if the doc id in the ground truth is not in the prediction

        # measure performance by answers
        if doc_id not in predictions['answer']:
            if_joint_eval = False
            print('missing answers for doc {}'.format(doc_id))
        else:
            exact_match, precision, recall, f1 = update_answer_metrics(
                metrics, predictions['answer'][doc_id], case['answer']
            )

        # measure performance by answers
        if doc_id not in predictions['sp']:
            if_joint_eval = False
            print('missing support facts for doc {}'.format(doc_id))
        else:
            sp_exact_match, sp_precision, sp_recall, sp_f1 = update_sp_metrics(
                metrics, predictions['sp'][doc_id], case['supporting_facts']
            )

        # if no doc in ground truth miss answers or suporting facts, joint evaluate
        # todo: better way to do conditional logic; risk is that the scores are not initialized
        if if_joint_eval:
            joint_precision = precision * sp_precision
            joint_recall = recall * sp_recall
            joint_em = exact_match * sp_exact_match

            joint_f1 = (
                    2 * joint_precision * joint_recall / (joint_precision + joint_recall)
                    if joint_precision+joint_recall > 0
                    else 0.
            )
            # update the metrics
            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_precision'] += joint_precision
            metrics['joint_recall'] += joint_recall

    # average the sum of evaluation scores
    sample_amount = len(ground_truth)
    for key in metrics.keys():
        metrics[key] /= sample_amount

    return metrics
