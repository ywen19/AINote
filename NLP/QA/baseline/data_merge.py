"""
Merge the raw data and CAIL2019 data
"""

import json
from tqdm import tqdm
import random

# todo: Hardcode here, change if time applies
train_cail_path = "../data/cail_train.json"
train_old_path = "../data/train.json"
dev_old_path = "../data/dev.json"

final_train_path = "../data/final_train.json"
final_dev_path = "../data/final_dev.json"


def _getExamplesFromOldData(filePath):
    """We will remove the id from the old data and reassign"""
    examples = []

    with open(filePath, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)
    for data in full_data:
        data.pop("_id", None)
        examples.append(data)

    return examples


def _combineWithCail():
    with open(train_cail_path, 'r', encoding='utf-8') as reader:
        cail_train_data = json.load(reader)

    old_train_data = _getExamplesFromOldData(train_old_path)
    old_dev_data = _getExamplesFromOldData(dev_old_path)

    example_amount = len(cail_train_data) + len(old_train_data) + len(old_dev_data)
    # get the random indices
    indices = [i for i in range(example_amount)]
    random.shuffle(indices)
    print(len(indices))

    train_data = []
    for i, case in enumerate(old_train_data):
        case["_id"] = indices[i]
        train_data.append(case)
    already_taken = len(old_train_data)  # trace the indices that have been used
    for i, case in enumerate(cail_train_data):
        case["_id"] = indices[already_taken+i]
        train_data.append(case)
    already_taken += len(cail_train_data)

    dev_data = []
    for i, case in enumerate(old_dev_data):
        case["_id"] = indices[already_taken+i]
        dev_data.append(case)

    # write out final data
    with open(final_train_path, 'w', encoding='utf8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    with open(final_dev_path, 'w', encoding='utf8') as f:
        json.dump(dev_data, f, indent=4, ensure_ascii=False)

