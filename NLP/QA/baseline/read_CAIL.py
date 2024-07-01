"""
This module reads in the CAIL2019 data, and reformat it to the original data format
"""

import json
import re
from difflib import SequenceMatcher
from tqdm import tqdm

from typing import List, Any, Tuple, Dict


def _readExamplesFromFile(filePath: str) -> List[Any]:
    with open(filePath, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)
    if "data" in full_data:
        return full_data["data"]
    return []


def _sengmentParagraph(paragraph: str) -> Tuple[List[str], List[str]]:
    """
    Split the paragraph to sentences(segments);
    Note that we want to keep punctuations, so we will also detect those
    """
    token_regex = r"[^!?,;。\.\!\?]+[!?。\.\!\?]?"
    punc_regex = r"[!?,;。\.\!\?]+[!?。\.\!\?]?"

    sents = re.findall(token_regex, paragraph)
    puncs = re.findall(punc_regex, paragraph)
    return sents, puncs


def _getCaseContext(case: Dict[str, Any]) -> List[Any]:
    """
    The case in 2019CAIL contains: case_id, domain and paragraph
    We need the context in the paragraph and format into [sent0, [sent0, sent1...]]
    """
    if "paragraphs" not in case:
        return []

    sentences = []
    case_info = case["paragraphs"][0]
    orig_context = case_info["context"]
    # split the paragraph to sentences
    sents, puncs = _sengmentParagraph(orig_context)
    puncs_last_index = len(puncs)-1

    for i, sent in enumerate(sents):
        if i <= puncs_last_index:
            sentences.append(sent+puncs[i])
        else:
            sentences.append(sent)

    if sentences:
        return [sentences[0], sentences]
    else:
        return []


def _getQAPairs(case: Dict[str, Any]) -> List[Tuple[str, str, int]]:
    """
    Get the list of question and answer pairs from the raw data;
    Note that we only care about the question itself and the answer;
    Also to fit into our pipeline, we need to lower case answer "YES" and "NO",
    and add "unknown" if the answer is empty in the raw data
    """
    if "paragraphs" not in case:
        return []

    qa_pairs = []

    case_info = case["paragraphs"][0]
    orig_qas = case_info["qas"]

    for qa in orig_qas:
        q = qa["question"]
        answer = qa["answers"]

        start_index = answer[0]["answer_start"] if answer else -1

        # reformat the answer
        if not answer:
            a = "unknown"
            start_index = -1
        elif answer[0]["text"] == "YES":
            a = "yes"
        elif answer[0]["text"] == "NO":
            a = "no"
        else:
            a = answer[0]["text"]

        qa_pairs.append((q, a, start_index))

    return qa_pairs


def _getSentenceIndexByTokenIndex(token_index, sentences):
    length_tokens = 0

    for i, sent in enumerate(sentences):
        length_tokens += len(sent)
        current_index = i+1
        if current_index > len(sentences)-1:
            return -1
        if length_tokens <= token_index <= length_tokens+len(sentences[current_index]):
            return i+1


def _getSupportFactForOneQA(qa_pair, sentences, threshold=0.3):
    """
    The original CAIL2019 data does not contain support facts;
    We are going to extract such info by measuring the similarity and choose the
    top 3 as support facts;
    """
    support_facts = []
    top_sp_index = -1

    if qa_pair[1] == "unknown":
        return support_facts

    sent_similar_dict: Dict[int, float] = {}  # stores the sentence index and its similarity to the target
    if qa_pair[2] == -1:  # if the answer start index is -1
        # compare the similarity between question and sentences
        for i, sent in enumerate(sentences):
            similarity_score = SequenceMatcher(None, qa_pair[0], sent)
            sent_similar_dict[i] = similarity_score.ratio()
        return support_facts
    else:
        top_sp_index = _getSentenceIndexByTokenIndex(qa_pair[-1], sentences)
        if top_sp_index != -1:
            support_facts.append(top_sp_index)
        for i, sent in enumerate(sentences):
            # compare the similarity between question+answer and sentences
            similarity_score = SequenceMatcher(None, qa_pair[0]+qa_pair[1], sent)
            sent_similar_dict[i] = similarity_score.ratio()

    # sort thr similarity score by values in descending order
    sorted_sent_similar_dict = {
        k: v for k, v in sorted(sent_similar_dict.items(), key=lambda item: item[1], reverse=True)
    }
    # get the max similarity score, if it's beyond threshold, we will not put anything into support facts
    max_sp = 2
    current_sp = len(support_facts)

    for key, value in sorted_sent_similar_dict.items():
        if top_sp_index != key and value > threshold:
            support_facts.append(int(key))
            current_sp += 1
            if current_sp >= max_sp:
                break

    return support_facts


def _formattedExamples(rawDataFilePath: str, formattedFilePath: str) -> None:
    all_examples = []
    full_data = _readExamplesFromFile(rawDataFilePath)
    for data in tqdm(full_data):
        # get the question-answer tuple
        # the original dataset has one pair of qa for each sample
        # so, we are going to repeat the context for every qa pair the case has
        qa_pairs = _getQAPairs(data)
        context = _getCaseContext(data)
        if not context:
            continue

        for qa_pair in qa_pairs:
            case_data_dict = {}
            case_data_dict["context"] = [context]
            case_data_dict["question"] = qa_pair[0]
            case_data_dict["answer"] = qa_pair[1]

            # get the support facts
            supporting_facts = []
            support_facts_indices = _getSupportFactForOneQA(qa_pair, context[1], 0.1)
            for fact_index in support_facts_indices:
                supporting_facts.append([context[0], fact_index])
            case_data_dict["supporting_facts"] = supporting_facts

            all_examples.append(case_data_dict)

    # write out the data
    with open(formattedFilePath, 'w', encoding='utf8') as f:
        json.dump(all_examples, f, indent=4, ensure_ascii=False)





