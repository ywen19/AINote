"""
This is the module to preprocess the nCoV_100k train data;
We will spit this set to labelled train and validation set;
Main information used for sentiment classification will be the weibo content;
"""

import datetime as dt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import gzip
import pickle
from transformers import BertTokenizer
from tqdm import tqdm

from typing import Tuple, List

RANDOM_SEED = 2020  # random seed from sample code


def convert_to_csv(orig_path: str, save_path: str) -> None:
    """
    The original data is in GB2312, we will convert to utf-8 and save to csv for future retrieve;
    Inspired by: https://blog.csdn.net/weixin_43162364/article/details/106383248
    """
    with open(orig_path, 'r', encoding='GB2312', errors='ignore') as file:
        lines = file.readlines()
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(''.join(lines))


def convert_cn_date_to_datetime(x: str):
    # passed in time example: "01月01日 23:50"
    x = "2020年" + x.split("日")[0]
    for char in ["年", "月"]:
        converted_date = x.replace(char, "/")
    return converted_date


def data_cleanup(data_path: str) -> None:
    """
    We will mainly focus on weibo contents and sentiment labels,
    therefore the cleanup job will mainly focus on these two sections;
    Any sample point with nan or exception values will be removed
    """

    all_data: pd.DataFrame = pd.read_csv(data_path, encoding="utf-8", engine="python")

    publish_time = all_data["微博发布时间"].values
    convert_time = []
    for i, t in enumerate(publish_time):
        convert_time.append(convert_cn_date_to_datetime(t))

    all_data = all_data[["微博中文内容", "情感倾向"]]
    all_data["time"] = convert_time

    # remove samples if contents is empty
    all_data = all_data.dropna(axis=0)
    # remove sentiment labels that not fit in [-1, 0, 1]
    all_data = all_data[all_data["情感倾向"].isin(["-1", "0", "1"])]
    # drop duplicates if any
    all_data = all_data.drop_duplicates()

    # there left 91060 out of 100000 samples
    # save this final data set out
    all_data.to_csv(data_path)




def split_datasets(full_data: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray]:
    all_y = full_data["情感倾向"].values  # all label values
    all_x = full_data["微博中文内容"].values  # all features' values

    # split data frame to train set and test set use sklearn train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        all_x, all_y, test_size=test_size, random_state=RANDOM_SEED
    )
    return x_train, x_val, y_train, y_val


class Example(object):

    def __init__(self,
                 sentiment_type,
                 doc_tokens,
                 doc_id,
                 ):
        self.sentiment_type = sentiment_type  # [1, 0, -1]
        self.doc_tokens = doc_tokens  # tokens extracted from a document
        self.doc_id = doc_id


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 sentiment_type,
                 doc_id,
                 ):

        self.doc_tokens = doc_tokens  # tokens in the case, including question and contexts
        self.doc_input_ids = doc_input_ids  # token ids for all tokens in the case
        self.doc_input_mask = doc_input_mask  # only mask out placeholders to fil max token count
        self.sentiment_type = sentiment_type  # [1,0,-1]
        self.doc_id = doc_id


def is_whitespace(char: str) -> bool:
    """
    determine whether the input string is whitespace(or whitespace that equal to a tab) or line change;
    """
    if char == " " or char == "\t" or char == "\r" or char == "\n" or ord(char) == 0x202F:
        return True
    return False


def generate_examples(x: np.ndarray, y: np.ndarray, has_label=True) -> List[Example]:
    """
    x and y are extracted from the split_datasets method;
    y is the sentiment label, x is the weibo content
    """

    examples = []
    for i in range(len(x)):

        if type(x[i]) != str:
            continue

        doc_id = i
        # sentiment type: should be the type of [-1, 0, 1]
        # todo: need better logic for transferable, fix later!
        sentiment_type = y[i] if has_label else -2  # random number

        # Tokenizing textual contents
        doc_tokens = []
        entity_start_end_position = []

        char_to_word_offset = []  # Accumulated along all sentences, store id for each byte character built into vocab
        prev_is_whitespace = True  # store the state of the byte before the current byte on whether it's whitespace

        # for debug
        titles = set()
        sent = x[i]  # weibo content

        # accumulate it if a sample contains more than 1 paragraph
        para_start_position = len(doc_tokens)

        # step through each byte in the sentence for vocabulary build
        # the whole logic is so messy
        # todo: optimize this there is no need in our case to separate each byte by space and do nested conditions
        sent = " ".join(sent)  # separate each byte in the sentence by space
        sent += " "  # add a space at the end of the sentence
        for char in sent:
            if is_whitespace(char):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(char)
                else:
                    doc_tokens[-1] += char
                prev_is_whitespace = False
            # store the index for each byte stored in the token vocab
            char_to_word_offset.append(len(doc_tokens) - 1)

        # avoid too many tokens extracted from the current document
        # scripting this way does not clip the doc when the token reaches exactly 382
        # but reach the last sentence that causes the token amount gets above 382...
        if len(doc_tokens) > 382:
            break

        if i < 10:
            print("sentiment type: {}".format(sentiment_type))
            print("doc tokens: {}".format(doc_tokens))
            print("doc id: {}".format(doc_id))

        example = Example(
            sentiment_type=sentiment_type,
            doc_tokens=doc_tokens,
            doc_id=doc_id
        )
        examples.append(example)

    return examples


def convert_examples_to_features(examples: List[Example], tokenizer: BertTokenizer, max_seq_length: int) \
        -> List[InputFeatures]:

    features = []
    failed = 0
    for (example_index, example) in enumerate(tqdm(examples)):
        if not example:
            continue

        sentiment_type = [0] * 3  # convert the sentiment type [-1,0,1] to one hot
        doc_id = example.doc_id

        doc_tokens = []
        # below are maps to retrieve token index from original document to the case token collection, and vice versa
        # todo: being fair given the feature u want to achieve, dictionary may be a better choice
        # todo: it cuts down the lists amount, and retrieve index in a more straightforward way...
        orig_to_tok_index = []
        orig_to_tok_back_index = []
        tok_to_orig_index = []

        doc_tokens = ["[CLS]"]
        for (i, token) in enumerate(example.doc_tokens):
            if example.sentiment_type == -1:
                sentiment_type[0] = 1
            elif example.sentiment_type == 0:
                sentiment_type[1] = 1
            else:
                sentiment_type[2] = 1


            orig_to_tok_index.append(len(doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                doc_tokens.append(sub_token)
            orig_to_tok_back_index.append(len(doc_tokens) - 1)

        # do a clip on all tokens in the sample case, always end with [SEP]
        doc_tokens = doc_tokens[:max_seq_length - 1] + ["[SEP]"]
        doc_input_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
        # mask out question tokens for segment ids
        doc_input_mask = [1] * len(doc_input_ids)
        # paddings to fill the max sequence/query length
        while len(doc_input_ids) < max_seq_length:
            doc_input_ids.append(0)
            doc_input_mask.append(0)

        assert len(doc_input_ids) == max_seq_length
        assert len(doc_input_mask) == max_seq_length

        if example_index < 10:
            print("sentiment type {}".format(example.sentiment_type))
            print("all_doc_tokens {}".format(doc_tokens))
            print("doc_input_ids {}".format(doc_input_ids))
            print("doc_input_mask {}".format(doc_input_mask))
            print("sentiment_type {}".format(sentiment_type))
            print("tok_to_orig_index {}".format(tok_to_orig_index))
            print("doc id {}".format(doc_id))
            print(len(doc_input_ids))

        features.append(
            InputFeatures(doc_tokens=doc_tokens,
                          doc_input_ids=doc_input_ids,
                          doc_input_mask=doc_input_mask,
                          sentiment_type=sentiment_type,
                          doc_id=doc_id)
        )
    return features


def save_features_to_gzip(x: np.ndarray,
                          y: np.ndarray,
                          output_path: str = "",
                          max_seq_length: int = 512,
                          tokenizer_path: str = "bert-base-chinese",
                          ) -> None:
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    examples = generate_examples(x, y)
    features = convert_examples_to_features(examples, tokenizer, max_seq_length)

    with gzip.open(output_path, 'wb') as fout:
        pickle.dump(features, fout)






