from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import gzip
import pickle
from tqdm import tqdm
from transformers import BertTokenizer




class Example(object):

    def __init__(self,
                 qas_id,
                 qas_type,
                 doc_tokens,
                 question_text,
                 sent_num,
                 sent_names,
                 sup_fact_id,
                 para_start_end_position,
                 sent_start_end_position,
                 entity_start_end_position,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id  # id to the case sample
        self.qas_type = qas_type  # empty string...but not suppose to be in logic
        self.doc_tokens = doc_tokens  # tokens extracted from a document
        self.question_text = question_text  # question type
        self.sent_num = sent_num  # amount of sentences in a document
        self.sent_names = sent_names  # list of (title, sentence local id in the paragraph)
        self.sup_fact_id = sup_fact_id  # ids for the sentence that are supporting facts to an answer
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.entity_start_end_position = entity_start_end_position  # empty list
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position  # answer start position
        self.end_position = end_position  # answer end position


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 query_tokens,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 sent_spans,
                 sup_fact_ids,
                 ans_type,
                 token_to_orig_map,
                 start_position=None,
                 end_position=None):

        self.qas_id = qas_id  # id to the case sample
        self.doc_tokens = doc_tokens  # tokens in the case, including question and contexts
        self.doc_input_ids = doc_input_ids  # token ids for all tokens in the case
        self.doc_input_mask = doc_input_mask  # only mask out placeholders to fil max token count
        self.doc_segment_ids = doc_segment_ids  # mask out placeholders and questions

        self.query_tokens = query_tokens  # tokens for question, starts with [CLS] and ends with [SEP]
        self.query_input_ids = query_input_ids  # token ids for tokens from question
        self.query_input_mask = query_input_mask  # only mask out placeholders to fil max token count
        self.query_segment_ids = query_segment_ids  # all mask out

        self.sent_spans = sent_spans  # start and end token index for a sentence; index from doc_tokens
        self.sup_fact_ids = sup_fact_ids  # list of sentence ids
        self.ans_type = ans_type  # 0-textual apn, 1-yes, 2-no, 3-unknown
        self.token_to_orig_map = token_to_orig_map  # mapping between all doc token index to original context index

        self.start_position = start_position
        self.end_position = end_position




def check_in_full_paras(answer, paras):
    """
    todo: function not used
    """
    full_doc = ""
    for p in paras:
        full_doc += " ".join(p[1])
    return answer in full_doc


def read_examples( full_file):
    # read the original data in(json format)
    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    # define a method inside a method is usually for encapsulation
    # in this case I'd say it is not highly recommended to do so...
    def is_whitespace(c: str) -> bool:
        """
        determine whether the input string is whitespace(or whitespace that equal to a tab) or line change;
        """
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    cnt = 0
    examples = []
    for case in tqdm(full_data):
        # step through each sample in the data

        # Step 1: Sample information that can directly get from the original data
        key = case['_id']
        # qas_type: should be the type of answers(texts/YesNo/unknown)
        # the original code commented as case['type'], but the data dict actually contains no such key
        qas_type = ""  # case['type']
        # element in sup_facts: (first sentence of the sample case, sentence id that supports the answer)
        # could be empty set with the answer as unknown (no supporting facts)
        sup_facts = set([(sp[0], sp[1]) for sp in case['supporting_facts']])
        # sup_titles stores the first sentence if a sample case
        # could be empty set with the answer as unknown (no supporting facts)
        sup_titles = set([sp[0] for sp in case['supporting_facts']]) 
        orig_answer_text = case['answer']

        # Step 2: Tokenizing textual contents
        sent_id: int = 0  # count of cases if the answer is from original case document
        doc_tokens = []
        sent_names = []
        sup_facts_sent_id = []
        # start and end position for a sentence inside a case sample(accumulated by sentences across multi paragraphs)
        sent_start_end_position = []
        para_start_end_position = []
        entity_start_end_position = []
        ans_start_position, ans_end_position = [], []
        # variable name confusing, todo: clearer naming -> ANSWER_IS_SPAN
        JUDGE_FLAG = orig_answer_text == 'yes' or orig_answer_text == 'no' or orig_answer_text=='unknown'  or orig_answer_text==""
        FIND_FLAG = False  # if answer in the case document

        char_to_word_offset = []  # Accumulated along all sentences, store id for each byte character built into vocab
        prev_is_whitespace = True  # store the state of the byte before the current byte on whether it's whitespace

        # for debug
        titles = set()
        para_data = case['context']  # [[first sentence, [sentences]]]
        for paragraph in para_data:  
            title = paragraph[0]
            sents = paragraph[1]   

            titles.add(title)
            # if the title is in support facts then True else False
            # the sup_titles are titles in the support fact
            # and all support facts start with the first sentence(title)
            is_gold_para = 1 if title in sup_titles else 0
            # the start position of a sentence in a sample
            # accumulate it if a sample contains more than 1 paragraph
            para_start_position = len(doc_tokens)  

            for local_sent_id, sent in enumerate(sents):
                # avoid the paragraphs in one sample got too long
                if local_sent_id >= 100:  
                    break

                # Determine the global sent id for supporting facts
                local_sent_name = (title, local_sent_id)   
                sent_names.append(local_sent_name)
                # if the pair exists in support facts, add the sent_id to the sup_facts_sent_id
                if local_sent_name in sup_facts:
                    sup_facts_sent_id.append(sent_id)   
                sent_id += 1   
                sent = " ".join(sent)  # separate each byte in the sentence by space
                sent += " "  # add a space at the end of the sentence

                sent_start_word_id = len(doc_tokens)           
                sent_start_char_id = len(char_to_word_offset)
                # step through each byte in the sentence for vocabulary build
                # the whole logic is so messy
                # todo: optimize this there is no need in our case to separate each byte by space and do nested conditions
                for c in sent:  
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    # store the index for each byte stored in the token vocab
                    char_to_word_offset.append(len(doc_tokens) - 1)
                # end position of the sentence in one case sample(multi paragraph may involve)
                sent_end_word_id = len(doc_tokens) - 1  
                sent_start_end_position.append((sent_start_word_id, sent_end_word_id))  

                # Answer char position(only apply when the answer is textual span)
                # it stores the occurrence indices of a answer if it appears in the current sentence
                answer_offsets = []
                offset = -1
                # insert space only because the previous vocab build insert spaces to the sentence
                tmp_answer = " ".join(orig_answer_text)
                while True:
                    # find the occurrences of tha answer in the sentence
                    # return -1 if cannot find
                    offset = sent.find(tmp_answer, offset + 1)
                    if offset != -1:
                        answer_offsets.append(offset)   
                    else:
                        break

                # answer_offsets = [m.start() for m in re.finditer(orig_answer_text, sent)]
                # if the answer is textual spans, we have no records on sentence and answer have a match,
                # and if the answer could be found in the sentence
                if not JUDGE_FLAG and not FIND_FLAG and len(answer_offsets) > 0:
                    FIND_FLAG = True   
                    for answer_offset in answer_offsets:
                        start_char_position = sent_start_char_id + answer_offset   
                        end_char_position = start_char_position + len(tmp_answer) - 1  
                       
                        ans_start_position.append(char_to_word_offset[start_char_position])
                        ans_end_position.append(char_to_word_offset[end_char_position])
                # avoid too many tokens extracted from the current document
                # scripting this way does not clip the doc when the token reaches exactly 382
                # but reach the last sentence that causes the token amount gets above 382...
                if len(doc_tokens) > 382:
                    break
            para_end_position = len(doc_tokens) - 1
            
            para_start_end_position.append((para_start_position, para_end_position, title, is_gold_para))  

        if len(ans_end_position) > 1:
            cnt += 1    
        if key < 10:
            print("qid {}".format(key))
            print("qas type {}".format(qas_type))
            print("doc tokens {}".format(doc_tokens))
            print("question {}".format(case['question']))
            print("sent num {}".format(sent_id+1))
            print("sup face id {}".format(sup_facts_sent_id))
            print("para_start_end_position {}".format(para_start_end_position))
            print("sent_start_end_position {}".format(sent_start_end_position))
            print("entity_start_end_position {}".format(entity_start_end_position))
            print("orig_answer_text {}".format(orig_answer_text))
            print("ans_start_position {}".format(ans_start_position))
            print("ans_end_position {}".format(ans_end_position))
       
        example = Example(
            qas_id=key,
            qas_type=qas_type,
            doc_tokens=doc_tokens,
            question_text=case['question'],
            # todo: sent_num -> if u really go through the print history, u would notice there is no need for +1!
            # todo: cos the id in code is not actually id, but a count, adds 1 to itself once stepping to a new sentence
            # todo: unless it functions as placeholder then it is stupid mistake lead by confusing naming
            sent_num=sent_id + 1,
            sent_names=sent_names,
            sup_fact_id=sup_facts_sent_id,
            para_start_end_position=para_start_end_position, 
            sent_start_end_position=sent_start_end_position,
            entity_start_end_position=entity_start_end_position,
            orig_answer_text=orig_answer_text,
            start_position=ans_start_position,   
            end_position=ans_end_position)
        examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
    """
    Args:
        examples (List[Example]): properties for each document;
        tokenizer (BertTokenizer): the pre-trained tokenizer to process our data to features;
        max_seq_length (int):
        max_query_length (int): max length if a question;

    Return: List[Optional[InputFeatures]], if not fail, optional could be removed
    """

    features = []
    failed = 0
    for (example_index, example) in enumerate(tqdm(examples)):  
        if example.orig_answer_text == 'yes':
            ans_type = 1
        elif example.orig_answer_text == 'no':
            ans_type = 2
        elif example.orig_answer_text == 'unknown':
            ans_type = 3
        else:
            ans_type = 0  # for answer that is textual spans

        # Tokenize the question
        query_tokens = ["[CLS]"]  # Bert token, stands for classification
        for token in example.question_text.split(' '):
            query_tokens.extend(tokenizer.tokenize(token))
        # the last element should always be '[SEP]' and to avoid question tokens being too long
        if len(query_tokens) > max_query_length - 1:
            query_tokens = query_tokens[:max_query_length - 1]
        query_tokens.append("[SEP]")

        # todo: names of below variants are confusing, there r better ways to rename
        sentence_spans = []
        all_doc_tokens = []
        # below are maps to retrieve token index from original document to the case token collection, and vice versa
        # todo: being fair given the feature u want to achieve, dictionary may be a better choice
        # todo: it cuts down the lists amount, and retrieve index in a more straightforward way...
        orig_to_tok_index = []
        orig_to_tok_back_index = []
        tok_to_orig_index = [0] * len(query_tokens)

        all_doc_tokens = ["[CLS]"]
        # todo: seriously why not do this in the upper loop this means u don't need to run the same thing twice
        # todo: code needs optimization!!
        for token in example.question_text.split(' '):
            all_doc_tokens.extend(tokenizer.tokenize(token))
        if len(all_doc_tokens) > max_query_length - 1:
            all_doc_tokens = all_doc_tokens[:max_query_length - 1]
        all_doc_tokens.append("[SEP]")

        for (i, token) in enumerate(example.doc_tokens):    
            orig_to_tok_index.append(len(all_doc_tokens))  
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)    
                all_doc_tokens.append(sub_token)
            orig_to_tok_back_index.append(len(all_doc_tokens) - 1)

        def relocate_tok_span(orig_start_position, orig_end_position, orig_text):
            # if no start position, basically means the answer is not textual context
            if orig_start_position is None:  
                return 0, 0

            tok_start_position = orig_to_tok_index[orig_start_position]
            if orig_end_position < len(example.doc_tokens) - 1:
                # +1 since the first element is the length of tokens from question
                tok_end_position = orig_to_tok_index[orig_end_position + 1] - 1
            else:
                # do a clipping
                tok_end_position = len(all_doc_tokens) - 1  
            
            return _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, orig_text)

        # only applicable to answers that fall into textual category
        ans_start_position, ans_end_position = [], []
        for ans_start_pos, ans_end_pos in zip(example.start_position, example.end_position):
            # tokenized answer spans that better match the annotated answer
            s_pos, e_pos = relocate_tok_span(ans_start_pos, ans_end_pos, example.orig_answer_text)
            ans_start_position.append(s_pos)  
            ans_end_position.append(e_pos)

        for sent_span in example.sent_start_end_position:
            # sent_span: (sent_start_word_id, sent_end_word_id)
            # clip if the start is already out of the max length, jump if the sentence has no more than 1 word
            if sent_span[0] >= len(orig_to_tok_index) or sent_span[0] >= sent_span[1]:
                continue  
            sent_start_position = orig_to_tok_index[sent_span[0]] 
            sent_end_position = orig_to_tok_back_index[sent_span[1]] 
            sentence_spans.append((sent_start_position, sent_end_position))

        # do a clip on all tokens in the sample case, always end with [SEP]
        all_doc_tokens = all_doc_tokens[:max_seq_length - 1] + ["[SEP]"]
        doc_input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)
        query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        # mask out question tokens for segment ids
        doc_input_mask = [1] * len(doc_input_ids)
        doc_segment_ids = [0] * len(query_input_ids) + [1] * (len(doc_input_ids) - len(query_input_ids))
        # paddings to fill the max sequence/query length
        while len(doc_input_ids) < max_seq_length:
            doc_input_ids.append(0)
            doc_input_mask.append(0)
            doc_segment_ids.append(0)

        query_input_mask = [1] * len(query_input_ids)
        query_segment_ids = [0] * len(query_input_ids)

        while len(query_input_ids) < max_query_length:
            query_input_ids.append(0)
            query_input_mask.append(0)
            query_segment_ids.append(0)

        assert len(doc_input_ids) == max_seq_length
        assert len(doc_input_mask) == max_seq_length
        assert len(doc_segment_ids) == max_seq_length
        assert len(query_input_ids) == max_query_length
        assert len(query_input_mask) == max_query_length
        assert len(query_segment_ids) == max_query_length
        # this is still clipping
        sentence_spans = get_valid_spans(sentence_spans, max_seq_length)

        sup_fact_ids = example.sup_fact_id
        # if the support fact sentence is out of the clipping, ignore
        sent_num = len(sentence_spans)
        sup_fact_ids = [sent_id for sent_id in sup_fact_ids if sent_id < sent_num]
        if len(sup_fact_ids) != len(example.sup_fact_id):
            # todo: failed not used anywhere, if is a debug feature please comment
            failed += 1
        if example.qas_id < 10:
            print("qid {}".format(example.qas_id))
            print("all_doc_tokens {}".format(all_doc_tokens))
            print("doc_input_ids {}".format(doc_input_ids))
            print("doc_input_mask {}".format(doc_input_mask))
            print("doc_segment_ids {}".format(doc_segment_ids))
            print("query_tokens {}".format(query_tokens))
            print("query_input_ids {}".format(query_input_ids))
            print("query_input_mask {}".format(query_input_mask))
            print("query_segment_ids {}".format(query_segment_ids))
            print("sentence_spans {}".format(sentence_spans))
            print("sup_fact_ids {}".format(sup_fact_ids))
            print("ans_type {}".format(ans_type))
            print("tok_to_orig_index {}".format(tok_to_orig_index))
            print("ans_start_position {}".format(ans_start_position))
            print("ans_end_position {}".format(ans_end_position))

        features.append(
            InputFeatures(qas_id=example.qas_id,
                          doc_tokens=all_doc_tokens,
                          doc_input_ids=doc_input_ids,
                          doc_input_mask=doc_input_mask,
                          doc_segment_ids=doc_segment_ids,
                          query_tokens=query_tokens,
                          query_input_ids=query_input_ids,
                          query_input_mask=query_input_mask,
                          query_segment_ids=query_segment_ids,
                          sent_spans=sentence_spans,
                          sup_fact_ids=sup_fact_ids,
                          ans_type=ans_type,
                          token_to_orig_map=tok_to_orig_index,
                          start_position=ans_start_position,
                          end_position=ans_end_position)
        )
    return features


def _largest_valid_index(spans, limit):
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx


def get_valid_spans(spans, limit):
    # this is still clipping
    new_spans = []
    for span in spans:
        if span[1] < limit:
            new_spans.append(span)
        else:
            new_span = list(span)
            new_span[1] = limit - 1
            new_spans.append(tuple(new_span))
            break
    return new_spans


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_output", required=True, type=str)
    parser.add_argument("--feature_output", required=True, type=str)

    # Other parameters
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=15, type=int, help="Batch size for predictions.")
    parser.add_argument("--full_data", type=str, required=True)   
    parser.add_argument('--tokenizer_path',type=str,required=True)


    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    examples = read_examples( full_file=args.full_data)
    with gzip.open(args.example_output, 'wb') as fout:
        pickle.dump(examples, fout)

    features = convert_examples_to_features(examples, tokenizer, max_seq_length=512, max_query_length=50)
    with gzip.open(args.feature_output, 'wb') as fout:
        pickle.dump(features, fout)











