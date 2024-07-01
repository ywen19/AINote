import torch
import numpy as np
from numpy.random import shuffle

IGNORE_INDEX = -100  # for non-textual answers mapping


class DataIteratorPack(object):
    def __init__(self, features, example_dict, bsz, device, sent_limit, entity_limit,
                 entity_type_dict=None, sequential=False,):
        self.bsz = bsz   # batch_size
        self.device = device
        self.features = features  # load from data_helper
        self.example_dict = example_dict  # load from data_helper
        # self.entity_type_dict = entity_type_dict
        self.sequential = sequential
        self.sent_limit = sent_limit
        # self.para_limit = 4 
        # self.entity_limit = entity_limit
        self.example_ptr = 0
        if not sequential:  # if sequential, random
            shuffle(self.features)  

    def refresh(self):
        # refresh the start example index to 0 -> basically refresh the iterator
        self.example_ptr = 0
        if not self.sequential:
            shuffle(self.features)

    def empty(self):
        # bool, if feature empty
        return self.example_ptr >= len(self.features)

    def __len__(self):
        # get the amount of batches
        return int(np.ceil(len(self.features)/self.bsz))

    def __iter__(self):
        # BERT input
        context_idxs = torch.LongTensor(self.bsz, 512)
        context_mask = torch.LongTensor(self.bsz, 512)
        segment_idxs = torch.LongTensor(self.bsz, 512)
        # query mapping: 1 for query sentence span
        query_mapping = torch.Tensor(self.bsz, 512).cuda(self.device)
        # start_mapping: start position(token index) of mapping a sentence that is support fact
        start_mapping = torch.Tensor(self.bsz, self.sent_limit, 512).cuda(self.device)
        # all_mapping: 1 for the sentence token positions
        all_mapping = torch.Tensor(self.bsz, 512, self.sent_limit).cuda(self.device)

        # Label tensor for textual answers
        y1 = torch.LongTensor(self.bsz).cuda(self.device)   
        y2 = torch.LongTensor(self.bsz).cuda(self.device)
        q_type = torch.LongTensor(self.bsz).cuda(self.device)
        # is_support: flag 1 if the ith example's jth sentence is a support fact
        is_support = torch.FloatTensor(self.bsz, self.sent_limit).cuda(self.device)

        while True:
            # break if the features are empty
            if self.example_ptr >= len(self.features):
                break
            start_id = self.example_ptr  
            cur_bsz = min(self.bsz, len(self.features) - start_id)   # mainly for the last batch
            cur_batch = self.features[start_id: start_id + cur_bsz]
            # todo: is the below sorting batch by the actual length of document??
            cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)  

            ids = []
            max_sent_cnt = 0
            max_entity_cnt = 0
            # zero out all the mapping
            for mapping in [start_mapping, all_mapping,  query_mapping]:
                mapping.zero_()   
           
            is_support.fill_(0)

            for i in range(len(cur_batch)):
                # step through each sample in the current batch
                case = cur_batch[i]   # case is an instance of InputFeatures
                # print(f'all_doc_tokens is {case.doc_tokens}')
                context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))  # token ids for all tokens in the case
                context_mask[i].copy_(torch.Tensor(case.doc_input_mask))  # mask out placeholders
                segment_idxs[i].copy_(torch.Tensor(case.doc_segment_ids))  # mask out placeholders and question(query)

                # where the query sentence is
                for j in range(case.sent_spans[0][0] - 1):
                    query_mapping[i, j] = 1
                # answer start and end position
                if case.ans_type == 0:  # if answer is textual contents
                    if len(case.end_position) == 0:
                        y1[i] = y2[i] = 0   
                    elif case.end_position[0] < 512:
                        y1[i] = case.start_position[0]   
                        y2[i] = case.end_position[0]
                    else:
                        y1[i] = y2[i] = 0
                    q_type[i] = 0
                elif case.ans_type == 1:  # yes
                    y1[i] = IGNORE_INDEX  
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 1  
                elif case.ans_type == 2:  # no
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 2
                elif case.ans_type == 3:  # unknown
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 3

                for j, sent_span in enumerate(case.sent_spans[:self.sent_limit]):
                    # note we use limit to avoid the support fact out of the max length we allow for each example
                    is_sp_flag = j in case.sup_fact_ids   
                    start, end = sent_span
                    if start < end:  # avoid empty sentence
                        is_support[i, j] = int(is_sp_flag)   
                        all_mapping[i, start:end+1, j] = 1   
                        start_mapping[i, j, start] = 1

                ids.append(case.qas_id)
                max_sent_cnt = max(max_sent_cnt, len(case.sent_spans))

            input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)  # length od doc(context and query)
            max_c_len = int(input_lengths.max())

            self.example_ptr += cur_bsz  # step to the start index of the next batch

            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous(),
                'query_mapping': query_mapping[:cur_bsz, :max_c_len].contiguous(),
                'y1': y1[:cur_bsz],
                'y2': y2[:cur_bsz],
                'ids': ids,
                'q_type': q_type[:cur_bsz],
                'start_mapping': start_mapping[:cur_bsz, :max_sent_cnt, :max_c_len],
                'all_mapping': all_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
                
                'is_support': is_support[:cur_bsz, :max_sent_cnt].contiguous(),
            }
