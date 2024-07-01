from numpy.random import shuffle
import numpy as np
import torch

class DataIteratorPack(object):
    def __init__(self, features, bsz, device, entity_limit=80,
                 entity_type_dict=None, sequential=False,):
        self.bsz = bsz   # batch_size
        self.device = device
        self.features = features  # load from data_helper
        self.sequential = sequential
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

        context_idxs = torch.LongTensor(self.bsz, 256)
        context_mask = torch.LongTensor(self.bsz, 256)
        # sentiment label
        sentiment_type = torch.LongTensor(self.bsz, 3)
        # doc id (used only for test)
        doc_id = []

        while self.example_ptr < len(self.features):
            # get the features in the current batch
            start_id = self.example_ptr
            cur_bsz = min(self.bsz, len(self.features) - start_id)  # mainly for the last batch
            cur_batch = self.features[start_id: start_id + cur_bsz]

            for i in range(len(cur_batch)):
                feature = cur_batch[i]
                doc_id.append(feature.doc_id)
                context_idxs[i].copy_(torch.Tensor(feature.doc_input_ids))
                context_mask[i].copy_(torch.Tensor(feature.doc_input_mask))
                sentiment_type[i].copy_(torch.Tensor(feature.sentiment_type))

            input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)  # length od doc(context and query)
            max_c_len = int(input_lengths.max())

            self.example_ptr += cur_bsz  # step to the start index of the next batch

            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
                'sentiment_type': sentiment_type[:cur_bsz],
                'doc_id': doc_id[:cur_bsz]
            }
