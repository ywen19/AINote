import torch
from torch import nn
from torch.autograd import Variable

import numpy as np
class SimplePredictionLayer(nn.Module):
    def __init__(self, config):
        super(SimplePredictionLayer, self).__init__()
        self.input_dim = config.input_dim  # 768(hidden size in Bert output)

        self.sp_linear = nn.Linear(self.input_dim, 1)
        self.start_linear = nn.Linear(self.input_dim, 1)
        self.end_linear = nn.Linear(self.input_dim, 1)

        # linear transformation to predict answer type
        self.type_linear = nn.Linear(self.input_dim, config.label_type_num)   # yes/no/unknown/text contexts

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        # (batch, 512, 512)
        S = outer.size(1)
        if S <= self.cache_S:
            # if no valid span (of answers)
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        # triu 生成上三角矩阵，tril生成下三角矩阵，这个相当于生成了(512, 512)的矩阵表示开始-结束的位置，答案长度最长为15
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))  # (batch, 512, 512)
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, input_state):
        query_mapping = batch['query_mapping']  # (batch, 512) 不一定是512，可能略小
        context_mask = batch['context_mask']  # (batch_size, 512), bert里实际有输入的位置
        all_mapping = batch['all_mapping']  # (batch_size, 512, max_sent) 每个句子的token对应为1

        # to compare with batch['y1'] and batch['y2']
        # y1 and y2 are start and end position for answers(for all segments in a document)
        # note that if answer is not textual type, y1 and y2 are both IGNORE_INDEX
        start_logits = self.start_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)  # (batch, 512)
        end_logits = self.end_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)    # (batch, 512)
        # todo: the orig comment(next to below function) not correct
        # todo: should be (batch_size, 512, sent_count, input_dim=768)
        sp_state = all_mapping.unsqueeze(3) * input_state.unsqueeze(2)

        # prediction logits for each sentence in an input, the probability if the current sentence is support fact
        sp_state = sp_state.max(1)[0]  # (batch_size, sent, embed_dim)
        sp_logits = self.sp_linear(sp_state)  # (batch_size, sent, 1)

        # prediction logits for answer type
        type_state = torch.max(input_state, dim=1)[0]  # (batch_size, input_dim)
        type_logits = self.type_linear(type_state)  # (batch_size, answer_types=4)

        # 找结束位置用的开始和结束位置概率之和
        # None here adds an axis
        # (batch, 512, 1) + (batch, 1, 512) -> (batch, 512, 512)
        outer = start_logits[:, :, None] + end_logits[:, None]
        outer_mask = self.get_output_mask(outer)  # (batch, 512, 512)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if query_mapping is not None:   # 这个是query_mapping (batch, 512)
            outer = outer - 1e30 * query_mapping[:, :, None]    # 不允许预测query的内容

        # 这两句相当于找到了outer中最大值的i和j坐标 for answer span
        start_position = outer.max(dim=2)[0].max(dim=1)[1]  # (batch_size)
        end_position = outer.max(dim=1)[0].max(dim=1)[1]  # (batch_size)

        return start_logits, end_logits, type_logits, sp_logits.squeeze(2), start_position, end_position


class BertSupportNet(nn.Module):
    """
    joint train bert and graph fusion net
    """

    def __init__(self, config, encoder):
        super(BertSupportNet, self).__init__()
        # self.bert_model = BertModel.from_pretrained(config.bert_model)
        self.encoder = encoder  # bert for embedding
        self.graph_fusion_net = SupportNet(config)

    def forward(self, batch, debug=False):
        # doc_ids: token ids for all tokens in the cases(contexts and questions)
        # doc_mask: mask out placeholders to meet 512(max sequence length)
        # segment_ids: mask out placeholders and questions
        doc_ids, doc_mask, segment_ids = batch['context_idxs'], batch['context_mask'], batch['segment_idxs']
        # roberta不可以输入token_type_ids
        all_doc_encoder_layers = self.encoder(input_ids=doc_ids,
                                              token_type_ids=segment_ids,#可以注释
                                              attention_mask=doc_mask)[0]
        # last hidden state output from Bert
        batch['context_encoding'] = all_doc_encoder_layers  # (batch_size, sequence_length<=512, hidden_size=768)

        return self.graph_fusion_net(batch)


class SupportNet(nn.Module):
    """
    Packing Query Version
    """

    def __init__(self, config):
        super(SupportNet, self).__init__()
        self.config = config  # 就是args
        # self.n_layers = config.n_layers  # 2
        self.max_query_length = 50
        self.prediction_layer = SimplePredictionLayer(config)

    def forward(self, batch, debug=False):
        context_encoding = batch['context_encoding']  # (batch_size, sequence_length<=512, hidden_size=768)
        predictions = self.prediction_layer(batch, context_encoding)

        start_logits, end_logits, type_logits, sp_logits, start_position, end_position = predictions

        return start_logits, end_logits, type_logits, sp_logits, start_position, end_position
