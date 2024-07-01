import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

"""
Bert + LSTM for classification;
Inspire by: https://github.com/mmahdim/BERT-sentiment-PyTorch/blob/main/MP2_3.ipynb
"""


class SentimentClassifier(nn.Module):
    def __init__(self, bert_model_name, hidden_size, device, n_layers=2, dropout=0.5, num_classes=3):
        super(SentimentClassifier, self).__init__()

        self.device = device

        self.bert = BertModel.from_pretrained(bert_model_name)
        embedding_size = self.bert.config.to_dict()["hidden_size"]

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):
        doc_ids, doc_mask = batch['context_idxs'], batch['context_mask']
        with torch.no_grad():
            # embedded: [batch size, sequence length, embed_dim]
            embedded = self.bert(input_ids=doc_ids, attention_mask=doc_mask)[0]  # last hidden state

        # h, c: [2(bidirectional)*num_layers, batch size, hidden size]
        _, (h, c) = self.lstm(embedded)

        # through linear layer
        # concatenate the two hidden states from bidirectional LSTM
        logits = self.fc(torch.cat((h[0], h[1]), dim=1))
        logits = self.softmax(logits)

        return logits

