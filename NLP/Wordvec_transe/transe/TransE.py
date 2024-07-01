# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from load_data import PyTorchTrainDataLoader
import ctypes
import os
import time
import numpy as np

class Config(object):

    def __init__(self, p_norm, hidden_size, nbatches, trainTimes, margin, learningRate, use_gpu):
        self.p_norm = p_norm
        self.hidden_size = hidden_size
        self.nbatches = nbatches
        self.entity = 0
        self.relation = 0
        self.trainTimes = trainTimes
        self.margin = margin
        self.learningRate = learningRate
        self.use_gpu = use_gpu

def to_var(x, use_gpu):
		if use_gpu:
			return Variable(torch.from_numpy(x).cuda())
		else:
			return Variable(torch.from_numpy(x))

class TransE(nn.Module):

    def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None):
        '''
        Paramters:
        p_norm: 1 for l1-norm, 2 for l2-norm
        norm_flag: if use normalization
        margin: margin in loss function
        '''
        super(TransE, self).__init__()

        self.dim = dim
        self.margin = margin
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            
        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False


    def _calc(self, h, t, r):
        # TO DO: implement score function
        # Hint: you can use F.normalize and torch.norm functions
        # inspired by: https://github.com/mklimasz/TransE-PyTorch/
        if self.norm_flag: # normalize embeddings with l2 norm
            h = F.normalize(h, p=2, dim=-1)
            t = F.normalize(t, p=2, dim=-1)
            r = F.normalize(r, p=2, dim=-1)
        score = h + r - t #[2, batch_size]
        return torch.norm(score, p=self.p_norm, dim=-1)

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h ,t, r)
        return score

    def predict(self, data):
        score = self.forward(data)
        return score.cpu().data.numpy()
    
    def loss(self, pos_score, neg_score):
        # TO DO: implement loss function
        # Hint: consider margin

        # the loss is calculated by taking the larger value among 0 and (margin+pos_score-neg_score)
        # which is basically a ReLU, so we will directly use ReLU
        # sort of inspired by the source code for torch marginrankloss
        if self.margin_flag:
            return F.relu(self.margin + (pos_score - neg_score).mean(), inplace=True)
        return F.relu((pos_score - neg_score).mean(), inplace=True)
        


def main():
    config = Config()
    train_dataloader = PyTorchTrainDataLoader(
                            in_path = "./data/", 
                            nbatches = config.nbatches,
                            threads = 8)
    
    transe = TransE(
            ent_tot = train_dataloader.get_ent_tot(),
            rel_tot = train_dataloader.get_rel_tot(),
            dim = config.hidden_size, 
            p_norm = config.p_norm, 
            norm_flag = True,
            margin=config.margin)
    
    optimizier = optim.SGD(transe.parameters(), lr=config.learningRate)

    if config.use_gpu:
        transe.cuda()
    
    for times in range(config.trainTimes):
        ep_loss = 0.
        for data in train_dataloader:
            optimizier.zero_grad()
            score = transe({
                    'batch_h': to_var(data['batch_h'], config.use_gpu).long(),
                    'batch_t': to_var(data['batch_t'], config.use_gpu).long(),
                    'batch_r': to_var(data['batch_r'], config.use_gpu).long()})
            pos_score, neg_score = score[0], score[1]
            loss = transe.loss(pos_score, neg_score)
            loss.backward()
            optimizier.step()
            ep_loss += loss.item()
        print("Epoch %d | loss: %f" % (times+1, ep_loss))
    
    print("Finish Training")
    
    f = open("entity2vec.txt", "w")
    enb = transe.ent_embeddings.weight.data.cpu().numpy()
    for i in enb:
        for j in i:
            f.write("%f\t" % (j))
        f.write("\n")
    f.close()

    f = open("relation2vec.txt", "w")
    enb = transe.rel_embeddings.weight.data.cpu().numpy()
    for i in enb:
        for j in i:
            f.write("%f\t" % (j))
        f.write("\n")
    f.close()

            
if __name__ == "__main__":
    main()


