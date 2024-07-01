import numpy as np

EPS = 1e-11


class SoftmaxCrossEntropyLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = np.zeros(0, dtype='f')
        
    def forward(self, logit, gt):
        
        self.gt = gt
        
        self.prob_hat = np.exp(logit)/np.sum(np.exp(logit), axis=1, keepdims=True)
        self.loss = -np.mean(np.sum(np.log(self.prob_hat+EPS)*self.gt, axis=1))
        
        label_hat = np.argmax(self.prob_hat, axis=1)
        label_true = np.argmax(self.gt, axis=1)
        self.acc = np.count_nonzero(label_hat==label_true)/self.gt.shape[0]
        
    def backward(self):
        return self.prob_hat-self.gt