import numpy as np

class EuclideanLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = 0.
        
    def forward(self, logit, gt):
        self.logit = logit
        self.gt = gt
        
        dist = np.sqrt(np.sum(np.square(self.gt-self.logit), axis=1))
        self.loss = np.mean(dist)
        
        label_hat = np.argmax(logit, axis=1)
        label_true = np.argmax(self.gt, axis=1)
        self.acc = np.count_nonzero(label_hat==label_true)/self.gt.shape[0]
        
    def backward(self):
        return self.logit-self.gt