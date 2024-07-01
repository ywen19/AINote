import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLoss(object):

    def __init__(self, num_input, num_output, trainable=True):
        """
        Apply a linear transformation to the incoming data: y = Wx + b
        Args:
            num_input: size of each input sample
            num_output: size of each output sample
            trainable: whether if this layer is trainable
        """

        self.num_input = num_input
        self.num_output = num_output
        self.trainable = trainable
        self.XavierInit()
        
        self.delta_W = 0.0
        self.delta_b = 0.0
        
    def get_one_hot_labels(self, labels):
        one_hot = np.zeros((labels.shape[0], self.num_output))
        x = np.arange(labels.shape[0])
        one_hot[x, labels] = 1.0
        return one_hot
    
    def softmax(self, value):
        exp = np.exp(value)
        return exp/np.sum(exp, axis=1, keepdims=True)
    
    def crossentropy_loss(self, one_hot_labels, prob_hat):
        loss = -np.mean(np.sum(one_hot_labels*np.log(prob_hat+EPS), axis=1))
        return loss
    
    def make_predict(self, prob_hat):
        label_hat = np.argmax(prob_hat, axis = 1)
        return label_hat
    
    def get_acc(self, labels, prob_hat):
        label_hat = self.make_predict(prob_hat)
        acc = np.count_nonzero(label_hat==labels)/labels.shape[0]
        return acc

    def forward(self, Input, labels):
        """
          Inputs: (minibatch)
          - Input: (batch_size, 784)
          - labels: the ground truth label, shape (batch_size, )
        """
        one_hot_labels = self.get_one_hot_labels(labels)
        
        # apply linear function and soft max
        linear = Input@self.W + self.b
        prob_hat = self.softmax(linear)
        
        # get loss and accuracy
        loss = self.crossentropy_loss(one_hot_labels, prob_hat)
        acc = self.get_acc(labels, prob_hat)

        return loss, acc, one_hot_labels, prob_hat

    def gradient_computing(self, one_hot_labels, prob_hat, Input):
        diff = one_hot_labels - prob_hat
        self.grad_W = Input.T.dot(diff)/one_hot_labels.shape[0]
        self.grad_b = np.sum(diff/one_hot_labels.shape[0], axis=0)
        #print(self.grad_b.shape)
        

    def XavierInit(self):
        """
        Initialize the weigths
        """
        raw_std = (2 / (self.num_input + self.num_output))**0.5
        init_std = raw_std * (2**0.5)
        self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
        self.b = np.random.normal(0, init_std, (1, self.num_output))
