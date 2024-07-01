import numpy as np

EPSILON = 1e-11

class AdamOptimizer(object):
    def __init__(self, model, beta1=0.9, beta2=0.99, W_threshold=EPSILON, count_threshold=600, step_size=0.01):
        self.model = model
        self.W_threshold = W_threshold
        self.count_threshold = count_threshold
        
        # first-order and second-order exponential decay
        self.beta1 = beta1
        self.beta2 = beta2
        
        # step size for each iteration
        self.step_size = step_size
        
        # mean and uncertained variance from previous timestep(iteration)
        self.mean_grad_W, self.var_grad_W, self.mean_grad_b, self.var_grad_b = 0.0, 0.0, 0.0, 0.0
        
        # count the continuos time when delta on weight gradient is smaller than a given threshold
        self.converge_count = 0
        
        # for testing
        self.run_count = 0
        
        
        
    def is_converge(self, new_W, new_b):
        # if delta is smaller than a given threshold for continuous times, we see it as converge
        dist_W = np.linalg.norm(self.old_W - new_W)
        dist_b = np.linalg.norm(self.old_b - new_b)
        
        if dist_W < self.W_threshold and dist_b < self.W_threshold:
            self.converge_count += 1
        else: 
            self.converge_count =0
            
        if self.converge_count > self.count_threshold:
            return True
        return False
        

    
    def step(self, t=1):
        self.run_count += 1
        
        # t is the current iteration
        layer = self.model
        if layer.trainable:
            if not hasattr(self, 'old_W'):
                self.old_W = layer.W
            if not hasattr(self, 'old_b'):
                self.old_b = layer.b
            
            # get the first-order estimate(mean)
            self.mean_grad_W = self.beta1*self.mean_grad_W + (1-self.beta1)*layer.grad_W
            self.mean_grad_b = self.beta1*self.mean_grad_b + (1-self.beta1)*layer.grad_b

            # get the second-order estimate(variance)
            self.var_grad_W = self.beta2*self.var_grad_W + (1-self.beta2)*(layer.grad_W**2)
            self.var_grad_b = self.beta2*self.var_grad_b + (1-self.beta2)*(layer.grad_b**2)

            # bias correction
            mean_grad_W_corr = self.mean_grad_W/(1-self.beta1**t + EPSILON)
            mean_grad_b_corr = self.mean_grad_b/(1-self.beta1**t + EPSILON)
            var_grad_W_corr = self.var_grad_W/(1-self.beta2**t + EPSILON)
            var_grad_b_corr = self.var_grad_b/(1-self.beta2**t + EPSILON)
            

            # update weight and bias
            layer.W += self.step_size * (mean_grad_W_corr/(np.power(var_grad_W_corr, 0.5)+EPSILON))
            layer.b += self.step_size * (mean_grad_b_corr/(np.power(var_grad_b_corr, 0.5)+EPSILON))
            
            if self.is_converge(layer.W, layer.b):
                return True
            else:
                self.old_W = layer.W
                self.old_b = layer.b
                return False
            
                
                
                
                
                
            