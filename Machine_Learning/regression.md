# Regression  

## 1. Regression

1. There are in general two types of predictions: classification and regression. Fundamentally, classification is about 
predicting a label, whilst regression is about predicting a quantity. Therefore, when it comes to classification tasks, 
we should not use regression models(except logistic regression, which is 2 binary classification model).
2. Regression is a statistical model to analyse and describe the correlation between variables.
3. Note: prediction results by regression models are usually continuous data(but can be discrete or binary):
    * 线性回归模型最后得到的是一个预测函数，因此可以预测连续值；逻辑回归得到的是一个分类器，最后的输出一般是一个类别(一般多为二分类问题)，
   典型的分类器是Sigmoid;


## 2. Linear Regression  
 
因变量和自变量之间是线性关系，就可以使用线性回归来建模。参考
$$\hat{y_{i}} = b_{1} + b_{2}x_{i}$$

