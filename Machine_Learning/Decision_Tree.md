# Decision Tree

## Classical target problem for decision tree learning：
    * Classification problems with non-numeric features;
    * Discrete features;
    * No need of considering similarity between attributes;
    * No order in features; such as [male, female] in real life, there is no priority among these two classes.  
   
For continuous features, we need to first discrete the values before flushing the features into the training. Common 
techniques can be seen from [Basis Section 1: Data Discretization](./Basis.md):  
**Entropy and impurity calculation rely only on the probability of a class, so if a feature is textual, we do 
not need to pre-process it to discrete numeric values.**