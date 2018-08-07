# Iris classification using ND4J

Trying to use ND4J to solve multiclass classification problem. I could use the DL4J API directly, but wanted to deep dive into the math for LR.

NOTE: In case you came to this project looking for an example on using ND4J API directly - note that though the matrix math works (vector sizes through various layers dont throw errors) the precision right now is terrible. Working on it.

### Math used in the code (Logistic Regression)
![Math in Logistic Regression](https://raw.githubusercontent.com/pkamath2/iris_classifier_nd4j/master/src/main/resources/LR-math.jpg)  

### Legend for the diagram:  
Super script [l] : Layer number.  
W : Weights for the layer.  
b : Bias  
Z : Output from one node.   
g : Transformation (In this case, tanh for intermediate layers and softmax for last layer).  
A : Output after transformation  
Y : Final output  

### Code Explained 
DataLoader: Loads data from csv under the resources folder.    
IrisClassifier: Has the main method which runs the fit and predict methods. 

