import numpy as np
from scipy.special import expit
from DataProcessing import getTrainingData, getTestingData


X,Y= getTrainingData()

#Sigmoid Activation function
def sigmoid(inputs):
    return expit(inputs)

#Softmax Activation Functions
def softmax(A):
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y


def feedForward(X,W1,b1,W2,b2,training):
    outputFirst=X.dot(W1)+b1
    #Z= sigmoid(outputFirst)
    Z=np.tanh(outputFirst)
    #dropout
    if training:
        m2 = np.random.binomial(1, 0.5, size=Z.shape)
    else:
        m2 = 0.5
    Z *= m2
    outputSecond = Z.dot(W2) + b2
    P=softmax(outputSecond)
    return P,Z,m2



