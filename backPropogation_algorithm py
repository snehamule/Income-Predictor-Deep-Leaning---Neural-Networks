import numpy as np
from DataProcessing import getTrainingData, getTestingData
from FeedForward import feedForward


Xtrain, Ytrain= getTrainingData()
Xtest, Ytest = getTestingData()

M = 7  # Number of Nodes at Hidden Layer
D = Xtrain.shape[1]  # input Size
K = Ytrain.shape[1]

# choosing random W1 and W2
# initialize bias 0
W1= np.random.randn(D,M)
b1=np.zeros(M)
W2= np.random.randn(M,K)
b2= np.zeros(K)


# Error Function Or Cost Function
#T : Predicted Value Y: Actual Value
def cross_entropy(T, Y):
    return -np.mean(T * np.log(Y))


# Accuracy Calculation
#P : Predicted  Value , Y : Target Attribute
def accuracy(P, Y):
    a= 100.0 * np.sum(np.argmax(P, 1) == np.argmax(Y, 1))
    return a / P.shape[0]

# This function is for W2 weight Calculation
def weight_w2_calculation(Ztrain,Ptrain,Ytrain):
    output=Ztrain.T.dot(Ptrain - Ytrain)
    return  output

# This function is for b2 Bias Calculation
def bias_b2_update(Ptrain,Ytrain):
    return (Ptrain - Ytrain).sum(axis=0)

#This function is for dZ
def dz_derivative_calculation(Ptrain,Ytrain,Ztrain):
    # Derivative of Tanh:  1 - Ztrain*Ztrain
    return (Ptrain - Ytrain).dot(W2.T) * (1 - Ztrain*Ztrain)


# This function is for W1 weight Calculation
def weight_w1_calculation(Xtrain,dZ):
    output= Xtrain.T.dot(dZ)
    return output


# This function is for W2 weight Calculation
def bias_b1_update(dZ):
    return  dZ.sum(axis=0)


#Gradient Desecent
learning_rate = 0.000001
# Training
for epoch in range(15000):
    Ptrain, Ztrain, m2train=feedForward(Xtrain,W1,b1,W2,b2,True)
    error_train= cross_entropy(Ytrain,Ptrain)

    # update W2 and W1 with L2 Regularization
    # In L2 regularization smoothing parameter uses 0.1
    W2 -= learning_rate*(weight_w2_calculation(Ztrain,Ptrain,Ytrain)+0.1*W2)
    b2 -= learning_rate* bias_b2_update(Ptrain,Ytrain)
    dZ =  dz_derivative_calculation(Ptrain,Ytrain,Ztrain)*m2train
    W1 -= learning_rate*(weight_w1_calculation(Xtrain,dZ)+0.1*W1)
    b1 -= learning_rate*bias_b1_update(dZ)
    if (epoch % 1000 == 0):
        print('Iteration:', epoch)
        print('classification rate for training data : ', accuracy(Ptrain, Ytrain))


#Testing
for i in range(15000):
    Ptest, Ztest, m2test = feedForward(Xtest, W1, b1, W2, b2, False)
    error_test = cross_entropy(Ytest, Ptest)


print('Number at node at Hidden Layer :',M)
print('Iterations ',15000)
print('Learning Rate',learning_rate)
print('Final accuracy for training data :',accuracy(Ptrain,Ytrain))
print('Final accuracy for testing data :',accuracy(Ptest,Ytest))

