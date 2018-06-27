import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# Read 'adult.csv'
df = pd.read_csv('adult.csv')
df.head()
df = df[(df.workclass != ' ?') & (df.occupation != ' ?') & (df['native-country'] != ' ?')]

#Drop 'fnlwgt','education',','relationship',','race','capital-loss','capital-gain'
df = df.drop('fnlwgt', axis = 1)
df = df.drop('education', axis=1)
df = df.drop('relationship', axis=1)
df = df.drop('race', axis=1)
df = df.drop('capital-loss', axis=1)
df = df.drop('capital-gain', axis=1)
df = df.reset_index(drop=True)


#One Hot Encoding  'sex', 'workclass', 'marital-status', 'occupation', 'native-country'
df = pd.get_dummies(df, columns = ['sex', 'workclass', 'marital-status', 'occupation', 'native-country'])

input = df.loc[:, df.columns != 'class']
output = df.loc[:, 'class']
#Shuffle Data
input,output = shuffle(input,output)

dataInput = input.as_matrix()

# normalize columns
dataInput[:, 0] = (dataInput[:, 0] - dataInput[:, 0].mean()) / dataInput[:, 0].std()
dataInput[:, 1] = (dataInput[:, 1] - dataInput[:, 1].mean()) / dataInput[:, 1].std()
dataInput[:, 2] = (dataInput[:, 2] - dataInput[:, 2].mean()) / dataInput[:, 2].std()



output = pd.get_dummies(output, columns = ['class'])
output.head()
dataOutput= output.as_matrix()


dataInput = input.as_matrix()

#trainingDataCount is  70 percentage of input data
trainingDataCount= int(0.70*(dataInput.shape[0]))
print(trainingDataCount)

#This function return trainingInput,traiongOutput
def getTrainingData():
    trainingInput= dataInput[:trainingDataCount]
    trainingOutPut=dataOutput[:trainingDataCount]
    return trainingInput, trainingOutPut

#This function return testingInput,testingOutput
def getTestingData():
    testingInput= dataInput[trainingDataCount:]
    testingOutPut=dataOutput[trainingDataCount:]
    return testingInput,testingOutPut
