# Income Predictor Application (Deep Learning - Neural Network)

This project is implemented  3 layers backpropogation algorithm from scratch using python. 
This project predict income for a adult (above 50 K or below 50K )using different parameters like age,place, education, number of years of experience etc. It uses tanh activation function in hidden layer, and softmax activation function in output layer.
To verify accuracy, build a same project by using scikit-learn library with same dataset. 
Accuracy for backpropogation algorithm : 93.25%
Accuracy for Scikit Learn algorithm : 95.78%

## Technology / libraries used: <br />
Python, Scikit learn, Panda, Numpy

## Setup required:<br />
python version: 3 or greater<br />
Libraries : ScikitLearn, Panda,Numpy


## Install python <br />
If python is not installed then need to install python:<br />
<br />
**For  osx operating system (mac)**<br />
	python get-pip.py 

**For windows operating system**<br />
	refer steps from [windows python installation steps](https://docs.python.org/3/using/windows.html).
	

## Check python version:
python -version


## Install Libraries<br /> 

**For  osx operating system (mac)**<br />
* Install Numpy : pip install numpy<br />
* Install  Panada : pip install pandas<br />
* Install  Scikitlearn: pip install scipy, scikit-learn<br />

**For windows operating system**<br />
* Install numpy : pip install numpy<br />
* Install pandas : python -m pip install pandas<br />
* Install  Scikitlearn: pip install -U scikit-learn<br />


## Dataset Download :<br />
This recommendation system use  Book-Crossing Dataset.
Download Adult Dataset from UCI dataset [Adult UCI dataset](https://archive.ics.uci.edu/ml/datasets/adult).  

## Run program : <br />
1. Download code from git  using  git clone .
2. Place downloaded dataset files in the same folder
3. For Process the Data run command 
```
	python process_data.py
```	
4. To run backpropogation algorithm, run command 
```
	python backPropogation_algorithm py
```
5. To run same program using scikit learn :
```
     python scitkit_learn_backpropogation_algo.py

```
