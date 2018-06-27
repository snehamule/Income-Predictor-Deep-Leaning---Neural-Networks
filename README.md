# Income Predictor Application (Deep Learning - Neural Network)

This project is implemented  3 layers backpropogation algorithm from scratch using python. 
This project predict income for a adult (above 50 K or below 50K )using different parameters like age,place, education, number of years of experience etc. It uses tanh activation function in hidden layer, and softmax activation function in output layer.
To verify accuracy, build a same project by using scikit-learn library with same dataset. 
Accuracy for backpropogation algorithm : 93.25%
Accuracy for Scikit Learn algorithm : 95.78%


## Technology used: <br />
Java , Java swing <br />

## Setup required:<br />
Java version: 1.8 or greater<br />
Database : Oracle database 10.g or above<br />
External jars : json-simple-1.1.jar


## Download json-simple-1.1.jar <br />
In this project Json-simple is used to parse json file. 
Download json-simple-1.1 [Downalod Json Simple-1.1](http://www.java2s.com/Code/Jar/j/Downloadjsonsimple11jar.htm)


## Dataset :<br />
This application used dataset which Yelp.com has announced the “Yelp Dataset Challenge”. It contains 42K businesses, 252K users, and 1.1M reviews. Data folder conatins business.json, rerview.json, user.json,business_category.json files.
User can also download those files from  [Yelp database challenge](https://www.yelp.com/dataset).  

## Run program : <br />
1. Download code from git  using  git clone .
2. Place downloaded dataset files in the same folder (Optional Step)
3. For  compile java program 
```
	javac MainPage.java
```	
4. To run java program 
```
	java MainPage
```

