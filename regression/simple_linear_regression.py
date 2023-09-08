# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:13:23 2023

@author: nsika
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "c://ml_dataset/Salary_Data.csv"

data = pd.read_csv(path)

#creating matrix of features X and dependent variable y
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#spliting data to training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 0)

#training the linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting on the test set
y_pred = regressor.predict(X_test)

#predicting a single test variable
prediction = regressor.predict([[10.3]])

#visualizing the predictions on the training set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Years Of Experience vs Salary\n(Training Set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()

#visulaizing the predictions on the test set
plt.scatter(X_test,y_test, color = 'black')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Years Of Experience vs Salary\n(Test Set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()