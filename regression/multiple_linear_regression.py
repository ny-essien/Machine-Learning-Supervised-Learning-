# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:26:08 2023

@author: nsika
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "c://ml_dataset/50_Startups.csv"

data = pd.read_csv(path)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#encoding categorical variable in the matrix of features X
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers= [('encoders', OneHotEncoder(), [3])], remainder= 'passthrough')
X = ct.fit_transform(X)

#spliting data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state= 0)

#Training the regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#making predictions on the test set
y_pred = regressor.predict(X_test)

#making a single prediction
prediction = regressor.predict([[0,1,0,66051.52,182645.56,118148.2]])

#finding the values of the intercept and coefficients
intercept = regressor.intercept_
coef = regressor.coef_

#final equation on this dataseet
#y = 42467.53 + 86.6384 * D1 + -872.646 * D2 + 786.007 * D3 + 0.773467 * R&D_SPEND +
#0.0328846 * ADMINISTRATION + 0.030661 * MARKETING_SPEND