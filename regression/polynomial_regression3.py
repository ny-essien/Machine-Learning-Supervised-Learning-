# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:24:54 2023

@author: nsika
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "c://ml_dataset/Position_salaries.csv"

data = pd.read_csv(path)

X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

"Training the linear regression model on the entire dataset"
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

"Training the polynomial regression model on the training set"
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 11) 
X_poly = poly_regressor.fit_transform(X,y)
regressor2 = LinearRegression()
regressor2.fit(X_poly,y)

"Making a single prediction on the linear regression model"
single_linear_pred = regressor.predict([[6]])

"making prediction on the polynomial regression model"
single_poly_pred = regressor2.predict(poly_regressor.transform([[6]]))

"Visualizing the Linear Regression Output"
plt.scatter(X,y, color = "red")
plt.plot(X,regressor.predict(X), color = "blue")
plt.title("Truth Or Bluff (Linear Regression)")
plt.xlabel("Position")
plt.ylabel("Expected Salary Compensation")
plt.show()

"Visualizing the polynomial Features for in higher resolution"
X_grid = np.arange(X.min(), X.max(), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y, color = "red" )
plt.plot(X_grid, regressor2.predict(poly_regressor.transform(X_grid)), color = "blue")
plt.title("Truth Or Bluff")
plt.xlabel("Employee Position")
plt.ylabel("Expected Salary Compensation")
plt.show()

"Calculating r2_score for Linear Regression Model"
from sklearn.metrics import r2_score
y_pred = regressor2.predict(poly_regressor.transform(X))
#y_pred = y_pred.reshape(len(y_pred),1)
y_true = y.copy()
#y_true = y_true.reshape(len(y_true), 1)
adj_r2 = r2_score(y_true, y_pred)