# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:43:23 2023

@author: nsika
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "c://ml_dataset/Position_salaries.csv"

data = pd.read_csv(path)

X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

#Traing a linear regression model on the entire dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

#Training the polynomial model on the entire dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree= 11)
X_poly = poly_regressor.fit_transform(X,y)
regressor2 = LinearRegression()
regressor2.fit(X_poly, y)

'Making a single prediction from the linear regression model'
lin_prediction = regressor.predict([[7]])

'Making a single prediction on the polynomial model'
poly_prediction = regressor2.predict(poly_regressor.transform([[6]]))


'visualizing the linear regression results on the dataset'
plt.scatter(X,y, color = "red")
plt.plot(X, regressor.predict(X), color = "blue")
plt.title("Truth Or Bluff (Linear Regression)")
plt.xlabel("Position")
plt.ylabel("Salaries")
plt.show()

'Visualizing the polynomial result on the dataset'
X_grid = np.arange(X.min(), X.max(), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color = "red")
plt.plot(X_grid, regressor2.predict(poly_regressor.transform(X_grid)), color = "blue")
plt.title("Truth Or Bluff (Polynomial Regression)")
plt.xlabel("Position")
plt.ylabel("Salaries")
plt.show()

