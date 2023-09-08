# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:55:48 2023

@author: nsika
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "c://ml_dataset/Position_salaries.csv"

data = pd.read_csv(path)
#Creating the matrix of features X and the dependent varible y
X = data.iloc[:,1:2].values
y = data.iloc[:,-1].values

#training the linear regression model on the entire data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

#training the polynomial model on the entire data set
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree= 11)
X_poly = poly_regressor.fit_transform(X,y)
regressor2 = LinearRegression()
regressor2.fit(X_poly,y)

prediction = regressor2.predict(poly_regressor.transform([[6]]))

#visualizing the linear regression model
plt.scatter(X,y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title("Truth Or Bluff (Linear Regression)")
plt.xlabel("Level")
plt.ylabel('Salary')
plt.show()

#visualizing the polynomial regression model on the entire data
plt.scatter(X,y, color = 'red')
plt.plot(X, regressor2.predict(poly_regressor.fit_transform(X)), color = "blue")
plt.title("Truth Or Bluff (Polynomial Regression)")
plt.xlabel("Level")
plt.ylabel('Salary')
plt.show()

#visualizing the polynomial regression at a higher dimension
X_grid = np.arange(X.min(), X.max(), 0.1 )
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, regressor2.predict(poly_regressor.fit_transform(X_grid)), color = "blue")
plt.title("Truth Or Bluff (Polynomial Regression) Higher Dimension")
plt.xlabel("Level")
plt.ylabel('Salary')
plt.show()

coefficient = regressor2.coef_
intercept = regressor.intercept_

#Evaluating the model performance
from sklearn.metrics import r2_score
adj_r2 = r2_score(y, regressor.predict(X))

 

