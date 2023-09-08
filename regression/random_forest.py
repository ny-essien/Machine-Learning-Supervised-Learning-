# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:57:33 2023

@author: nsika
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = "c://ml_dataset/Position_salaries.csv"

data = pd.read_csv(path)

#creating the matrix of features X and the dependent variable y
X = data.iloc[:,1:2].values
y = data.iloc[:,-1].values

#training the random forest regression model on the entire dataset
from sklearn.ensembIle import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 16, random_state=0)
regressor.fit(X,y)

#making a single prediction
prediction = regressor.predict([[6]])

#visualizing the results in higher dimension
X_grid = np.arange(X.min(), X.max(), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = "green")
plt.title("Truth or Bluff (Random Forest Regressor)")
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Evaluating model performance
from sklearn.metrics import r2_score
adj_r = r2_score(y,regressor.predict(X))

