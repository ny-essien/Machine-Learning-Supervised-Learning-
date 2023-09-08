# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:22:11 2023

@author: nsika
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "c://ml_dataset/Position_salaries.csv"
data = pd.read_csv(path)

X = data.iloc[:, 1:2].values
y = data.iloc[:, -1].values

#training the decision tree on the entire dateset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#making a single prediction
prediction = regressor.predict([[6]]) 

#visualizing the decision tree for higher resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

from sklearn.metrics import r2_score
adj_r2 = r2_score(y, regressor.predict(X))
