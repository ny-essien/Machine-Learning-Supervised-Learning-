# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:14:12 2023

@author: nsika
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = ("c://ml_dataset/Position_salaries.csv")

data = pd.read_csv(path)
#creating the matrix of features X and the dependent varible y
X = data.iloc[:, 1:2].values
y = data.iloc[:, -1].values

#reshape y to a 2-dimensional shape
y = y.reshape(len(y), 1)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)

#training the Support vector regression model on the entire data
from sklearn.svm import SVR
regressor = SVR(kernel= 'rbf')
regressor.fit(X,y)

#making a single prediction
prediction = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6]])).reshape(-1,1))

#visualizing the SVR model on the entire data
plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = "blue")
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#visualizing the SVR model in a higher resolution
X_grid = np.arange(min(sc_x.inverse_transform(X)), max(sc_x.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid)).reshape(-1,1)), color = "blue")
plt.title('Truth or Bluff (SVR) Higher Resolution')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Evaluating model performance
from sklearn.metrics import r2_score
adj_r2 = r2_score(y, regressor.predict(X))