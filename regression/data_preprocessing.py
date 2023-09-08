# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:07:04 2023

@author: nsika
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "c://ml_dataset/Data.csv"

data = pd.read_csv(path)

#setting the matrix of features X and the dependent variable y
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#handling the missing variable
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy = 'mean')
imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#encoding the data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers= [('encoders', OneHotEncoder(), [0])], remainder= 'passthrough')
X = ct.fit_transform(X)

#encoding the dependent varible
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#splitting the data into training set and test set
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 0)

#scaling the dependent variable
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:, 3:] = sc.transform(X_test[:,3:])

