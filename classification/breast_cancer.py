# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 19:27:00 2023

@author: nsika
"""

import numpy as np
import pandas as pd

path = "C://ml_dataset/breast_cancer.csv"
data = pd.read_csv(path)

X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2, random_state= 42 )

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)