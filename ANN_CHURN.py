# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 23:05:58 2019

@author: SURYA
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()
dataset.info()

X = dataset.iloc[:,3:13]
Y = dataset.iloc[:,13]

X.columns[X.dtypes=='object']
X.isnull().sum()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X.Geography = encoder.fit_transform(X.Geography)
X.Gender = encoder.fit_transform(X.Gender)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import ReLU, LeakyReLU, ELU
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(output_dim = 6, init = 'he_uniform', activation= 'relu', input_dim = 10))

classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform', activation = 'relu'))

classifier.add(Dense(units=1, init = 'glorot_uniform', activation = 'sigmoid'))


classifier.compile(optimizer='Adam', loss = 'binary_crossentropy', metrics=['accuracy'])


model = classifier.fit(X_train, Y_train, validation_split=0.33, nb_epoch = 100)

from sklearn.metrics import accuracy_score
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)
score = score=accuracy_score(Y_pred,Y_test)
score
