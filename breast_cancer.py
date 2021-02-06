#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Scikit-learn breast cancer dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
#Classes 2
#Samples per class 212(M),357(B)
#Samples total 569
#Dimensionality 30
#Features real, positive


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.utils import to_categorical 

"""
import tensorflow as tf 
import numpy as np
from tensorflow.keras.utils import to_categorical 
# felosztja az adathalmazt tesztelő és tanító cosportokra
from sklearn.model_selection import train_test_split
# adathalmaz betöltése .data = jellemzők, .target = kimeneti osztályok
from sklearn.datasets import load_breast_cancer
"""
 

bc = load_breast_cancer()
#jellemzők
X = bc.data
#elvárt kimeneti osztályok
Y = bc.target
Y = to_categorical(Y)

#80% tanítási, 20% tesztelési részre osztás
XTraining, XTest, YTraining, YTest = train_test_split(X, Y, test_size=0.2)
#model létrehozása
model = tf.keras.models.Sequential()
#learning rate = 0.1
opt = tf.keras.optimizers.RMSprop(lr=0.1)

# 3 sűrűn összekötött réteg hozzáadása 
N1 = 10
N2 = 4
model.add(tf.keras.layers.Dense(N1, input_dim=30, activation=tf.nn.tanh))
model.add(tf.keras.layers.Dense(N2, activation=tf.nn.softmax))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(XTraining, YTraining, epochs=2000, validation_split=0.2)
loss, acc = model.evaluate(XTest, YTest, verbose=2)
print(acc)

# model mentese
model.save("breast_cancer.hdf5", True, True)
