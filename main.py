import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset = pd.read_csv('E:/EE - Semester 7/FYP/Data Set/SolarDataSetNew.csv')
dataset = dataset.sample(frac=1)
print(dataset)
X = dataset.iloc[:, 0:9].values
Y = dataset.iloc[:, 9].values
#print(X)
#print(Y)

#min_max_scaler = preprocessing.MinMaxScaler()
#X_scale = min_max_scaler.fit_transform(X)
#print(X_scale)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

model = Sequential([
    Dense(10, activation='relu', input_shape=(9,)),
    #Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')])

#sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

model.compile(optimizer=optimizers.SGD(lr=0.001, clipnorm=0.5), loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=64, epochs=100, validation_data=(X_val, Y_val))

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

'''
from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.01)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X_train, Y_train, verbose=False)

predictions = my_model.predict(X_test)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, Y_test)))
'''
