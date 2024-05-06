from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
#from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.models import save_model



df = pd.read_csv(r'./dataset/kddcup_corrige.csv', header=None)


train_test_split = .7
msk = np.random.rand(len(df)) < train_test_split
traindata = df[msk] #train_data
testdata = df[~msk]

X = traindata.iloc[:,0:42]
Y = traindata.iloc[:,41]
T = testdata.iloc[:,0:42]
C = testdata.iloc[:,41]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)

X_train = np.array(trainX)
X_test = np.array(testT)

batch_size = 1024

# 1. define the network
model = Sequential()
model.add(Dense(1024,input_dim=42,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(768,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(512,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(256,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(128,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/dnn5layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
#csv_logger = CSVLogger('kddresults/dnn5layer/training_set_dnnanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, validation_data=(X_test, y_test),batch_size=batch_size, epochs=1)#callbacks=[checkpointer,csv_logger])

save_model(model, "./dnn5layer_model.keras")






