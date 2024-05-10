from __future__ import print_function
import pandas as pd
import numpy as np
#np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.models import save_model

from sklearn.preprocessing import Normalizer

from codecarbon import track_emissions

traindata = pd.read_csv(r'./Tanguy_DNN/dataset/traindata1.csv', header=None)
testdata = pd.read_csv(r'./Tanguy_DNN/dataset/testdata1.csv', header=None)

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
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

@track_emissions
def training(X_train, y_train, batch_size, model): 

  # try using different optimizers and different optimizer configs

  #checkpointer = callbacks.ModelCheckpoint(filepath="./Tanguy_DNN/resultats/checkpoints/checkpoint-#{epoch:02d}.keras", verbose=1, save_best_only=True, monitor='loss')
  #csv_logger = CSVLogger('training_set_dnnanalysis.csv',separator=',', append=False)

  model.fit(X_train, y_train, validation_data=None, batch_size=batch_size, epochs=1)
  #callbacks=[checkpointer,csv_logger])
  return model

model = training(X_train, y_train, batch_size, model)
save_model(model, "./Tanguy_DNN/resultats/final/dnn5layer_model.keras")






