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
from keras.utils import to_categorical

from codecarbon import track_emissions

traindata = pd.read_csv(r"./Tanguy_DNN/datasets/decoupage_KDDcup99/traindata3.csv", header=None)
testdata = pd.read_csv(r"./Tanguy_DNN/datasets/decoupage_KDDcup99/testdata3.csv", header=None)

X = traindata.iloc[:,0:41]
Y = traindata.iloc[:,41]
T = testdata.iloc[:,0:41]
C = testdata.iloc[:,41]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)

#y_train= to_categorical(y_train1)
#y_test= to_categorical(y_test1)

X_train = np.array(trainX)
X_test = np.array(testT)

batch_size = 64

# 1. define the network
model = Sequential()
model.add(Dense(1024,input_dim=41,activation='relu'))  
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

  checkpointer = callbacks.ModelCheckpoint(filepath="./Tanguy_DNN/resultats/checkpoints/checkpoint-#{epoch:02d}.keras", verbose=2, save_best_only=True, monitor='loss')
  csv_logger = CSVLogger('./Tanguy_DNN/training_set_dnnanalysis.csv',separator=',', append=False)

  model.fit(X_train, y_train, validation_data=None, batch_size=batch_size, epochs=100, verbose=1,callbacks=[checkpointer,csv_logger])
  return model

model = training(X_train, y_train, batch_size, model)
save_model(model, "./Tanguy_DNN/resultats/final/dnn5layer_model.keras")

@track_emissions
def testing(X_test, y_test, model):
    # Évaluation du modèle sur les données de test
    scores = model.evaluate(X_test, y_test, verbose=2)
    
    # Affichage de la précision et de la perte
    print("Test Accuracy: %.2f%%" % (scores[1]*100))
    print("Test Loss: %.2f" % scores[0])

# Appel de la fonction testing
testing(X_test, y_test, model)




