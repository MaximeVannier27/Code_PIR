from __future__ import print_function
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import callbacks
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.preprocessing import Normalizer
from codecarbon import track_emissions
import json
import os

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

X_train = np.array(trainX)
X_test = np.array(testT)

batch_size = 1024

# Définir un dictionnaire pour stocker les métriques de chaque batch
metrics_history = {"batch_size": batch_size, "metrics": {"accuracy": [], "loss": []}}

# Définir un callback personnalisé pour enregistrer les métriques à chaque batch
class BatchMetricsCallback(callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        metrics_history["metrics"]["accuracy"].append(logs.get("accuracy"))
        metrics_history["metrics"]["loss"].append(logs.get("loss"))

# Définir la fonction d'entraînement avec l'utilisation du callback personnalisé
@track_emissions
def training(X_train, y_train, batch_size, model):
    checkpointer = ModelCheckpoint(filepath="./Tanguy_DNN/resultats/checkpoints/checkpoint-{epoch:02d}.keras", verbose=1, save_best_only=True, monitor='loss')
    csv_logger = CSVLogger('training_set_dnnanalysis.csv', separator=',', append=False)
    model.fit(X_train, y_train, validation_data=None, batch_size=batch_size, epochs=2, verbose=1, callbacks=[checkpointer, csv_logger, BatchMetricsCallback()])
    return model

# Appeler la fonction d'entraînement
model = Sequential()
model.add(Dense(1024, input_dim=41, activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(768, activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(512, activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(256, activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(128, activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model = training(X_train, y_train, batch_size, model)

# Enregistrer le dictionnaire de métriques dans un fichier JSON
metrics_file_path = "./Tanguy_DNN/resultats/metrics/metrics_history.json"
with open(metrics_file_path, "w") as json_file:
    json.dump(metrics_history, json_file)
