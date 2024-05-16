# General imports

import numpy as np 
import pandas as pd
from argparse import ArgumentParser
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import scipy.stats as st
import statistics as stat
import os
import sys
import random
import pickle
#import tensor as tf

# Keras imports
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Sequential
from keras.callbacks import TensorBoard, EarlyStopping

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix


def AutoEncoder(_autoencoder, input_dim):


    """
        Initializes Deep Autoencoder structure, initialize tensorboard, and initialize Standard Scaler
        Arguments:
            input_dim
    """
    
    # Initialize self._autoencoder
    _autoencoder.add(Dense(int(0.75 * input_dim), activation="relu", input_shape=(input_dim,)))
    _autoencoder.add(Dense(int(0.5 * input_dim), activation="relu"))
    _autoencoder.add(Dense(int(0.33 * input_dim), activation="relu"))
    _autoencoder.add(Dense(int(0.25 * input_dim), activation="relu"))
    _autoencoder.add(Dense(int(0.33 * input_dim), activation="relu"))
    _autoencoder.add(Dense(int(0.5 * input_dim), activation="relu"))
    _autoencoder.add(Dense(int(0.75 * input_dim), activation="relu"))
    _autoencoder.add(Dense(input_dim))

    # Initialize tensorboard
    _tensorboard = TensorBoard(
        log_dir="logs",
        histogram_freq=0,
        write_graph=True,
        write_images=True)

    # Set EarlyStopping Parameters
    _early_stopping = EarlyStopping(
        monitor='loss',
        min_delta=0.001,
        patience=4,
        verbose=0,
        mode='auto',
        restore_best_weights=True)

    # StandardScaler object
    _scaler = StandardScaler()

def mean_confidence_interval(data, confidence=0.98):
    a = 1.0 * np.array(data)
    n = len(a)
    se =  st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return  h



if __name__ == "__main__":


    epochs = 100
    batch_size = 1024

    
    #dataset = pd.read_csv("kddcup_corrige.csv", header=None,names=['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'])
    #print("loading completed")

    traindata1 = pd.read_csv(r'./dataset/traindata1.csv', header=None)
    traindata2 = pd.read_csv(r'./dataset/traindata2.csv', header=None)
    traindata3 = pd.read_csv(r'./dataset/traindata3.csv', header=None)

    
    begnin1 = traindata1[traindata1[41] == 0]
    X1_train = begnin1.iloc[:,0:41]
    Y1_train = traindata1.iloc[:,41]

    begnin2 = traindata2[traindata2[41] == 0]
    X2_train = begnin2.iloc[:,0:41]
    Y2_train = traindata2.iloc[:,41]

    begnin3 = traindata3[traindata3[41] == 0]
    X3_train = begnin3.iloc[:,0:41]
    Y3_train = traindata3.iloc[:,41]

    testdata1 = pd.read_csv(r'./dataset/testdata1.csv', header=None)
    testdata2 = pd.read_csv(r'./dataset/testdata2.csv', header=None)
    testdata3 = pd.read_csv(r'./dataset/testdata3.csv', header=None)
    
    X1_test = testdata1.iloc[:,0:41]
    Y1_test = testdata1.iloc[:,41]
    X2_test = testdata2.iloc[:,0:41]
    Y2_test = testdata2.iloc[:,41]
    X3_test = testdata3.iloc[:,0:41]
    Y3_test = testdata3.iloc[:,41]


    print("Creating model")

    # Initialize auto-encoder
    model1 = Sequential()
    AutoEncoder(model1, 41)
    model2 = Sequential()
    AutoEncoder(model2, 41)
    model3 = Sequential()
    AutoEncoder(model3, 41)
    # StandardScaler object
    _scaler = StandardScaler()

    print("\nTraining")

    tracker = EmissionsTracker()
    tracker.start()
    # Train model
    #with EmissionsTracker(project_name="Autoencoder") as train_tracker:
    # scale begnin train & validation datasets
    
    #validate_scaled = _scaler.fit_transform(validate)
    model1.compile(loss="mean_squared_error", optimizer="sgd", metrics=['accuracy'])
    model2.compile(loss="mean_squared_error", optimizer="sgd", metrics=['accuracy'])
    model3.compile(loss="mean_squared_error", optimizer="sgd", metrics=['accuracy'])

    X_train_scaled = _scaler.fit_transform(X1_train)
    history1 = model1.fit(X_train_scaled, X_train_scaled, validation_data=None, batch_size=batch_size, epochs=epochs)

    X_train_scaled = _scaler.fit_transform(X2_train)
    history2 = model2.fit(X_train_scaled, X_train_scaled, validation_data=None, batch_size=batch_size, epochs=epochs)

    X_train_scaled = _scaler.fit_transform(X3_train)
    history3 = model3.fit(X_train_scaled, X_train_scaled, validation_data=None, batch_size=batch_size, epochs=epochs)
    tracker.stop()
   
    train_loss1 = history1.history['loss']
    train_accuracy1 = history1.history['accuracy']

    train_loss2 = history2.history['loss']
    train_accuracy2 = history2.history['accuracy']

    train_loss3 = history3.history['loss']
    train_accuracy3 = history3.history['accuracy']
    
    mean_train_loss=[]
    mean_train_accuracy=[]
    liste_barres = []
    
    for i in range (len(train_loss1)):
        liste_h = []
        mean_train = (train_loss1[i] + train_loss2[i] + train_loss3[i]) /3
        mean_train_loss.append(mean_train)
        mean_accuracy = (train_accuracy1[i] + train_accuracy2[i] + train_accuracy3[i]) /3
        mean_train_accuracy.append(mean_accuracy)
        liste_h.append(train_loss1[i])
        liste_h.append(train_loss2[i])
        liste_h.append(train_loss3[i])
        h = mean_confidence_interval(liste_h)
        liste_barres.append(h)




    print("\nTesting")
    # Evaluate model
    X_test_scaled = _scaler.fit_transform(X1_test)
    model1.evaluate(X_test_scaled, Y1_test,  batch_size=batch_size)

    X_test_scaled = _scaler.fit_transform(X2_test)
    model2.evaluate(X_test_scaled, Y2_test,  batch_size=batch_size)

    X_test_scaled = _scaler.fit_transform(X3_test)
    model3.evaluate(X_test_scaled, Y3_test,  batch_size=batch_size)

    print("\n")
    print(mean_train_loss)
    print("\n")
    print(mean_train_accuracy)
    print("\n")
    print(liste_barres)
    

    plt.plot(mean_train_loss, label='Training Loss 1', color='blue')
    #plt.plot(train_loss2, label='Training Loss 2', color = 'red')
    #plt.plot(train_loss3, label='Training Loss 3', color = 'green')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(mean_train_accuracy, label='Training Accuracy 1', color='blue')
    #plt.plot(train_accuracy2, label='Training Accuracy 2', color='red')
    #plt.plot(train_accuracy3, label='Training Accuracy 3', color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    with open('val_loss_remy.pkl', 'wb') as file:
        # Utiliser pickle.dump() pour écrire l'objet dans le fichier
        pickle.dump(mean_train_loss, file)


    with open('std_loss_remy.pkl', 'wb') as file:
        # Utiliser pickle.dump() pour écrire l'objet dans le fichier
        pickle.dump(liste_barres, file)
    
    
        
       
    
