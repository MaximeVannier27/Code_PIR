# General imports

import numpy as np 
import pandas as pd
from argparse import ArgumentParser
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





if __name__ == "__main__":


    epochs = 100
    batch_size = 1024

    
    dataset = pd.read_csv("kddcup_corrige.csv", header=None,names=['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'])
    print("loading completed")

    traindata = pd.read_csv(r'./dataset/traindata1.csv', header=None)

    X_train = traindata.iloc[:,0:42]
    Y_train = traindata.iloc[:,42]

    testdata = pd.read_csv(r'./dataset/testdata1.csv', header=None)
    
    X_test = testdata.iloc[:,0:42]
    Y_test = testdata.iloc[:,42]


    print("Creating model")

    # Initialize auto-encoder
    model = Sequential()
    AutoEncoder(model, 42)
    # StandardScaler object
    _scaler = StandardScaler()

    print("\nTraining")
    # Train model

    # scale begnin train & validation datasets
    X_train_scaled = _scaler.fit_transform(X_train)
    #validate_scaled = _scaler.fit_transform(validate)

    # Compile model & begin training
    model.compile(loss="mean_squared_error", optimizer="sgd", metrics=['accuracy'])
    model.fit(X_train_scaled, X_train_scaled, validation_data=None, batch_size=batch_size, epochs=epochs)

    X_test_scaled = _scaler.fit_transform(X_test)


    print("\nTesting")

    model.evaluate(X_test_scaled, Y_test,  batch_size=batch_size)

    # Evaluate model
    

        
       
    
