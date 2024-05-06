# General imports

import numpy as np 
import pandas as pd
from argparse import ArgumentParser


# Keras imports
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Sequential
from keras.callbacks import TensorBoard, EarlyStopping

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix


class AutoEncoder():

    def __init__(self, input_dim):
        """
            Initializes Deep Autoencoder structure, initialize tensorboard, and initialize Standard Scaler
            Arguments:
                input_dim
        """

        # Initialize self._autoencoder
        self._autoencoder = Sequential()
        self._autoencoder.add(Dense(int(0.75 * input_dim), activation="relu", input_shape=(input_dim,)))
        self._autoencoder.add(Dense(int(0.5 * input_dim), activation="relu"))
        self._autoencoder.add(Dense(int(0.33 * input_dim), activation="relu"))
        self._autoencoder.add(Dense(int(0.25 * input_dim), activation="relu"))
        self._autoencoder.add(Dense(int(0.33 * input_dim), activation="relu"))
        self._autoencoder.add(Dense(int(0.5 * input_dim), activation="relu"))
        self._autoencoder.add(Dense(int(0.75 * input_dim), activation="relu"))
        self._autoencoder.add(Dense(input_dim))

        # Initialize tensorboard
        self._tensorboard = TensorBoard(
            log_dir="logs",
            histogram_freq=0,
            write_graph=True,
            write_images=True)

        # Set EarlyStopping Parameters
        self._early_stopping = EarlyStopping(
            monitor='loss',
            min_delta=0.001,
            patience=4,
            verbose=0,
            mode='auto',
            restore_best_weights=True)

        # StandardScaler object
        self._scaler = StandardScaler()

    def train(self, train, validate, batch_size, epochs):
        """
            Trains Deep Autoencoder on training set.
            Arguments:
                df: test dataframe
                learning_rate:
                batch_size:
                epochs:
        """

        # scale begnin train & validation datasets
        train_scaled = self._scaler.fit_transform(train)
        validate_scaled = self._scaler.fit_transform(validate)

        # Compile model & begin training
        self._autoencoder.compile(loss="mean_squared_error", optimizer="sgd")
        self._autoencoder.fit(
            train_scaled,
            train_scaled,
            validation_data = (validate_scaled, validate_scaled),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                self._tensorboard,
                self._early_stopping
            ]
        )

    def test(self, train, validate, test):
        """
            Tests performance of Deep Autoencoder on a test set.
            Arguments:
                df: test dataframe
                tr: anomaly threshold
        """

        # scale optimization set
        validate_scaled = self._scaler.fit_transform(validate.iloc[:, :-1].values)

        # make predictions from optimization set
        validate_preds = model.predict(validate_scaled)

        # measure optimisation set MSE
        mse = np.mean(np.power(validate_scaled - validate_preds, 2), axis =1)

        # measure optimziation set anomaly threshold
        tr = mse.mean() + mse.std()

        # merge malicious & begnin test sets
        test_set = pd.concat([test, train], sort=True, ignore_index=True)

        # scale test set
        test_scaled = self._scaler.transform(test_set.iloc[:,:-1].values)

        # make predictions from test set
        test_pred = model.predict(test_scaled)

        # Partition test set
        test_input, test_target = test_set.iloc[:, :-1].values, test_set.iloc[:, -1].values

        # Scale test input
        test_scaled = self._scaler.fit_transform(test_input)

        # Predict test targets
        test_preds = self._autoencoder.predict(test_scaled)

        # measure test set MSE 
        mse = np.mean(np.power(test_scaled - test_preds, 2), axis=1)

        # detect anomalies
        predictions = (mse > tr).astype(int)

        # evaluate performance
        print(f"Anomaly threshold: {round(tr, 2)}")
        print(f"Accuracy: {round(accuracy_score(test_target, predictions), 4)*100}%")
        print(f"Recall: {round(recall_score(test_target, predictions), 4)*100}%")
        print(f"Precision: {round(precision_score(test_target, predictions), 4)*100}%")

def correct_dataset():
    # Load dataset
    df = pd.read_csv("kddcup.csv", header=None,names=['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'])
    #df = pd.concat([x for x in pd.read_csv("fichier.csv", low_memory=False, chunksize=100000)], ignore_index=True)
    #df = df.loc[:, ~df.columns.str.contains('^Unnamed')].reset_index(drop=True)
    liste_service = []
    liste_flag = []
    liste_protocol = []

    print("lessgo")
    for i in range(len(df)):
        #FORMATAGE SERVICE
        if df.iloc[i, 2] not in liste_service:
            liste_service.append(df.iloc[i, 2])
        df.at[i, 'service'] = liste_service.index(df.iloc[i, 2])

        #FORMATAGE FLAG
        if df.iloc[i, 3] not in liste_flag:
            liste_flag.append(df.iloc[i, 3])
        df.at[i, 'flag'] = liste_flag.index(df.iloc[i, 3])

        #FORMATAGE LABEL (normal = 0, attaque = 1)
        if df.iloc[i, 41] == "normal.":
            df.at[i, 'label'] = 0
        else:
            df.at[i, 'label'] = 1

        #FORMATAGE PROTOCOL
        if df.iloc[i, 1] not in liste_protocol:
            liste_protocol.append(df.iloc[i, 1])
        df.at[i, 'protocol_type'] = liste_protocol.index(df.iloc[i, 1])


    print("taille liste_service : ",len(liste_service))
    for i in range(len(liste_service)):
        print(i,":",liste_service[i])

    print("\ntaille liste_flag : ",len(liste_flag))
    for i in range(len(liste_flag)):
        print(i,":",liste_flag[i])


    print("\ntaille liste_protocol : ",len(liste_protocol))
    for i in range(len(liste_protocol)):
        print(i,":",liste_protocol[i])


    # Écriture du DataFrame modifié dans un nouveau fichier CSV
    df.to_csv("kddcup_corrige.csv", index=False)


if __name__ == "__main__":

    # CLI arguments
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', help='path to dataset', type=str)
    parser.add_argument('-e', '--epochs', help='No. of epochs', type=int)
    parser.add_argument('-bs', '--batch_size', help='Batch size', type=int)
    #parser.add_argument('-lr', '--learning_rate', help='Learning rate', type=float)
    args = parser.parse_args()
    try:
        df = pd.read_csv("kddcup_corrige.csv", header=None,names=['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'])
        print("loading completed")
    except:
        print("Need to correct kddcup")
        correct_dataset()
        df = pd.read_csv("kddcup_corrige.csv", header=None,names=['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'])

    # Create K-folds
    kf = KFold(n_splits=3, random_state=42, shuffle=True)
    

     # Partition begnin dataset
    begnin = df[df["label"] == 0]
    begnin_partitions = kf.split(begnin)

    # Partition malicious dataset
    malicious = df[df["label"] == 1]
    malicious_partitions = kf.split(malicious)

    for begnin_data, malicious_data in zip(begnin_partitions, malicious_partitions):

        # begnin training, testing set split
        train_idx, test_idx = begnin_data

        begnin_train, begnin_test = begnin.iloc[train_idx,:], begnin.iloc[test_idx, :]

        # create begnin optimization set
        begnin_train, begnin_optimization = np.array_split(begnin_train, 2)

        # malicious testing set split
        train_idx, test_idx = malicious_data
        malicious_test = malicious.iloc[test_idx, :]

        print("Creating model")
        # Initialize auto-encoder
        model = AutoEncoder(41)
        print("Starting training")
        # Train model
        model.train(begnin_train.iloc[:, :-1], begnin_optimization.iloc[:, :-1], args.batch_size, args.epochs)

        # Evaluate model
        #model.test(begnin_test, begnin_optimization, malicious_test)
    
