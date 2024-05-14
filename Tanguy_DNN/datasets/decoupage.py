import pandas as pd
import numpy as np

df = pd.read_csv(r'./Tanguy_DNN/datasets/NSL_KDD_formate.csv', header=None)

train_test_split = .8
msk = np.random.rand(len(df)) < train_test_split
traindata = df[msk] #train_data
testdata = df[~msk]

# Chemin de sauvegarde des fichiers CSV
train_csv_path = "./Tanguy_DNN/datasets/decoupage_KDDcup99/traindata3.csv"
test_csv_path = "./Tanguy_DNN/datasets/decoupage_KDDcup99/testdata3.csv"

# Sauvegarder les données dans des fichiers CSV
traindata.to_csv(train_csv_path, index=False, header=None)
testdata.to_csv(test_csv_path, index=False, header=None)

print("Les données ont été sauvegardées avec succès.")