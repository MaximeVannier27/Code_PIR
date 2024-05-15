import pandas as pd
import numpy as np

# Charger le fichier CSV
df = pd.read_csv("./Tanguy_DNN/datasets/datasets_formatage/kddcup_formatage.csv", header=None)

# Sélectionner la dernière colonne
last_col = df.iloc[:, -1]

# Trouver les indices des 0 et des 1
indices_zeros = last_col[last_col == 0].index
indices_ones = last_col[last_col == 1].index

# Calculer combien de valeurs on va échanger (20% de chaque)
n_zeros_to_ones = int(0.2 * len(indices_zeros))
n_ones_to_zeros = int(0.2 * len(indices_ones))

# Sélectionner des indices aléatoires pour l'échange
indices_zeros_to_ones = np.random.choice(indices_zeros, n_zeros_to_ones, replace=False)
indices_ones_to_zeros = np.random.choice(indices_ones, n_ones_to_zeros, replace=False)

# Effectuer l'échange
df.iloc[indices_zeros_to_ones, -1] = 1
df.iloc[indices_ones_to_zeros, -1] = 0

# Sauvegarder le fichier modifié
df.to_csv("./Tanguy_DNN/datasets/datasets_formatage/kddcup_formatage_biaise.csv", index=False, header=None)




