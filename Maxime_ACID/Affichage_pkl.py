import pickle

root = './results_150/handler.pkl'
# Charger le fichier .pkl
with open(root, 'rb') as fichier:
    contenu = pickle.load(fichier)

# Afficher le contenu
print(contenu)
