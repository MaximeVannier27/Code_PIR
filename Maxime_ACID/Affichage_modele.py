import pickle

root = './results/'
# Charger le fichier .pkl
with open(root + 'trained_model.pkl', 'rb') as fichier:
    contenu = pickle.load(fichier)

# Afficher le contenu
print(contenu)
