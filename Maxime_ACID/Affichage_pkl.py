import pickle

root = './results_ekip/jeu_de_donnees/conso_CPU.pkl'
# Charger le fichier .pkl
with open(root, 'rb') as fichier:
    contenu = pickle.load(fichier)

# Afficher le contenu
print(contenu)
