import pickle

root = './results/'
# Charger le fichier .pkl
with open(root+"jeu_de_donnees/dico_enviro.pkl", 'rb') as fichier:
    contenu = pickle.load(fichier)

# Afficher le contenu
print(contenu)
