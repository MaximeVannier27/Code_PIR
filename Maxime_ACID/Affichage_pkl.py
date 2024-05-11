import pickle

root = './results/'
# Charger le fichier .pkl
with open(root + 'jeu_de_donnees/'+'moyenne_loss.pkl', 'rb') as fichier:
    contenu = pickle.load(fichier)

# Afficher le contenu
print(contenu)
