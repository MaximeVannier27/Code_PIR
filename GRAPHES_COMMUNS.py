import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

#Path Maxime
maxime_root = './Maxime_ACID/results_ekip/'
maxime_donnees = maxime_root + 'jeu_de_donnees/'
####################################################


#REMPLISSEZ VOS PATH SI C'EST PLUS SIMPLE POUR VOUS





#PARTIE LOSS

#Loss Maxime
with open(maxime_donnees+"moyenne_loss.pkl", "rb") as f:
    maxime_loss = pkl.load(f)
with open(maxime_donnees+"std_loss.pkl", "rb") as f:
    maxime_std_loss = pkl.load(f)


#REMPLISSEZ EN INITIALISANT VOS DONNEES DE LOSS ET DE STD LOSS (BARRES D'ERREURS) SI VOUS EN AVEZ


#GRAPHES LOSS

taille_x=max(len(maxime_loss), METTRE LA TAILLE DE VOTRE LISTE ICI     )
x = [i for i in range(taille_x)]

maxime_loss = [maxime_loss[i] if (i<len(maxime_loss)) else None for i in range(taille_x)]
maxime_std_loss = [maxime_std_loss[i] if (i<len(maxime_std_loss)) else None for i in range(taille_x)]

#REMPLISSEZ COMME MOI POUR ADAPTER AUTO LA TAILLE DE VOS DONNEES




plt.figure(figsize=(15, 7.5),facecolor='lightgrey')
plt.errorbar(x, maxime_loss, yerr=[maxime_std_loss[i] if (i%50==0) else 0 for i in range(len(maxime_std_loss))], label='ACID', color='blue')
#METTEZ LA MEME LIGNE QUE MOI MAIS AVEC VOS TRUCS



plt.xlabel('Itérations')
plt.ylabel('Loss')
plt.title('Loss en fonction des itérations pour chaque implémentation étudiée (à partir du même dataset)')
plt.legend()
plt.grid()
plt.show()
plt.savefig('./loss_iterations_commun.png')

##############################################################################################################
#PARTIE ENVIRO

with open(maxime_donnees+"conso_CPU.pkl", "rb") as f:
    maxime_enviro = pkl.load(f)
maxime_enviro = float(maxime_enviro)
#RECUPERER VOTRE VALEUR DE CONSO MOYENNE OU ECRIVEZ LA DIRECTEMENT DANS UNE VARIABLE


#GRAPHES ENVIRO
x = ["ACID", METTRE VOTRE ALGO ICI MAIS DANS LE MÊME ORDRE QUE DANS VALEURS_HISTO]
valeurs_histo = [maxime_enviro, METTRE VOTRE VALEUR (EN FLOAT) ICI DANS LE MÊME ORDRE QUE DANS X]




plt.figure(figsize=(15, 7.5),facecolor='lightgrey')
plt.bar(x, valeurs_histo, color=['b', 'g', 'r', 'y'], alpha=0.4, edgecolor='black', linewidth=1.2)
plt.grid()
plt.title('Consommation électrique moyenne lors du training de chaque implémentation étudiée (à partir du même dataset)')
plt.xlabel('Implémentations')
plt.ylabel('Consommation électrique moyenne (Wh)')
plt.savefig('./conso_electrique_commun.png')
plt.show()

##############################################################################################################
#PARTIE PRECISION

maxime_precision = ca arrive l'ekip

#METTEZ VOTRE VALEUR DE PRECISION DANS UNE VARIABLE
x = ["ACID", METTRE VOTRE ALGO ICI MAIS DANS LE MÊME ORDRE QUE DANS VALEURS_HISTO]
valeurs_histo = [maxime_precision, METTRE VOTRE VALEUR (EN FLOAT) ICI DANS LE MÊME ORDRE QUE DANS X]


plt.figure(figsize=(15, 7.5),facecolor='lightgrey')
plt.bar(x, valeurs_histo, color=['b', 'g', 'r', 'y'], alpha=0.4, edgecolor='black', linewidth=1.2)
plt.grid()
plt.title('Précision moyenne lors du testing de chaque implémentation étudiée (sur le même dataset)')
plt.xlabel('Implémentations')
plt.ylabel('Précision moyenne')
plt.savefig('./precision_commun.png')
plt.show()