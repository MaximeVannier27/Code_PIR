import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import math

#Path Maxime
maxime_root = './Maxime_ACID/results_ekip/'
maxime_donnees = maxime_root + 'jeu_de_donnees/'
louise_donnees = './Louise_DaGMM/'
####################################################


#REMPLISSEZ VOS PATH SI C'EST PLUS SIMPLE POUR VOUS





#PARTIE LOSS

#Loss Maxime
with open(maxime_donnees+"moyenne_loss.pkl", "rb") as f:
    maxime_loss = pkl.load(f)
with open(maxime_donnees+"std_loss.pkl", "rb") as f:
    maxime_std_loss = pkl.load(f)


#Loss Louise
with open(louise_donnees+"moy_loss.pkl", "rb") as f:
    louise_loss = pkl.load(f)
with open(louise_donnees+"erreur_loss.pkl", "rb") as f:
    louise_std_loss = pkl.load(f)

#Loss Tanguy
with open('./val_loss_DNN.pkl', "rb") as f:
    tanguy_loss = pkl.load(f)
with open('./val_ic_DNN.pkl', "rb") as f:
    tanguy_std_loss = pkl.load(f)

#Loss Rémy
with open('./val_loss_remy.pkl', "rb") as f:
    remy_loss = pkl.load(f)
with open('./std_loss_remy.pkl', "rb") as f:
    remy_std_loss = pkl.load(f)



#GRAPHES LOSS

taille_x=max(len(maxime_loss), len(louise_loss))
x = [i for i in range(taille_x)]

maxime_loss = [maxime_loss[i] if (i<len(maxime_loss)) else 0 for i in range(taille_x)]
maxime_std_loss = [maxime_std_loss[i] if (i<len(maxime_std_loss)) else 0 for i in range(taille_x)]
louise_loss = [louise_loss[i] if (i<len(louise_loss)) else 0 for i in range(taille_x)]
louise_std_loss = [louise_std_loss[i] if (i<len(louise_std_loss)) else 0 for i in range(taille_x)]
tanguy_loss = [tanguy_loss[i] if (i<len(tanguy_loss)) else 0 for i in range(taille_x)]
tanguy_std_loss = [tanguy_std_loss[i] if (i<len(tanguy_std_loss)) else 0 for i in range(taille_x)]
remy_loss = [remy_loss[i] if (i<len(remy_loss)) else 0 for i in range(taille_x)]
remy_std_loss = [remy_std_loss[i] if (i<len(remy_std_loss)) else 0 for i in range(taille_x)]
#REMPLISSEZ COMME MOI POUR ADAPTER AUTO LA TAILLE DE VOS DONNEES

print("Tanguy",tanguy_std_loss[:1000])
print("Remy",remy_std_loss[:1000])




plt.figure(figsize=(15, 7.5),facecolor='lightgrey')
# plt.errorbar(x[:1000], [(maxime_loss[i]) for i in range(len(maxime_loss[:1000]))], yerr=[((maxime_std_loss[i]*0.5)) if (i%70==0) else 0 for i in range(len(maxime_std_loss[:1000]))], label='ACID', color='blue',ecolor='lightblue')
# plt.errorbar(x[:1000], [(louise_loss[i]) for i in range(len(louise_loss[:1000]))], yerr=[((louise_std_loss[i])) if (i%75==0) else 0 for i in range(len(louise_std_loss[:1000]))], label='DAGMM', color='green',ecolor='lightgreen')
# plt.errorbar(x[:1000], [(tanguy_loss[i]) for i in range(len(tanguy_loss[:1000]))], yerr=[((tanguy_std_loss[i])) if (i%80==0) else 0 for i in range(len(tanguy_std_loss[:1000]))], label='DNN', color='orange',ecolor='lightcoral')
# plt.errorbar(x[:1000], [(remy_loss[i]) for i in range(len(remy_loss[:1000]))], yerr=[((remy_std_loss[i])) if (i%85==0) else 0 for i in range(len(remy_std_loss[:1000]))], label='N-BAIOT', color='purple',ecolor='violet')

plt.plot(x[:1000], [(maxime_loss[i]) for i in range(len(maxime_loss[:1000]))], label='ACID', color='blue')
plt.plot(x[:1000], [(louise_loss[i]) for i in range(len(louise_loss[:1000]))], label='DAGMM', color='green')
plt.plot(x[:1000], [(tanguy_loss[i]) for i in range(len(tanguy_loss[:1000]))], label='DNN', color='orange')
plt.plot(x[:1000], [(remy_loss[i]) for i in range(len(remy_loss[:1000]))],label='N-BAIOT', color='purple')




plt.xlabel('Itérations')
plt.ylabel('Loss')
plt.ylim(-0.5, 3)
plt.title('Loss en fonction des itérations pour chaque implémentation étudiée (à partir du même dataset)')
plt.legend()
plt.grid()
plt.savefig('./loss_iterations_commun.png')
plt.show()


##############################################################################################################
#PARTIE ENVIRO

with open(maxime_donnees+"conso_CPU.pkl", "rb") as f:
    maxime_enviro = pkl.load(f)
maxime_enviro = float(maxime_enviro)
louise_enviro = 25.08
tanguy_enviro = 31.9
remy_enviro = 6.60
#RECUPERER VOTRE VALEUR DE CONSO MOYENNE OU ECRIVEZ LA DIRECTEMENT DANS UNE VARIABLE


#GRAPHES ENVIRO
x = ["ACID", "DAGMM", "DNN","N-BAIOT"]
valeurs_histo = [maxime_enviro,louise_enviro,tanguy_enviro,remy_enviro]




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

maxime_precision = 0.9999697217360574
louise_precision = 0.6850
tanguy_precision = 0.9996
remy_precision = 0.6877
#METTEZ VOTRE VALEUR DE PRECISION DANS UNE VARIABLE
x = ["ACID", "DAGMM", "DNN","N-BAIOT"]
valeurs_histo = [maxime_precision,louise_precision,tanguy_precision,remy_precision]


plt.figure(figsize=(15, 7.5),facecolor='lightgrey')
plt.bar(x, valeurs_histo, color=['b', 'g', 'r', 'y'], alpha=0.4, edgecolor='black', linewidth=1.2)
plt.grid()
plt.title('Précision moyenne lors du testing de chaque implémentation étudiée (sur le même dataset)')
plt.xlabel('Implémentations')
plt.ylabel('Précision moyenne')
plt.savefig('./precision_commun.png')
plt.show()