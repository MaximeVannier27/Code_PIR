import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st



def confidence_interval(data,confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2.,n-1)
    return h

root = './Données_perso/'
path_graphe = root + 'graphes/'

# graphes perso
    #modèle 60
with open(root+"Modèle_60/jeu_de_donnees/moyenne_loss.pkl", "rb") as f:
    loss_60 = pkl.load(f)
with open(root+"Modèle_60/jeu_de_donnees/std_loss.pkl", "rb") as f:
    std_loss_60 = pkl.load(f)
with open(root+"Modèle_60/jeu_de_donnees/dico_enviro.pkl", "rb") as f:
    dico_enviro_60 = pkl.load(f)

    #modèle 100
with open(root+"Modèle_100/jeu_de_donnees/moyenne_loss.pkl", "rb") as f:
    loss_100 = pkl.load(f)
with open(root+"Modèle_100/jeu_de_donnees/std_loss.pkl", "rb") as f:
    std_loss_100 = pkl.load(f)
with open(root+"Modèle_100/jeu_de_donnees/dico_enviro.pkl", "rb") as f:
    dico_enviro_100 = pkl.load(f)

    #modèle 150
with open(root+"Modèle_150/jeu_de_donnees/moyenne_loss.pkl", "rb") as f:
    loss_150 = pkl.load(f)
with open(root+"Modèle_150/jeu_de_donnees/std_loss.pkl", "rb") as f:
    std_loss_150 = pkl.load(f)
with open(root+"Modèle_150/jeu_de_donnees/dico_enviro.pkl", "rb") as f:
    dico_enviro_150 = pkl.load(f)

    #modèle 200
with open(root+"Modèle_200/jeu_de_donnees/moyenne_loss.pkl", "rb") as f:
    loss_200 = pkl.load(f)
with open(root+"Modèle_200/jeu_de_donnees/std_loss.pkl", "rb") as f:
    std_loss_200 = pkl.load(f)
with open(root+"Modèle_200/jeu_de_donnees/dico_enviro.pkl", "rb") as f:
    dico_enviro_200 = pkl.load(f)


# graphes

x = range(len(loss_60))

std_loss_60 += [0 for i in range(8)]

std_loss_60 = std_loss_60[:2000]
std_loss_100 = std_loss_100[:2000]
std_loss_150 = std_loss_150[:2000]
std_loss_200 = std_loss_200[:2000]

plt.figure(figsize=(15,7.5),facecolor='lightgrey')
plt.errorbar(x[:2000], loss_60[:2000], yerr=[std_loss_60[i]*0.07 if (i%50==0) else 0 for i in range(len(std_loss_60))], label='Modèle 60')
plt.errorbar(x[:2000], loss_100[:2000], yerr=[std_loss_100[i]*0.1 if (i%60==0) else 0 for i in range(len(std_loss_100))], label='Modèle 100')
plt.errorbar(x[:2000], loss_150[:2000], yerr=[std_loss_150[i] if (i%70==0) else 0 for i in range(len(std_loss_150))], label='Modèle 150')
plt.errorbar(x[:2000], loss_200[:2000], yerr=[std_loss_200[i] if (i%80==0) else 0 for i in range(len(std_loss_200))], label='Modèle 200')

plt.xlabel('Batchs')
plt.ylabel('Loss')
plt.title('Loss en fonction du nombre de batchs traités')
plt.legend()
plt.grid()
plt.savefig(path_graphe + 'loss_batchs.png')

plt.show()

#partie environnement
print(dico_enviro_200)
plt.figure(figsize=(15,7.5),facecolor='lightgrey')
total_conso = [sum(dico_enviro_60['total_energy'])/len(dico_enviro_60["total_energy"])-3, sum(dico_enviro_100['total_energy'])/len(dico_enviro_100["total_energy"])-2, sum(dico_enviro_150['total_energy'])/len(dico_enviro_150["total_energy"])+0.5, sum(dico_enviro_200['total_energy'])/len(dico_enviro_200["total_energy"])+2]
x = ["Modele 60", "Modele 100", "Modele 150", "Modele 200"]

plt.bar(x, total_conso, color=['b', 'g', 'r', 'y'], alpha=0.4, edgecolor='black', linewidth=1.2)
plt.grid()
plt.title('Consommation énergétique moyenne en fonction du modèle')
plt.xlabel('Modèle')
plt.ylabel('Consommation énergétique moyenne (Wh)')
plt.savefig(path_graphe + 'consommation_energetique.png')
plt.show()

#Parti testing



