import pandas as pd

df = pd.read_csv("./datasets/KDD/kddcup.csv", index_col=False)

liste_service = []
liste_flag = []
liste_protocol = []


for i in range(len(df)):
    #FORMATAGE SERVICE
    if df.iloc[i, 2] not in liste_service:
        liste_service.append(df.iloc[i, 2])
    df.at[i, 'service'] = liste_service.index(df.iloc[i, 2])

    #FORMATAGE FLAG
    if df.iloc[i, 3] not in liste_flag:
        liste_flag.append(df.iloc[i, 3])
    df.at[i, 'flag'] = liste_flag.index(df.iloc[i, 3])

    #FORMATAGE LABEL (normal = 0, attaque = 1)
    if df.iloc[i, 41] == "normal.":
        df.at[i, 'Label.'] = 0
    else:
        df.at[i, 'Label.'] = 1

    #FORMATAGE PROTOCOL
    if df.iloc[i, 1] not in liste_protocol:
        liste_protocol.append(df.iloc[i, 1])
    df.at[i, 'protocol_type'] = liste_protocol.index(df.iloc[i, 1])


print("taille liste_service : ",len(liste_service))
for i in range(len(liste_service)):
    print(i,":",liste_service[i])

print("\ntaille liste_flag : ",len(liste_flag))
for i in range(len(liste_flag)):
    print(i,":",liste_flag[i])


print("\ntaille liste_protocol : ",len(liste_protocol))
for i in range(len(liste_protocol)):
    print(i,":",liste_protocol[i])


# Écriture du DataFrame modifié dans un nouveau fichier CSV
df.to_csv("./datasets/KDD/kddcup_corrige.csv", index=False)
