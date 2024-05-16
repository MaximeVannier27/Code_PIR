import json
import matplotlib.pyplot as plt
import numpy as np

# Charger les données depuis le fichier JSON
metrics_file_path = "./Tanguy_DNN/resultats/metrics/metrics_dnn5layer_model_base.json"
metrics_file_path2 = "./Tanguy_DNN/resultats/metrics/metrics_dnn1layer_1024.json"
with open(metrics_file_path, "r") as json_file:
    metrics_history = json.load(json_file)

# Extraire les métriques d'accuracy et de loss pour chaque batch
accuracies = metrics_history["metrics"]["accuracy"]
losses = metrics_history["metrics"]["loss"]
batch_size = metrics_history["batch_size"]

# Calculer le nombre de batches par époque
batches_per_epoch = 61216

# Créer une liste d'index de batch et d'époque
batches = list(range(1, len(losses) + 1))
epochs = [(batch // batches_per_epoch) + 1 for batch in batches]

# Créer une liste de couleurs pour chaque époque
colors = plt.cm.viridis(np.linspace(0, 1, max(epochs)))

# Créer les graphiques pour l'accuracy et la perte (loss) en fonction des batches
plt.figure(figsize=(10, 5))

# Graphique pour l'accuracy
for epoch, color in zip(set(epochs), colors):
    epoch_batches = [batch for batch, ep in zip(batches, epochs) if ep == epoch]
    epoch_accuracies = [acc for acc, ep in zip(accuracies, epochs) if ep == epoch]
    plt.plot(epoch_batches, epoch_accuracies, linestyle='-', label=f'Epoch {epoch}', color=color, linewidth=1)

plt.title('Accuracy')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Graphique pour la perte (loss)
plt.figure(figsize=(10, 5))

for epoch, color in zip(set(epochs), colors):
    epoch_batches = [batch for batch, ep in zip(batches, epochs) if ep == epoch]
    epoch_losses = [loss for loss, ep in zip(losses, epochs) if ep == epoch]
    plt.plot(epoch_batches, epoch_losses, linestyle='-', label=f'Epoch {epoch}', color=color, linewidth=1)

plt.title('Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
