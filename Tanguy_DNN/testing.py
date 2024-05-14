import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import Normalizer
from codecarbon import track_emissions

def test_model(model_path, test_data_path):
    testdata = pd.read_csv(test_data_path, header=None)
    T = testdata.iloc[:,0:41]
    C = testdata.iloc[:,41]

    scaler = Normalizer().fit(T)
    testT = scaler.transform(T)

    X_test = np.array(testT)
    y_test = np.array(C)

    # Load the model
    model = load_model(model_path)

    # Evaluation du modèle sur les données de test
    scores = model.evaluate(X_test, y_test, verbose=1)

    # Affichage de la précision et de la perte
    print("Test Accuracy: %.2f%%" % (scores[1] * 100))
    print("Test Precision: %.2f%%" % (scores[2] * 100))
    print("Test Recall: %.2f%%" % (scores[3] * 100))
    print("Test Loss: %.2f" % scores[0])

if __name__ == "__main__":
    model_path = "./Tanguy_DNN/resultats/final/dnn5layer_model.keras"
    test_data_path = "./Tanguy_DNN/datasets/decoupage_KDDcup99/testdata3.csv"
    test_model(model_path, test_data_path)
