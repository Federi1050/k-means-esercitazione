import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


class KClusterModel:
    def __init__(self, numero_cluster, dataset, scaler):
        self.__scaler = scaler
        self.__modello = self.creazione_modello(numero_cluster, dataset)

    def creazione_modello(self, numero_cluster, dataset):
        modello = KMeans(
            n_clusters=numero_cluster,
            random_state=42,
            n_init="auto",
            max_iter = 1000
        )
        modello.fit(dataset)
        return modello

    @staticmethod
    def inerzie(data, k_max=11):
        """
            Usa il metodo del gomito per determinare il numero ottimale di cluster.

            Parametri:
            - X: dataset (array o dataframe)
            - k_max: numero massimo di cluster da testare
            """

        inerzie = []

        # Proviamo diversi valori di k
        for k in range(1, k_max + 1):
            modello = KMeans(n_clusters=k, random_state=42, n_init="auto")
            modello.fit(data)
            inerzie.append(modello.inertia_)  # SSE

        # Grafico del gomito
        fig = plt.figure(figsize=(8, 5))
        plt.plot(range(1, k_max + 1), inerzie, marker='o')
        plt.title("Metodo del Gomito (Elbow Method)")
        plt.xlabel("Numero di cluster (k)")
        plt.ylabel("Inerzia (SSE)")
        plt.grid(True)

        gomito_path = os.path.join("output_grafici", "gomito.png")
        fig.savefig(gomito_path, bbox_inches="tight")
        plt.close(fig)

    def valutazione_modello(self, colonna_labels_original,dataset):
        # siluette quanto performa
        labels = self.__modello.predict(dataset)
        score_silhuette = silhouette_score(dataset, labels)

        # metrica ari
        # visto che abbiamo le labels originali
        ari = adjusted_rand_score(colonna_labels_original, labels)

        # prendendo ultima colonna dataset vedere quanti corrispondenti
        # visto che abbiamo le labels originali

        return {
            "silhuette_score": score_silhuette,
            "ARI" : ari
        }

    def classificazione(self, input_data):
        # 1. Converti dict -> array 2D
        X = pd.DataFrame([input_data])

        # 2. Scaling
        X_scaled_array = self.__scaler.transform(X)

        X_scaled_df = pd.DataFrame(X_scaled_array, columns=X.columns)

        # 3. Predizione cluster
        cluster = self.__modello.predict(X_scaled_df)[0]

        return cluster


