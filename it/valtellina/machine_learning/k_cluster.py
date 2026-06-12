import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class KClusterModel:
    def __init__(self, numero_cluster, dataset):
        self.__modello = self.creazione_modello(numero_cluster, dataset)

    def creazione_modello(self, numero_cluster, dataset):
        modello = KMeans(
            n_clusters=numero_cluster,
            random_state=42,
            n_init="auto",
            max_iter = 300
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

    def valutazione_modello(self):
        pass
        # prendendo ultima colonna dataset vedere quanti corrispondenti
        # siluette quanto performa
        # metrica ari
        

    def classificazione(self, object):
        pass

