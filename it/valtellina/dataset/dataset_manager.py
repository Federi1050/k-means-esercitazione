import os

from matplotlib import pyplot as plt

from it.valtellina.dataset.creatore_grafici import CreatoreGrafici
from it.valtellina.dataset.manipolazione_dataset import ManipolazioneDataset


class DatasetManager:
    def __init__(self):
        self.__ds = self.scarica_dataset()

    def scarica_dataset(self):
        ds = ManipolazioneDataset.scarica_dataset()
        return ds

    def analisi_complessiva(self):
        # missing values
        missing = self.__ds.isnull().sum()

        # outliers
        outliers = ManipolazioneDataset.outliers(self.__ds)

        # distribuzione
        dist = ManipolazioneDataset.distribuzione(self.__ds)

        # normalità
        norm = ManipolazioneDataset.normality(self.__ds)

        # correlation
        correlation = self.__ds.corr()

        return {
            "missing": missing,
            "outliers": outliers,
            "dist": dist,
            "norm": norm,
            "correlation": correlation
        }

    def cleaning(self):
        self.__ds = ManipolazioneDataset.clean(self.__ds)

    def stampa_dataset(self, **kwargs):
        numero = kwargs.get("numero")

        if numero is not None:
            print(f"\nHEAD ({numero} righe):")
            print(self.__ds.head(numero))

            print(f"\nTAIL ({numero} righe):")
            print(self.__ds.tail(numero))
        else:
            print("\nDATASET COMPLETO:")
            print(self.__ds)

    def stampa_grafici(self, output_dir="output_grafici"):
        # se cartella non esiste creiamola
        os.makedirs(output_dir, exist_ok=True)

        # correlazione
        corr = CreatoreGrafici.plot_correlation(self.__ds)
        corr_path = os.path.join(output_dir, "correlation.png")
        corr.savefig(corr_path, bbox_inches="tight")
        plt.close(corr)

        # hist plots
        hist = []
        for colonna in self.__ds.columns:
            fig = CreatoreGrafici.plot_hist(self.__ds, colonna)
            path = os.path.join(output_dir, f"hist_{colonna}.png")
            fig.savefig(path, bbox_inches="tight")
            hist.append(fig)
            plt.close(fig)

        # box plot
        boxs = []
        for colonna in self.__ds.columns:
            fig = CreatoreGrafici.plot_box(self.__ds, colonna)
            path = os.path.join(output_dir, f"box_{colonna}.png")
            fig.savefig(path, bbox_inches="tight")
            boxs.append(fig)
            plt.close(fig)

        return {
            "hist": hist,
            "boxs": boxs,
            "correlation": corr
        }

    def getDataset(self):
        return self.__ds


