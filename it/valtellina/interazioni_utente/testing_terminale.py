from it.valtellina.dataset.dataset_manager import DatasetManager
from it.valtellina.machine_learning.k_cluster import KClusterModel


class TestingTerminale:

    @staticmethod
    def test():
        print("Carico il mio CSV")
        ds_mg = DatasetManager()
        print()

        print("Stampa dataset")
        ds_mg.stampa_dataset()
        print()
        ds_mg.stampa_dataset(numero=7)
        print()

        print("Analisi del dataset")
        analisi = ds_mg.analisi_complessiva()
        TestingTerminale.stampa_analisi(analisi)
        print()

        print("Stampa grafici")
        #ds_mg.stampa_grafici(output_dir="output_grafici/grafici_pre_cleaning")
        print()

        print("Clening dei dati")
        ds_mg.cleaning()
        print()

        print("Analisi del dataset dopo cleaning")
        analisi = ds_mg.analisi_complessiva()
        TestingTerminale.stampa_analisi(analisi)
        print()

        print("Stampa grafici dopo cleaning")
        #ds_mg.stampa_grafici(output_dir="output_grafici/grafici_post_cleaning")
        print()

        print("Creazione del grafico gomito per k-cluster")
        KClusterModel.inerzie(ds_mg.getDataset())

        print("Creazione modello k-cluster")
        kcluster = KClusterModel(3, ds_mg.getDataset())
        print()

        print("Valutazione modello k-cluster")
        val = kcluster.valutazione_modello()
        print(val)
        print()


    @staticmethod
    def stampa_analisi(analisi):
        # MISSING VALUES
        print("\nMISSING VALUES")
        print(analisi["missing"])

        # OUTLIERS
        print("\nOUTLIERS (IQR)")
        print(analisi["outliers"]["iqr score"])

        print("\nOUTLIERS (Z-SCORE)")
        print(analisi["outliers"]["z score"])

        # NORMALITÀ
        print("\nNORMALITÀ (Shapiro-Wilk)")
        for r in analisi["norm"]:
            print(
                f"- {r['variabile']:15} | "
                f"p-value: {r['p_value']:.4f} | "
                f"normale: {r['normale']}"
            )

        # CORRELAZIONE
        print("\nMATRICE DI CORRELAZIONE")
        print(analisi["correlation"].round(3))

        print("\n" + "=" * 60)