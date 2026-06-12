import numpy as np
import pandas as pd
from scipy.stats import zscore, shapiro, normaltest
from sklearn.preprocessing import StandardScaler

class ManipolazioneDataset():

    @staticmethod
    def scarica_dataset():
        df = pd.read_csv(
            "dataset/seeds_dataset.txt",
            sep=r"\s+",
            header=None,
            names=[
                "area",
                "perimeter",
                "compactness",
                "kernel_length",
                "kernel_width",
                "asymmetry",
                "groove_length",
                "class"
            ]
        )
        return df

    @staticmethod
    def outliers(dataset):
        return {
            "iqr score" : ManipolazioneDataset.__iqrScore(dataset),
            "z score" : ManipolazioneDataset.__zScore(dataset)
        }

    @staticmethod
    def __iqrScore(data):
        # tra 2 e 3 quartile
        result = {}
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers_count = data[(data[col] < lower) | (data[col] > upper)].shape[0]
            result[col] = outliers_count
        return result

    @staticmethod
    def __zScore(data, threshold = 3):
        # quanti sono
        result = {}
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            col_data = data[col].dropna()
            if col_data.std() == 0:
                result[col] = 0
                continue
            z_scores = zscore(col_data)
            outliers_count = np.sum(np.abs(z_scores) > threshold)
            result[col] = int(outliers_count)
        return result

    @staticmethod
    def distribuzione(data):
        # Test di Shapiro-Wilk per verificare la normalità.
        # H0: i dati seguono una distribuzione normale.
        # p-value < 0.05 -> rifiuto H0 (dati non normali)

        num_cols = data.select_dtypes(include=['number']).columns
        results = {}

        for col in num_cols:
            stat, p_value = shapiro(data[col])

            # p-value > 0.05 significa che non ho evidenze
            # sufficienti per rifiutare la normalità
            is_normal = p_value > 0.05

            results[col] = {
                "Shapiro_stat": round(stat, 4),
                "p-value": round(p_value, 4),
                "is_normal": is_normal
            }

        return results

    @staticmethod
    def normality(data, alpha=0.05):
        numeric_cols = data.select_dtypes(include=np.number).columns

        risultati = []

        for col in numeric_cols:

            x = data[col].dropna()

            if len(x) < 8:
                risultati.append({
                    "variabile": col,
                    "n": len(x),
                    "test": None,
                    "statistica": np.nan,
                    "p_value": np.nan,
                    "normale": None,
                    "note": "Campione troppo piccolo"
                })
                continue

            # Shapiro per n <= 5000
            if len(x) <= 5000:
                stat, p = shapiro(x)
                test = "Shapiro-Wilk"

            # D'Agostino-Pearson per n > 5000
            else:
                stat, p = normaltest(x)
                test = "D'Agostino-Pearson"

            risultati.append({
                "variabile": col,
                "n": len(x),
                "test": test,
                "statistica": float(round(stat, 4)),
                "p_value": float(round(p, 6)),
                "normale": bool(p > alpha),
                "note": ""
            })

        return risultati

    @staticmethod
    def clean(data):
        # 1. rimuove duplicati
        data = data.drop_duplicates()

        # 2. separa features (ultima colonna = class/grouping)
        X = data.iloc[:, :-1]

        # 3. scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 4. ritorno come DataFrame
        clean_df = pd.DataFrame(X_scaled, columns=X.columns)

        return clean_df, scaler