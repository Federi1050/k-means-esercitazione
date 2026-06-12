import matplotlib.pyplot as plt
import seaborn as sns

class CreatoreGrafici():

    @staticmethod
    def plot_correlation(data):
        corr = data.corr(numeric_only=True)
        fig = plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice di correlazione")
        # plt.show()
        return fig

    @staticmethod
    def plot_hist(data, col):
        fig = plt.figure(figsize=(6, 4))
        plt.hist(
            data[col].dropna(),
            bins=30,
            edgecolor='black',
            linewidth=2)
        plt.title(f"Distribuzione di {col}")
        # plt.show()
        return fig

    @staticmethod
    def plot_box(data, col):
        fig = plt.figure(figsize=(6, 4))
        sns.boxplot(y=data[col])
        plt.title(f"Boxplot di {col}")
        # plt.show()
        return fig

