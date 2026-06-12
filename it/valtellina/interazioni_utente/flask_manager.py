import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template_string
from io import BytesIO
import base64

from matplotlib import pyplot as plt

from it.valtellina.dataset.dataset_manager import DatasetManager
from it.valtellina.machine_learning.k_cluster import KClusterModel


class FlaskManager():  # è una classe INTERFACCIA
    def __init__(self):
        self.app = Flask(__name__)
        self.__register_routes()  # inizializzazione delle varie root tutte insieme

        self.ds_mg = DatasetManager()
        self.__colonna_labels = self.ds_mg.getDataset()["class"]
        scaler, pca = self.ds_mg.cleaning()
        self.kclaster = KClusterModel(3, self.ds_mg.getDataset(), scaler, pca)

    def run(self, **kwargs):
        self.app.run(**kwargs)  # **kwargs non specifico i parametri che dopo dovrò inserire

    def __register_routes(self):  # __ indica un metodo che vede solo questa classe
        @self.app.route('/')  # in automatico il metodo è GET
        def home():
            possibili_routes = """
            <h1>Seeds API</h1>

            <p>
                REST API per l'analisi del dataset Seeds e la valutazione
                di modelli di Machine Learning per la previsione della sopravvivenza.
            </p>

            <h2>Available Endpoints</h2>

            <h3>Dataset Analysis</h3>
            <ul>
                <li>GET /api/missing-values</li>
                <li>GET /api/outliers</li>
                <li>GET /api/cleaning-data</li>
            </ul>

            <h3>Plots</h3>
            <ul>
                <li>GET /api/plot-survival-stats</li>
                <li>GET /api/plot-distribution/&lt;feature&gt;</li>
                <li>GET /api/plot-outliers/&lt;feature&gt;</li>
                <li>GET /api/correlation-matrix</li>
            </ul>

            <h3>Machine Learning Models</h3>
            <ul>
                <li>GET /api/logistic-regression-naive</li>
                <li>GET /api/logistic-regression-grid</li>
                <li>GET /api/XGBoost</li>
                <li>GET /api/XGBoost-encoded</li>
            </ul>

            <h2>Examples</h2>
            <pre>
            GET /api/plot-distribution/Age

            GET /api/plot-outliers/Fare

            GET /api/logistic-regression-grid
            </pre>
        """
            return possibili_routes

        @self.app.route('/show_data', methods=['POST'])
        def dataset_show():
            input = request.get_json()
            numero = input.get('numero')
            if not numero:
                risposta = self.ds_mg.stampa_dataset()
            else:
                risposta = self.ds_mg.stampa_dataset(numero = numero)
            return jsonify(risposta.to_dict(orient="records"))  # per ritornare un dataframe si usa .to_dict()

        @self.app.route('/analisi')
        def info():
            risp = self.ds_mg.analisi_complessiva()
            risp = self.converti(risp)
            return jsonify(risp)

        @self.app.route('/grafici')
        def grafici():
            grafici = self.ds_mg.stampa_grafici("output_grafici/grafici_post_cleaning")
            figs = []
            for value in grafici.values():  # bisogna scomporre la scritta dall'immagine
                if value is None:
                    continue

                # Se è una lista di figure
                if isinstance(value, list):  # infatti gli istogrammi sono più immagini in una lista
                    figs.extend(value)
                else:
                    figs.append(value)

            images = []

            for fig in figs:
                img = BytesIO()
                fig.savefig(img, format="png", bbox_inches="tight")
                img.seek(0)

                encoded = base64.b64encode(img.getvalue()).decode()
                images.append(encoded)

                plt.close(fig)

            html = """
               <h1>Plots</h1>
               {% for img in images %}
                   <img src="data:image/png;base64,{{ img }}" style="margin:10px;">
               {% endfor %}
               """

            return render_template_string(html, images=images)  # prende il template html e usa su tutte le immagini
            # trasformate in formato base64

        @self.app.route('/grafico_gomito')
        def grafico_gomito():
            fig = KClusterModel.inerzie(self.ds_mg.getDataset())
            img = BytesIO()
            fig.savefig(img, format="png", bbox_inches="tight")
            img.seek(0)
            plt.close(fig)
            encoded = base64.b64encode(img.getvalue()).decode()
            html = """
                <h1>Plots</h1>
                <img src="data:image/png;base64,{{ encoded }}" style="margin:10px;">
            """
            return render_template_string(html, encoded=encoded)

        @self.app.route('/val_kCluster')
        def valMod_kCluster():
            val = self.kclaster.valutazione_modello(self.__colonna_labels, self.ds_mg.getDataset())
            return jsonify(val)

        @self.app.route('/previsione_kCluster', methods=['POST'])
        def previsione_kCluster():
            data = request.get_json()
            obj = {
                'area': data.get('area'),
                'perimeter': data.get('perimeter'),
                'compactness': data.get('compactness'),
                'kernel_length': data.get('kernel_length'),
                'kernel_width': data.get('kernel_width'),
                'asymmetry': data.get('asymmetry'),
                'groove_length': data.get('groove_length')
            }  # creando questo dizionario posso controllare che siano presenti tutti gli attributi necessari
            pred = self.kclaster.classificazione(obj)
            print("PREDIZIONE:", pred)

            return jsonify({"cluster numero": self.kclaster.classificazione(obj).tolist()})


    def converti(self, obj):
        # pandas
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")

        if isinstance(obj, pd.Series):
            return obj.to_dict()

        # numpy scalars
        if isinstance(obj, np.generic):
            return obj.item()

        # dict
        if isinstance(obj, dict):
            return {k: self.converti(v) for k, v in obj.items()}

        # list / tuple
        if isinstance(obj, (list, tuple)):
            return [self.converti(x) for x in obj]

        # python native types (bool, int, float, str, None)
        if isinstance(obj, (bool, int, float, str)) or obj is None:
            return obj

        return obj