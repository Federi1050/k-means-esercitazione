# Seeds Classifier

## K-means esercitazione
Progetto di Combi Federico, Meni Gianluca.

Il progetto Seeds Classifier è un'API in Flask 
progettata per analizzare il dataset Seeds e classificare i record 
in clusters. Inoltre, dato un nuovo campione, permette di classificarlo;
il tutto tramite l'utilizzo dell'algoritmo K-means.

## DATASET
Il dataset si trova a questo link: https://archive.ics.uci.edu/dataset/236/seeds.

Il dataset è un file ".txt" composto dalle seguenti feature:
1. area A 
2. perimeter P 
3. compactness C = 4*pi*A/P^2 
4. length of kernel
5. width of kernel
6. asymmetry coefficient
7. length of kernel groove

## Funzionalità Principali

- **Analisi del dataset**: 
Esplorazione delle feature  del dataset, 
analisi delle distribuzioni e delle frequenze delle classi, 
studio delle relazioni tra variabili (EDA)
- **Preprocessing dei dati**: 
Scaling e PCA (opzionale) dei dati per l'applicazione del modello non supervisionato.
- **Utilizzo del modello di Machine Learning**:
Applicazione del modello K-means,
con valutazione delle performance tramite metriche
di Silhouette e ARI (Adjusted Rand Index).
- **Tuning degli Ipermparametri**: 
Confronto dei possibili valori di k (numero cluster) e relative prestazioni
per identificare la miglior configurazione di parametri.

## Avvio di Seeds Classifier

### Requisiti

Prima di avviare Seeds Classifier, assicurati di avere installato tutte le dipendenze necessarie. Puoi trovarle nel file `requirements.txt`.

### Avvio in locale

Per visualizzare l'API in locale è necessario recarsi al seguente indirizzo:

   ```
   http://127.0.0.1:5000/
   ```
## Avvio con Docker

### Prerequisiti

Assicurati di avere installato Docker. Puoi scaricare l'applicazione [qui](https://www.docker.com/products/docker-desktop/).

Verifica l’installazione con:

```bash
docker --version
```
### Utilizzo di Docker

1. Costruisci l'immagine Docker da linea di comando:
   ```bash
   docker build -t seeds-classifier .
   ```
   
2. Si possono controllare le informazioni dell'immagine appena creata con il comando:
   ```bash
   docker image ls
   ```

3. Costruisci e avvia il container da linea di comando (copiando questo comando, verrà chiamato "titanic"):
   ```bash
   docker run -d --name seeds -p 5000:5000 seeds-classifier
   ```
   Se l'operazione è andata a buon fine è possibile vedere lo stavo attivo del container tramite il comando:
   ```bash
   docker ps
   ```

4. Accedi all'applicazione (nella sua route home) tramite il tuo browser all'indirizzo:
   ```
   http://127.0.0.1:5000/
   ```
## Utilizzo dell'API

L'API è consultabile direttamente da browser.

Se dovesse servire, è possibile installare dei plug-in, tra cui:

- Rest-Client (Chrome): [download](https://chromewebstore.google.com/detail/rest-client/oienkoejnhkbcibhdnpjoemdnmiokgah)
- Rested (Firefox): [download](https://addons.mozilla.org/en-US/firefox/addon/rested/)

L'utilizzo dei plug-in non permette però la restituzione delle immagini, facendo risultare "strana" la risposta di alcuni endpoint.

## API Endpoints di Seeds Classifier

Seeds Classifier è provvisto di diversi endpoint GET e POST, consultabili nella route **home**.
```
http://127.0.0.1:5000/
```
Le funzionalità dell'API sono le seguenti:

### Show Dataset

Endpoint che permette di visualizzare una parte oppure tutto il dataset. Per ottenere solo una parte
del dataset è necessario specificare il numero di righe nella richiesta: {"numero": 10}
```
http://127.0.0.1:5000/show_data
```

### Analisi

Endpoint che restituisce un’analisi statistica completa del dataset, che comprende missing values,
outliers, distribuzione, test di normalità e correlazione
```
http://127.0.0.1:5000/analisi
```

### Grafici

Endpoint che genera e restituisce tutti i grafici esplorativi del dataset in formato immagine
```
http://127.0.0.1:5000/grafici
```

### Grafico Elbow Method

Endpoint che Genera il grafico del metodo del gomito per la scelta del numero ottimale di cluster K
```
http://127.0.0.1:5000/grafico_gomito
```

### Valutazione Metriche Modello

Endpoint che valuta il modello K-Means usando metriche di clustering
```
http://127.0.0.1:5000/val_kCluster
```

### Previsione Cluster su nuovo campione

Endpoint che esegue la previsione del cluster per un nuovo campione di semi.
Bisogna strutturare la richiesta JSON seguendo la struttura del dataset:
```
{
  "area": 14.5,
  "perimeter": 15.2,
  "compactness": 0.87,
  "kernel_length": 5.6,
  "kernel_width": 3.2,
  "asymmetry": 2.1,
  "groove_length": 5.0
}
```

```
http://127.0.0.1:5000/previsione_kCluster
```
