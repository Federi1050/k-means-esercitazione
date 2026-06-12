FROM python:3.11
WORKDIR /k-means-esercitazione
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]