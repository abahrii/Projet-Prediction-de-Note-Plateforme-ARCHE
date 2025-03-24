# Utiliser une image Python légère
FROM python:3.8

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Mettre à jour pip et installer les dépendances
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copier l’ensemble du code de l’application dans le conteneur
COPY model_regression_notes.pkl /app/

# Exposer le port utilisé par Streamlit
EXPOSE 8501

# Définir la commande à exécuter pour lancer l’application
CMD ["streamlit", "run", "app_stream_notes.py", "--server.port=8501", "--server.address=localhost"]


