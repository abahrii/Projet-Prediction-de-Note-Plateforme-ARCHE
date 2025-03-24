# Prédiction de Note - Plateforme ARCHE

Ce projet a pour objectif de **prédire la note** obtenue par un apprenant dans un cours, en se basant sur ses **logs (traces numériques)** issues de la plateforme ARCHE (fichiers `logs_from_logs.csv` et `notes_from_notes.csv`). Le code utilise **Python**, **Streamlit** et **scikit-learn** pour extraire des variables, entraîner un modèle de régression, et proposer une interface web interactive.

---

## 1. Description du Projet

1. **Extraction des données**  
   - Les logs (fichier `logs_from_logs.csv`) contiennent les actions des apprenants sur la plateforme (horodatage, événement, etc.).  
   - Les notes (fichier `notes_from_notes.csv`) contiennent la note finale de chaque apprenant.

2. **Création des variables**  
   - **Nombre d’actions** (total par type d’événement).  
   - **Temps total passé** : calculé en cumulant les intervalles d’activité (seuil de 5 minutes).  
   - **Nombre de jours actifs**, **heure moyenne** d’activité, **interactions actives** vs. passives, **intervalle moyen** entre actions, etc.

3. **Modélisation**  
   - Utilisation de **scikit-learn** (régressions linéaires, Ridge, Lasso, ElasticNet, SVR, etc.) pour prédire la note.  
   - Validation croisée et optimisation via **GridSearchCV**.  
   - Sélection du **meilleur modèle** selon le score R² et sauvegarde dans `model_regression_notes.pkl`.

4. **Interface Streamlit**  
   - Saisie des variables (jours actifs, interactions actives, etc.).  
   - Bouton de **prédiction** pour obtenir la note estimée.  
   - **Graphiques de sensibilité** montrant l’impact d’une variable sur la note prédite (les autres restant fixes).

---

## 2. Structure Principale

- **`app_stream_notes.py`** : Application Streamlit principale qui :
  1. Charge le modèle (`model_regression_notes.pkl`).
  2. Permet de saisir les variables d’entrée.
  3. Affiche la prédiction et des graphiques d’analyse de sensibilité.

- **`projet_pkl.py`** : Script de modélisation qui :
  1. Charge et fusionne les données de logs et de notes.
  2. Extrait les variables (actions, temps total, etc.).
  3. Entraîne plusieurs modèles, compare leurs performances et enregistre le meilleur.

- **`Dockerfile`** et **`docker-compose.yml`** : Fichiers pour conteneuriser l’application et la lancer via Docker.

---


