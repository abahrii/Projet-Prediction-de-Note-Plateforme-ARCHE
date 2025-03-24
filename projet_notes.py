#!/usr/bin/env python3
import sys
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from time import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# Définition des chemins des fichiers de données
DATA_SET_LOGS = "logs_from_logs.csv"
DATA_SET_NOTES = "notes_from_notes.csv"

# ------------------------------------------------------------------------------
# Classe pour le chargement des données
# ------------------------------------------------------------------------------
class DataLoader:
    def __init__(self, logs_path=DATA_SET_LOGS, notes_path=DATA_SET_NOTES):
        self.logs_path = logs_path
        self.notes_path = notes_path

    def load_logs(self) -> pd.DataFrame:
        """
        Charger les données de logs depuis un fichier CSV.
        Le fichier contenir une colonne 'heure' au format date.
        """
        try:
            data = pd.read_csv(self.logs_path, parse_dates=["heure"])
            return data
        except Exception as e:
            logging.error(f"Erreur de chargement du dataset: {self.logs_path}")
            traceback.print_exc()
            sys.exit()

    def load_notes(self) -> pd.DataFrame:
        """
        Charger les notes des apprenants depuis un fichier CSV.
        Convertit la colonne 'note' en numérique et supprime les valeurs manquantes.
        """
        try:
            data = pd.read_csv(self.notes_path)
            data['note'] = pd.to_numeric(data['note'], errors='coerce')
            data.dropna(inplace=True)
            data['note'] = data['note'].astype(int)
            return data
        except Exception as e:
            logging.error(f"Erreur de chargement du dataset: {self.notes_path}")
            traceback.print_exc()
            sys.exit()

# ------------------------------------------------------------------------------
# Classe pour l'extraction et la visualisation des variables
# ------------------------------------------------------------------------------
class FeatureExtractor:
    def __init__(self, logs: pd.DataFrame):
        self.logs = logs

    def nb_action(self) -> pd.DataFrame:
        """
        Calculer le nombre d'actions réalisées par chaque apprenant.
        Retourne une DataFrame avec 'pseudo' et 'nb_actions', triée par ordre décroissant.
        """
        res = self.logs.groupby("pseudo").size().reset_index(name="nb_actions")
        return res.sort_values("nb_actions", ascending=False)

    def visualisation_nb_actions(self, nb_actions_df: pd.DataFrame):
        """
        Visualiser le nombre d'actions par apprenant à l'aide d'un diagramme en barres.
        """
        plt.figure(figsize=(10, 5))
        plt.bar(nb_actions_df["pseudo"], nb_actions_df["nb_actions"], color="blue")
        plt.xlabel("Apprenant (pseudo)")
        plt.ylabel("Nombre d'actions")
        plt.title("Nombre d'actions par apprenant")
        plt.show()

    def _cumul_temp(self, x: pd.Series) -> float:
        """
        Calculer le temps cumulé (en secondes) sur une série d'horodatages.
        On cumule uniquement les intervalles inférieurs à 300 secondes pour ignorer de longues pauses.
        """
        t0 = x.iloc[0]
        total = 0
        for t in x.values:
            diff = pd.Timedelta(t - t0).seconds
            if diff < 300:
                total += diff
            t0 = t
        return total

    def calculer_temps(self) -> pd.DataFrame:
        """
        Calculer le temps passé par jour par chaque apprenant.
        Retourne une DataFrame avec 'pseudo', 'jour' et 'temps_jour'.
        """
        self.logs["jour"] = self.logs["heure"].dt.date
        hr = self.logs.groupby(["pseudo", "jour"]).agg(temps_jour=("heure", self._cumul_temp)).reset_index()
        hr["jour"] = pd.to_datetime(hr["jour"])
        return hr

    def compute_intervalle_moyen(self) -> pd.DataFrame:
        """
        Calculer l'intervalle moyen (en secondes) entre les actions pour chaque utilisateur.
        """
        def moyenne_diff(group):
            group = group.sort_values("heure")
            diffs = group["heure"].diff().dropna().dt.total_seconds()
            return diffs.mean() if len(diffs) > 0 else np.nan
        intervalle = self.logs.groupby("pseudo").apply(moyenne_diff).reset_index(name="intervalle_moyen")
        return intervalle

    def extraire_autres_variables(self) -> pd.DataFrame:
        """
        Extraire des variables supplémentaires à partir des logs.
        Ces variables incluent :
          - Nombre total d'actions
          - Temps total passé (en secondes) et transformation logarithmique du temps total
          - Nombre de jours actifs
          - Nombre d'événements distincts
          - Heure moyenne d'activité
          - Interactions actives et passives, et leurs ratios
          - Nombre d'occurrences pour chaque événement actif spécifique
          - Intervalle moyen entre les actions
        """
        nb_actions_df = self.nb_action()
        temps_df = self.calculer_temps()
        total_temps_df = temps_df.groupby("pseudo")["temps_jour"].sum().reset_index()
        total_temps_df.rename(columns={"temps_jour": "temps_total"}, inplace=True)
        total_temps_df["log_temps_total"] = np.log(total_temps_df["temps_total"] + 1)
        
        unique_days_df = self.logs.groupby("pseudo")["heure"].apply(lambda x: x.dt.date.nunique()).reset_index(name="unique_days")
        unique_events_df = self.logs.groupby("pseudo")["evenement"].nunique().reset_index(name="unique_events")
        self.logs["heure_jour"] = self.logs["heure"].dt.hour
        mean_hour_df = self.logs.groupby("pseudo")["heure_jour"].mean().reset_index(name="moy_heure")
        
        active_events = [
            "Travail de devoir remis", "Questionnaire soumis", "Fichier déposé",
            "Discussion consultée", "Travail de devoir créé"
        ]
        passive_events = [
            "Module de cours consulté", "Cours consulté", "Rapport de session consulté",
            "Statut du travail remis consulté", "Rapport d’évaluation utilisateur consulté"
        ]
        self.logs["active"] = self.logs["evenement"].isin(active_events).astype(int)
        self.logs["passive"] = self.logs["evenement"].isin(passive_events).astype(int)
        active_df = self.logs.groupby("pseudo")["active"].sum().reset_index(name="active_interactions")
        passive_df = self.logs.groupby("pseudo")["passive"].sum().reset_index(name="passive_interactions")
        
        features = nb_actions_df.merge(total_temps_df, on="pseudo", how="inner")
        features = features.merge(unique_days_df, on="pseudo", how="inner")
        features = features.merge(unique_events_df, on="pseudo", how="inner")
        features = features.merge(mean_hour_df, on="pseudo", how="inner")
        features = features.merge(active_df, on="pseudo", how="inner")
        features = features.merge(passive_df, on="pseudo", how="inner")
        features["active_ratio"] = features["active_interactions"] / features["nb_actions"]
        features["passive_ratio"] = features["passive_interactions"] / features["nb_actions"]
        
        for event in active_events:
            event_df = self.logs[self.logs["evenement"] == event].groupby("pseudo").size().reset_index(name=event + "_count")
            features = features.merge(event_df, on="pseudo", how="left")
        
        features.fillna(0, inplace=True)
        intervalle_df = self.compute_intervalle_moyen()
        features = features.merge(intervalle_df, on="pseudo", how="left")
        features["intervalle_moyen"].fillna(0, inplace=True)
        
        return features

    def afficher_data_correlation(self, data: pd.DataFrame):
        """
        Afficher une heatmap de la matrice de corrélation pour un ensemble de variables sélectionnées.
        """
        cols = ['note', 'moy_heure', 'temps_total', 'log_temps_total', 'nb_actions', 'unique_events',
                'active_interactions', 'passive_interactions', 'active_ratio',
                'unique_days', 'passive_ratio', 'Travail de devoir remis_count', 
                'Questionnaire soumis_count', 'Fichier déposé_count', 
                'Discussion consultée_count', 'Travail de devoir créé_count',
                'intervalle_moyen']
        correlation = data[cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Matrice de corrélation")
        plt.show()

# ------------------------------------------------------------------------------
# Classe pour l'entraînement et l'évaluation des modèles
# ------------------------------------------------------------------------------
class ModelTrainer:
    def __init__(self, data_model: pd.DataFrame):
        self.data_model = data_model

    def modelisation(self):
        """
        Réalise la modélisation en entraînant plusieurs modèles de régression.
        Effectue une validation croisée via GridSearchCV pour certains modèles.
        Sauvegarde le meilleur modèle (selon le score R² sur le test) dans un fichier pickle.
        Affiche des graphiques comparant les prédictions aux valeurs réelles.
        """
        # Sélection des features pour la modélisation
        features_list = [ 'unique_days',  'active_interactions', 'Travail de devoir remis_count', 
                          'Questionnaire soumis_count', 'Fichier déposé_count', 
                          'Travail de devoir créé_count']
        
        correlation = data_model[features_list + ['note']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Matrice de corrélation")
        plt.show()

        X = self.data_model[features_list]
        y = self.data_model['note']
        
        # Division des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Définition des grilles de paramètres pour certains modèles
        param_ridge = {'alpha': [1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 30, 50, 100, 200]}
        param_lasso = {'alpha': [1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 30, 50, 100, 200]}
        param_elasticnet = {
            'alpha': [1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 30, 50, 100, 200],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        param_svr = {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 1, 10]}
        
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
            "SVR (Linear Kernel)": SVR(kernel='linear'),
            "LogisticRegression": LogisticRegression(),
            "Ridge_CV": GridSearchCV(Ridge(), param_grid=param_ridge, scoring='r2', cv=5),
            "Lasso_CV": GridSearchCV(Lasso(), param_grid=param_lasso, scoring='r2', cv=5),
            "ElasticNet_CV": GridSearchCV(ElasticNet(), param_grid=param_elasticnet, scoring='r2', cv=5),
            "SVR_CV": GridSearchCV(SVR(kernel='linear'), param_grid=param_svr, scoring='r2', cv=5),
            "RandomForest": RandomForestRegressor(n_estimators=100),
            "XGBoost": XGBRegressor(objective='reg:squarederror')
        }
        
        results = {}       # Stockage des métriques de performance
        models_fitted = {} # Stockage des modèles entraînés
        
        for name, mdl in models.items():
            mdl.fit(X_train_scaled, y_train)
            models_fitted[name] = mdl
            y_train_pred = mdl.predict(X_train_scaled)
            y_test_pred = mdl.predict(X_test_scaled)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            r2_test = r2_score(y_test, y_test_pred)
            r2_train = r2_score(y_train, y_train_pred)
            results[name] = {"Model": name, "RMSE": rmse, "R²_test": r2_test, "R²_train": r2_train}
            
            # Visualisation des prédictions vs valeurs réelles
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_test_pred, alpha=0.7,
                        label=f"{name} (R²_test={r2_test:.4f}, R²_train={r2_train:.4f})")
            plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
            plt.xlabel("Notes réelles")
            plt.ylabel("Notes prédites")
            plt.title(f"Prédictions vs Réalité - {name}")
            plt.legend()
            plt.show()
        
        # Sélection du meilleur modèle basé sur R²_test
        best_model_name = max(results, key=lambda x: results[x]["R²_test"])
        best_r2 = results[best_model_name]["R²_test"]
        best_model = models_fitted[best_model_name]
        joblib.dump(best_model, "model_regression_notes.pkl")
        print(f"\nLe meilleur modèle ({best_model_name} avec R²_test = {best_r2:.4f}) a été sauvegardé sous 'model_regression_notes.pkl'.")
        
        return results, models_fitted

# ------------------------------------------------------------------------------
# Partie principale du programme
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Chargement des données
    loader = DataLoader()
    logs = loader.load_logs()
    notes = loader.load_notes()
    
    # Visualisation rapide du nombre d'actions par utilisateur
    extractor = FeatureExtractor(logs)
    nb_actions_df = extractor.nb_action()
    extractor.visualisation_nb_actions(nb_actions_df)
    
    # Extraction des variables issues des logs
    features = extractor.extraire_autres_variables()
    
    # Fusion des features avec les notes (basée sur 'pseudo')
    data_model = features.merge(notes, on="pseudo", how="inner")
    print("Aperçu des données utilisées pour la modélisation :")
    print(data_model.head())
    
    # Affichage de la matrice de corrélation entre les variables
    extractor.afficher_data_correlation(data_model)
    
    # Exécution de la modélisation et comparaison des modèles
    trainer = ModelTrainer(data_model)
    results, models_fitted = trainer.modelisation()
    print("\nComparaison des modèles (modélisation simple vs validation croisée) :")
    for model_name, metrics in results.items():
        print(f"{model_name}: RMSE = {metrics['RMSE']:.2f}, R²_train = {metrics['R²_train']:.4f}, R²_test = {metrics['R²_test']:.4f}")
