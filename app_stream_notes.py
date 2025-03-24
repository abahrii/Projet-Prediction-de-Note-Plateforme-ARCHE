import streamlit as st 
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Titre principal de l'application
st.title("Prédiction de Note - Plateforme ARCHE")

# Chargement du modèle sauvegardé avec st.cache_resource (sans argument allow_output_mutation)
@st.cache_resource
def load_model():
    
    model = joblib.load("model_regression_notes.pkl")
    return model

# Charger le modèle
model = load_model()

# ------------------------------------------------------------------------------ 
# Section de saisie des variables d'entrée dans la sidebar
# ------------------------------------------------------------------------------ 
st.sidebar.header("Entrées Utilisateur")
st.sidebar.write("Saisissez les valeurs des variables utilisées pour la prédiction :")

# Variables d'entrée pour le modèle
unique_days = st.sidebar.number_input("Nombre de jours actifs", min_value=0, value=5, step=1)
active_interactions = st.sidebar.number_input("Nombre d'interactions actives", min_value=0, value=20, step=1)
travail_devoir_remis = st.sidebar.number_input("Travail de devoir remis (compte)", min_value=0, value=10, step=1)
questionnaire_soumis = st.sidebar.number_input("Questionnaire soumis (compte)", min_value=0, value=5, step=1)
fichier_depose = st.sidebar.number_input("Fichier déposé (compte)", min_value=0, value=5, step=1)
travail_devoir_cree = st.sidebar.number_input("Travail de devoir créé (compte)", min_value=0, value=3, step=1)

st.sidebar.write("Cette application prédit la note des apprenants à partir des logs et notes historiques.")

# ------------------------------------------------------------------------------ 
# Prédiction unique : Bouton de prédiction
# ------------------------------------------------------------------------------ 
if st.button("Prédire la note"):
    # Création du vecteur de caractéristiques dans l'ordre attendu par le modèle :
    # [unique_days, active_interactions, Travail de devoir remis_count, Questionnaire soumis_count, Fichier déposé_count, Travail de devoir créé_count]
    X_new = np.array([[unique_days, active_interactions, travail_devoir_remis, questionnaire_soumis, fichier_depose, travail_devoir_cree]])
    prediction = model.predict(X_new)
    st.success(f"La note prédite est : {prediction[0]:.2f}")

# ------------------------------------------------------------------------------ 
# Graphique robuste : Sensibilité de la note prédite en fonction de deux variables
# ------------------------------------------------------------------------------ 
st.header("Graphique robuste de Sensibilité")

# Valeurs fixes pour les autres variables
fixed_travail_devoir_remis = travail_devoir_remis
fixed_questionnaire_soumis = questionnaire_soumis
fixed_fichier_depose = fichier_depose
fixed_travail_devoir_cree = travail_devoir_cree

# Définir les plages de variation pour 'unique_days' et 'active_interactions'
unique_days_range = np.linspace(1, 30, 30)
active_interactions_range = np.linspace(5, 100, 30)

# Création d'une grille 2D
U, A = np.meshgrid(unique_days_range, active_interactions_range)

# Calculer la note prédite pour chaque combinaison
predictions = np.zeros(U.shape)
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        X_temp = np.array([[U[i, j], A[i, j], fixed_travail_devoir_remis,
                            fixed_questionnaire_soumis, fixed_fichier_depose, fixed_travail_devoir_cree]])
        predictions[i, j] = model.predict(X_temp)[0]

# Création du graphique en carte de contours
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(U, A, predictions, cmap='viridis')
fig.colorbar(contour, ax=ax, label="Note prédite")
ax.set_xlabel("Nombre de jours actifs")
ax.set_ylabel("Nombre d'interactions actives")
ax.set_title("Sensibilité de la note prédite")
st.pyplot(fig)
