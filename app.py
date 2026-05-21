import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 1. Charger le modèle et les moyennes avec cache
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

@st.cache_data
def load_moyennes():
    return joblib.load("moyennes.pkl")

model = load_model()
moyennes = load_moyennes()

# 2. Configuration page
st.set_page_config(page_title="Conso Predikt", page_icon="⚡")
st.title("⚡ Conso Predikt")
st.markdown("Prédisez la consommation électrique de votre maison en quelques secondes.")

# 3. Inputs utilisateur
hour   = st.slider("🕐 Heure de la journée", 0, 23, 12)
t_out  = st.slider("🌡️ Température extérieure (°C)", -5.0, 40.0, 20.0)
rh_out = st.slider("💧 Humidité extérieure (%)", 20.0, 100.0, 50.0)

# 4. Bouton prédire
if st.button(" Prédire la consommation"):

    # Remplir avec les moyennes
    entree = moyennes.copy()
    entree["hour"]   = hour
    entree["T_out"]  = t_out
    entree["RH_out"] = rh_out

    # Convertir en DataFrame
    df_entree = pd.DataFrame([entree])

    # Prédire
    prediction = model.predict(df_entree)[0]

    # Afficher avec animation
    st.snow()
    st.success(f"Consommation estimée : **{prediction:.2f} Wh**")

    # Comparaison avec la moyenne globale des consommations
    moyenne_globale = moyennes.mean()  # moyenne de toutes les features
    st.write(f" Consommation moyenne globale des données : **{moyenne_globale:.2f} Wh**")

    # Graphique comparatif simple
    fig, ax = plt.subplots(figsize=(4,2))
    bars = ax.bar(["Prédiction", "Moyenne globale"], [prediction, moyenne_globale],
                  color=["#3498db","#95a5a6"], edgecolor="white")
    ax.set_ylabel("Wh")
    ax.set_title("Comparaison prédiction vs moyenne globale")

    # Ajouter les valeurs au-dessus des barres
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval:.0f}", ha="center", fontsize=7)

    st.pyplot(fig)

# Footer personnalisé
st.markdown("---")
st.caption("Projet ML — Akim Ratiarison, étudiant en Data Science")
