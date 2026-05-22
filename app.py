"""
⚡ Conso Predikt — Application Streamlit
Prédiction de la consommation électrique d'une maison
Modèle : Random Forest allégé (10 features, 100 arbres)
Dataset : UCI Appliances Energy Prediction
Auteur  : Akim Ratiarison — L2 IAD
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np


# 1. CONFIGURATION DE LA PAGE
#
st.set_page_config(
    page_title="Conso Predikt",
    page_icon="⚡",
    layout="wide"
)


# 2. CHARGEMENT DU MODÈLE

@st.cache_resource
def charger_modele():
    model         = joblib.load('random_forest_model_light.pkl')
    features      = joblib.load('features.pkl')
    moyennes      = joblib.load('moyennes_light.pkl')
    moyenne_cible = joblib.load('moyenne_cible.pkl')
    return model, features, moyennes, moyenne_cible

model, features, moyennes, moyenne_cible = charger_modele()


BORNES = {
    'hour':        (0,     23,    12,    1),
    'T3':          (17.0,  29.0,  22.0,  0.5),
    'Press_mm_hg': (729.0, 772.0, 755.0, 0.5),
    'RH_3':        (28.0,  50.0,  39.0,  1.0),
    'T_out':       (-5.0,  26.0,  10.0,  0.5),
    'RH_2':        (20.0,  56.0,  40.0,  1.0),
    'RH_1':        (27.0,  63.0,  40.0,  1.0),
    'RH_9':        (29.0,  53.0,  41.0,  1.0),
    'RH_5':        (29.0,  96.0,  50.0,  1.0),
    'T8':          (16.0,  27.0,  22.0,  0.5),
}

LABELS = {
    'hour':        '🕐 Heure de la journée',
    'T3':          '🌡️ Température pièce principale (°C)',
    'Press_mm_hg': '🌬️ Pression atmosphérique (mm Hg)',
    'RH_3':        '💧 Humidité pièce principale (%)',
    'T_out':       '🌤️ Température extérieure (°C)',
    'RH_2':        '💧 Humidité pièce 2 (%)',
    'RH_1':        '💧 Humidité pièce 1 (%)',
    'RH_9':        '💧 Humidité pièce 9 (%)',
    'RH_5':        '💧 Humidité pièce 5 (%)',
    'T8':          '🌡️ Température pièce secondaire (°C)',
}


# 3. HEADER
st.markdown("""
    <div style='text-align:center; padding:20px 0 5px 0;'>
        <h1 style='font-size:2.8em; margin-bottom:0;'>⚡ Conso Predikt</h1>
        <p style='color:gray; font-size:1.05em; margin-top:5px;'>
            Prédiction de la consommation électrique d'une maison<br>
            <b>Random Forest</b> · UCI Appliances Energy 
        </p>
    </div>
    <hr style='margin-bottom:10px;'>
""", unsafe_allow_html=True)


# 4. SIDEBAR

with st.sidebar:
    st.markdown("##  Infos du modèle")
    st.markdown(f"""
| Paramètre | Valeur |
|---|---|
| Algorithme | Random Forest |
| Arbres | 100 |
| Features | {len(features)} |
| R² | ~0.59 |
| MAE | ~30 Wh |
| Dataset | UCI (19 735 obs.) |
    """)
    st.markdown("---")
    st.markdown("##  Comment utiliser")
    st.markdown("""
1. Réglez les curseurs
2. Cliquez **Prédire**
3. Lisez le résultat
    """)
    st.markdown("---")
    st.markdown("##  Contexte")
    st.markdown("""
Dans les pays à accès électrique limité comme **Madagascar**, prédire la consommation permet d'anticiper les pics et d'optimiser la distribution d'énergie.
    """)
    st.markdown("---")
    st.caption(" ML — Sujet 12 | Akim Ratiarison — INSI L2")

# 5. INPUTS

st.markdown("##  Paramètres de la maison")

col1, col2 = st.columns(2)
valeurs = {}

features_col1 = features[:5]
features_col2 = features[5:]

with col1:
    for feat in features_col1:
        mn, mx, defaut, step = BORNES[feat]
        label = LABELS[feat]
        if feat == 'hour':
            valeurs[feat] = st.slider(label, int(mn), int(mx), int(defaut), step=int(step))
        else:
            valeurs[feat] = st.slider(label, float(mn), float(mx), float(defaut), step=float(step))

with col2:
    for feat in features_col2:
        mn, mx, defaut, step = BORNES[feat]
        label = LABELS[feat]
        valeurs[feat] = st.slider(label, float(mn), float(mx), float(defaut), step=float(step))


# 6. PRÉDICTION

st.markdown("---")

if st.button(" Prédire la consommation", use_container_width=False):

    # Construire le vecteur d'entrée dans l'ordre exact des features
    entree = pd.DataFrame([[valeurs[f] for f in features]], columns=features)

    # Prédiction
    prediction = float(model.predict(entree)[0])
    prediction = max(10.0, prediction)

    # Niveau de consommation
    
    if prediction < 60:
        niveau  = "Faible"
        couleur = "#27ae60"
        emoji   = "🟢"
        message = "La maison est au repos — nuit ou absence prolongée."
    elif prediction < 150:
        niveau  = "Modérée"
        couleur = "#f39c12"
        emoji   = "🟡"
        message = "Activité domestique normale — quelques appareils en marche."
    elif prediction < 300:
        niveau  = "Élevée"
        couleur = "#e67e22"
        emoji   = "🟠"
        message = "Plusieurs appareils en fonctionnement simultané."
    else:
        niveau  = "Très élevée"
        couleur = "#e74c3c"
        emoji   = "🔴"
        message = "Pic de consommation — heure de pointe ou forte chaleur."

    # Animation
    st.snow()

    # Résultat principal
    st.markdown(f"""
    <div style='
        background:#1a1a2e;
        border-left:6px solid {couleur};
        border-radius:12px;
        padding:25px 35px;
        margin:15px 0;
    '>
        <div style='font-size:2.5em; font-weight:bold; color:{couleur};'>
            {emoji} {prediction:.1f} Wh
        </div>
        <div style='font-size:1.1em; margin:8px 0; color:white;'>
            Niveau : <b>{niveau}</b>
        </div>
        <div style='color:#aaa;'>{message}</div>
    </div>
    """, unsafe_allow_html=True)

    # Métriques complémentaires
    c1, c2, c3 = st.columns(3)
    c1.metric("Consommation prédite", f"{prediction:.1f} Wh")
    c2.metric("Moyenne dataset", f"{moyenne_cible:.1f} Wh",
              delta=f"{prediction - moyenne_cible:+.1f} Wh")
    c3.metric("Niveau", f"{emoji} {niveau}")

    # Tableau récap
    st.markdown("###  Paramètres utilisés")
    recap = pd.DataFrame({
        'Feature':    list(LABELS[f] for f in features),
        'Valeur':     [f"{valeurs[f]}" for f in features],
        'Importance': ['⭐⭐⭐⭐⭐' if f == 'hour' else
                       '⭐⭐⭐⭐'  if f in ['T3','Press_mm_hg','RH_3'] else
                       '⭐⭐⭐'   for f in features]
    })
    st.dataframe(recap, use_container_width=True, hide_index=True)


# 7. FOOTER

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.82em;'>
    ⚡ Conso Predikt  · Akim Ratiarison  L2<br>
    Modèle : Random Forest · Dataset : UCI Appliances Energy Prediction
</div>
""", unsafe_allow_html=True)