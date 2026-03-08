import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="Energy Volatility Dashboard", layout="wide")

st.title("⚡ Dashboard de Prédiction de la Volatilité Électrique")
st.markdown("Ce dashboard présente les résultats des modèles entraînés pour prédire la volatilité du marché français.")


# 1. Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("../Data/volatility_dashboard_data.csv", index_col=0, parse_dates=True)
    return df


try:
    df = load_data()

    # 2. Barre latérale : Sélection des modèles
    st.sidebar.header("Configuration")
    modeles_disponibles = [col for col in df.columns if col != 'Real_Volatility']
    selection = st.sidebar.multiselect(
        "Sélectionnez les modèles à afficher :",
        modeles_disponibles,
        default=modeles_disponibles[:3]  # On en affiche 3 par défaut
    )

    # 3. Graphique Interactif Principal
    st.subheader("📈 Comparaison des prédictions vs Réalité")
    fig = go.Figure()

    # Ajout de la réalité
    fig.add_trace(go.Scatter(x=df.index, y=df['Real_Volatility'], name="Réalité", line=dict(color='black', width=2)))

    # Ajout des modèles sélectionnés
    for m in selection:
        fig.add_trace(go.Scatter(x=df.index, y=df[m], name=m, opacity=0.7))

    fig.update_layout(hovermode="x unified", template="plotly_white", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # 4. Tableau des scores (ton fameux classement !)
    st.divider()
    st.subheader("🏆 Classement de Performance")

    # On recalcule rapidement les métriques pour l'affichage
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    scores = []
    for m in modeles_disponibles:
        mask = df[m].notna()
        mae = mean_absolute_error(df.loc[mask, 'Real_Volatility'], df.loc[mask, m])
        rmse = np.sqrt(mean_squared_error(df.loc[mask, 'Real_Volatility'], df.loc[mask, m]))
        scores.append({"Modèle": m, "MAE": mae, "RMSE": rmse})

    df_scores = pd.DataFrame(scores).sort_values("RMSE")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(df_scores.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}"}))
    with col2:
        st.info("Le modèle le plus performant sur cette période est le **" + df_scores.iloc[0]['Modèle'] + "**.")

except Exception as e:
    st.error(f"Erreur lors du chargement du fichier : {e}")
    st.info("Assure-vous que le fichier '../Data/volatility_dashboard_data.csv' existe bien.")
