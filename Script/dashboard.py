import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import subprocess
import os
from pathlib import Path

st.set_page_config(page_title="Volatility Forecast Dashboard", layout="wide")

st.title("Prédiction de la volatilité de l'électricité")
st.markdown("Projet de Elisa Bon, Coralie Brouillet et Alexis Moisdon")
st.markdown("""
## Contexte
Ce dashboard pilote **en direct** notre pipeline d'Intelligence Artificielle.
Vous pouvez relancer l'entraînement des modèles depuis le menu latéral pour mettre à jour les prédictions.
""")
st.divider()

# ====================
# ORCHESTRATION DU PIPELINE
# ====================
CSV_PATH = "../Data/volatility_dashboard_data.csv"


def run_pipeline_safely():
    """Lance le main.py dans un processus séparé pour éviter les crashs TensorFlow/Streamlit"""
    try:
        # Exécute la commande python main.py en arrière-plan
        subprocess.run(["python", "main.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Le pipeline a échoué. Vérifiez le terminal.")
        return False


@st.cache_data(show_spinner=False)
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ====================
# FONCTIONS MÉTRIQUES & GRAPHIQUES
# ====================
def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = y_true.astype(float).to_numpy()
    y_pred = y_pred.astype(float).to_numpy()
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    eps = 1e-12
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def ranking_table(df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    rows = []
    for m in model_cols:
        met = metrics(df["Real_Volatility"], df[m])
        rows.append({"Model": m, "MAE": met["MAE"], "RMSE": met["RMSE"], "MAPE (%)": met["MAPE (%)"]})
    return pd.DataFrame(rows)


def rolling_rmse(df: pd.DataFrame, model_col: str, window: int) -> pd.Series:
    err2 = (df["Real_Volatility"] - df[model_col]) ** 2
    return np.sqrt(err2.rolling(window=window).mean())


def line_chart(df: pd.DataFrame, models: list[str], pretty_dict: dict, key: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Real_Volatility"], mode="lines", name="Vraie Volatilité",
                             line=dict(color='black', width=2)))
    for col in models:
        fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=pretty_dict[col]))
    fig.update_layout(title="Volatilité : Réelle vs Prédictions", xaxis_title="Date", yaxis_title="Volatilité",
                      hovermode="x unified", height=520)
    st.plotly_chart(fig, use_container_width=True, key=key)


# ====================
# INTERFACE UTILISATEUR
# ====================
with st.sidebar:
    st.header("⚙ Paramètres")

    # BOUTON POUR RELANCER LE MAIN.PY DEPUIS STREAMLIT
    if st.button("🚀 Relancer l'IA (Mettre à jour)"):
        with st.spinner("Entraînement des modèles en cours... Merci de patienter."):
            success = run_pipeline_safely()
        if success:
            st.success("✅ Modèles entraînés avec succès !")
            st.cache_data.clear()  # On vide le cache pour forcer la lecture du nouveau CSV

# On vérifie si les données existent déjà
if not os.path.exists(CSV_PATH):
    st.warning(
        "⚠️ Aucun résultat trouvé. Cliquez sur '🚀 Relancer l'IA' dans le menu de gauche pour générer les données.")
    st.stop()

# Chargement sécurisé des données
try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error("Erreur lors du chargement des résultats.")
    st.stop()

pretty_names = {
    "Random_Forest_Prediction": "Random Forest",
    "LSTM_Prediction": "LSTM (Deep Learning)",
    "GARCH_X_Prediction": "ARX-GARCH (Mod3)",
    "GARCH_Mod1_Prediction": "GARCH (Mod1)",
    "MA7_Prediction": "Moyenne Mobile (7j)",
    "Naive_Prediction": "Modèle Naïf (J-1)"
}
all_models = list(pretty_names.keys())

with st.sidebar:
    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
    start_date, end_date = st.date_input("Période", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    default_selected = ["Random_Forest_Prediction", "GARCH_X_Prediction", "LSTM_Prediction"]
    selected_models = st.multiselect("Modèles à afficher", options=all_models, default=default_selected,
                                     format_func=lambda x: pretty_names[x])
    rolling_window = st.slider("Fenêtre d'erreur (jours)", min_value=7, max_value=120, value=30, step=1)

mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
dff = df.loc[mask].copy()

if not selected_models:
    st.warning("Veuillez sélectionner au moins un modèle.")
    st.stop()

st.subheader("Indicateurs de performance")
kpi_cols = st.columns(len(selected_models))

for i, m in enumerate(selected_models):
    met = metrics(dff["Real_Volatility"], dff[m])
    with kpi_cols[i]:
        st.markdown(f"<h4 style='text-align:center'>{pretty_names[m]}</h4>", unsafe_allow_html=True)
        st.metric("MAE", f"{met['MAE']:.4f}")
        st.metric("RMSE", f"{met['RMSE']:.4f}")

st.subheader("📊 Courbes comparatives")
line_chart(dff, selected_models, pretty_names, key="main_lines")

st.subheader("🏆 Comparaison Globale")
rank_df = ranking_table(dff, all_models).copy()
rank_df["Model_Name"] = rank_df["Model"].map(pretty_names)
crit = st.selectbox("Critère de classement", ["RMSE", "MAE", "MAPE (%)"], index=0)
rank_df_sorted = rank_df.sort_values(crit, ascending=True).reset_index(drop=True)

st.success(f"✅ Meilleur modèle sur la période filtrée : **{rank_df_sorted.loc[0, 'Model_Name']}**")

fig_bar = go.Figure()
for metric_name in ["MAE", "RMSE"]:
    fig_bar.add_trace(go.Bar(x=rank_df_sorted["Model_Name"], y=rank_df_sorted[metric_name], name=metric_name))
fig_bar.update_layout(title="Métriques par modèle", barmode="group", height=420)
st.plotly_chart(fig_bar, use_container_width=True)
