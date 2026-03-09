import os
import pandas as pd
import numpy as np
import warnings

from data_preprocessing import load_and_clean_data, merge_datasets
# On n'importe PAS feature_engineering.py pour éviter le bug
from models import train_garch_model, train_arx_model, get_baseline_models, train_random_forest, train_lstm_model


def engineer_features(df):
    print("Création des variables (Feature Engineering)...")
    df = df.sort_values('Date').reset_index(drop=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df['Elec_Log_Returns'] = np.log(df['Price (EUR/MWhe)'] / df['Price (EUR/MWhe)'].shift(1))
        df['Gas_Log_Returns'] = np.log(df['Dernier ((EUR/MWh)'] / df['Dernier ((EUR/MWh)'].shift(1))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for i in range(1, 6):
        df[f'lag_ret_{i}'] = df['Elec_Log_Returns'].shift(i)

    df['day_of_week'] = df['Date'].dt.dayofweek
    df['Real_Volatility'] = df['Elec_Log_Returns'].abs()

    df_features = df.dropna().reset_index(drop=True)
    df_features = df_features.iloc[:1770]

    print(f"Feature Engineering terminé. {len(df_features)} lignes conservées.")
    return df_features



def run_all_pipeline():
    print("=====================================================")
    print("🚀 LANCEMENT DU PIPELINE AUTOMATISÉ (DDEFI 2025)")
    print("=====================================================")

    PATH_ELEC = "../Data/european_wholesale_electricity_price_data_daily.csv"
    PATH_GAZ = "../Data/dutch_ttf_natural_gas.csv"
    PATH_METEO = "../Data/meteo_france_2017_2024.json"

    try:
        print("\n[1/4] NETTOYAGE DES DONNÉES...")
        df_e, df_g, df_m = load_and_clean_data(PATH_ELEC, PATH_GAZ, PATH_METEO)
        df_merged = merge_datasets(df_e, df_g, df_m)

        print("\n[2/4] CALCUL DES RENDEMENTS ET VOLATILITÉ...")
        df_final = engineer_features(df_merged)

        print("\n[3/4] ENTRAÎNEMENT DES MODÈLES (TOUS)...")
        # On fait tourner TOUS tes modèles
        garch_mod1 = train_garch_model(df_final)
        arx_mod3 = train_arx_model(df_final)
        naive_preds, ma7_preds = get_baseline_models(df_final)
        rf_model, rf_preds, rf_index = train_random_forest(df_final)
        lstm_model, lstm_preds, lstm_index = train_lstm_model(df_final)

        print("\n[4/4] GÉNÉRATION DU FICHIER POUR LE DASHBOARD...")
        df_results = pd.DataFrame(index=rf_index)


        df_results['Real_Volatility'] = df_final.loc[rf_index, 'Real_Volatility']
        df_results['GARCH_X_Prediction'] = arx_mod3.loc[rf_index]
        df_results['Random_Forest_Prediction'] = rf_preds

        lstm_series = pd.Series(lstm_preds, index=lstm_index)
        df_results['LSTM_Prediction'] = lstm_series


        df_results['GARCH_Mod1_Prediction'] = garch_mod1.loc[rf_index]
        df_results['Naive_Prediction'] = naive_preds.loc[rf_index]
        df_results['MA7_Prediction'] = ma7_preds.loc[rf_index]

        # On transforme l'index en une vraie colonne Date
        df_results = df_results.reset_index(names="Date")

        # Sauvegarde du CSV
        output_path = "../Data/volatility_dashboard_data.csv"
        df_results.to_csv(output_path, index=False)

        print(f"✅ TERMINÉ ! Fichier CSV parfaitement formaté créé : {output_path}")
        print("=====================================================")

        return df_results

    except Exception as e:
        print(f"\n❌ ERREUR LORS DE L'EXÉCUTION : {e}")
        raise e


if __name__ == "__main__":
    run_all_pipeline()
