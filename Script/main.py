import os
import pandas as pd

from data_preprocessing import load_and_clean_data, merge_datasets
from feature_engineering import engineer_features
from models import train_garch_model, train_arx_model, get_baseline_models, train_random_forest, train_lstm_model


def main():
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
        # 1. GARCH (Mod1)
        garch_mod1 = train_garch_model(df_final)
        # 2. ARX (Mod3)
        arx_mod3 = train_arx_model(df_final)
        # 3. Baselines (Naive & MA7)
        naive_preds, ma7_preds = get_baseline_models(df_final)
        # 4. Random Forest
        rf_model, rf_preds, rf_index = train_random_forest(df_final)
        # 5. LSTM
        lstm_model, lstm_preds, lstm_index = train_lstm_model(df_final)

        print("\n[4/4] GÉNÉRATION DU FICHIER POUR LE VISUEL...")
        df_results = pd.DataFrame(index=rf_index)

        # On sauvegarde la cible et les 6 modèles avec les noms exacts
        df_results['Real_Volatility'] = df_final.loc[rf_index, 'Real_Volatility']
        df_results['MA7 (7j)'] = ma7_preds.loc[rf_index]
        df_results['ARX (Mod3)'] = arx_mod3.loc[rf_index]
        df_results['Random Forest'] = rf_preds
        df_results['GARCH (Mod1)'] = garch_mod1.loc[rf_index]
        df_results['Naive (J-1)'] = naive_preds.loc[rf_index]

        # LSTM alignement
        lstm_series = pd.Series(lstm_preds, index=lstm_index)
        df_results['LSTM (Deep Learning)'] = lstm_series

        output_path = "../Data/volatility_dashboard_data.csv"
        df_results.to_csv(output_path)

        print(f"✅ TERMINÉ ! Fichier complet créé : {output_path}")
        print("=====================================================")

    except Exception as e:
        print(f"\n❌ ERREUR LORS DE L'EXÉCUTION : {e}")


if __name__ == "__main__":
    main()
