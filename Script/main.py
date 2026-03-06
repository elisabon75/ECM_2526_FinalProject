import os
import pandas as pd

# Importation de tes modules personnalisés
from data_preprocessing import load_and_clean_data, merge_datasets
from feature_engineering import engineer_features
from models import train_garch_model, train_random_forest, train_lstm_model

def main():
    print("=====================================================")
    print("🚀 LANCEMENT DU PIPELINE AUTOMATISÉ (DDEFI 2025)")
    print("=====================================================")

    # 1. Configuration des chemins vers tes fichiers dans le dossier Data
    # Adapte les noms de fichiers si nécessaire
    PATH_ELEC = "../Data/european_wholesale_electricity_price_data_daily.csv"
    PATH_GAZ = "../Data/dutch_ttf_natural_gas.csv"
    PATH_METEO = "../Data/meteo_france_2017_2024.json"

    try:
        # ÉTAPE 1 : Chargement et Nettoyage
        print("\n[1/4] NETTOYAGE DES DONNÉES...")
        df_e, df_g, df_m = load_and_clean_data(PATH_ELEC, PATH_GAZ, PATH_METEO)
        df_merged = merge_datasets(df_e, df_g, df_m)

        # ÉTAPE 2 : Feature Engineering
        print("\n[2/4] CALCUL DES RENDEMENTS ET VOLATILITÉ...")
        df_final = engineer_features(df_merged)

        # ÉTAPE 3 : Entraînement des Modèles
        print("\n[3/4] ENTRAÎNEMENT DES MODÈLES (GARCH, RF, LSTM)...")
        
        # Modèle 1 : GARCH
        garch_vol = train_garch_model(df_final)
        
        # Modèle 2 : Random Forest
        rf_model, rf_preds, rf_index = train_random_forest(df_final)
        
        # Modèle 3 : LSTM (Deep Learning)
        lstm_model, lstm_preds, lstm_index = train_lstm_model(df_final)

        # ÉTAPE 4 : Exportation des résultats pour le Dashboard
        print("\n[4/4] GÉNÉRATION DU FICHIER POUR LE VISUEL...")
        
        # On crée un DataFrame qui regroupe les prédictions des 3 modèles
        # (On aligne les prédictions sur les dates correspondantes)
        df_results = pd.DataFrame(index=rf_index)
        df_results['Real_Volatility'] = df_final.loc[rf_index, 'Elec_Volatility_30d']
        df_results['GARCH_Prediction'] = garch_vol.loc[rf_index]
        df_results['RF_Prediction'] = rf_preds
        df_results['LSTM_Prediction'] = lstm_preds

        output_path = "../Data/volatility_dashboard_data.csv"
        df_results.to_csv(output_path)
        
        print(f"✅ TERMINÉ ! Fichier créé : {output_path}")
        print("=====================================================")

    except Exception as e:
        print(f"\n❌ ERREUR LORS DE L'EXÉCUTION : {e}")

if __name__ == "__main__":
    main()
