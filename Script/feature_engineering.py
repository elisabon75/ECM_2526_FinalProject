import pandas as pd
import numpy as np
import warnings


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
    print(f"Feature Engineering terminé. {len(df_features)} lignes conservées.")

    df_features = df.dropna().reset_index(drop=True)

    # Dans  Kaggle, la séparation test_size=0.2 donnait 354 jours de test.
    # Cela signifie que le dataset complet faisait exactement 1770 lignes (354 * 5).
    # On force donc le dataset à s'arrêter exactement à 1770 lignes pour couper
    # les jours "nouveaux" qui faussent les moyennes
    df_features = df_features.iloc[:1770]

    print(f"Feature Engineering terminé. {len(df_features)} lignes conservées.")
    return df_features
