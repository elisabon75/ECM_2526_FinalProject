import pandas as pd
import numpy as np
import json


def load_and_clean_data(chemin_elec, chemin_gaz, chemin_meteo):
    print("Extraction et nettoyage des données brutes...")

    # 1. ÉLECTRICITÉ (Daily)

    df_elec = pd.read_csv(chemin_elec)

    if 'Country' in df_elec.columns:
        df_elec = df_elec[df_elec['Country'] == 'France'].copy()

    # Nettoyage et dates
    df_elec = df_elec.infer_objects(copy=False)
    df_elec = df_elec.interpolate(method='linear', numeric_only=True)
    df_elec['Date'] = pd.to_datetime(df_elec['Date'])

    daily_avg_elec = df_elec.groupby('Date')['Price (EUR/MWhe)'].mean().reset_index()

    # 2. GAZ (TTF)

    df_gas = pd.read_csv(chemin_gaz)

    colonnes_prix = ['Dernier ((EUR/MWh)', 'Ouv.', ' Plus Haut', 'Plus Bas']
    for col in colonnes_prix:
        if col in df_gas.columns and df_gas[col].dtype == 'object':
            df_gas[col] = df_gas[col].str.replace(',', '.').astype(float)

    df_gas['Date'] = pd.to_datetime(df_gas['Date'], format='%d/%m/%Y')
    df_gas = df_gas.sort_values('Date')
    df_gas = df_gas.infer_objects(copy=False)
    df_gas = df_gas.interpolate(method='linear', numeric_only=True)

    # 3. MÉTÉO (JSON)

    with open(chemin_meteo, 'r', encoding='utf-8') as f:
        meteo_data = json.load(f)

    df_meteo = pd.DataFrame(meteo_data['daily'])

    if 'time' in df_meteo.columns:
        df_meteo = df_meteo.rename(columns={'time': 'Date'})

    df_meteo['Date'] = pd.to_datetime(df_meteo['Date'])

    return daily_avg_elec, df_gas, df_meteo


def merge_datasets(df_elec, df_gaz, df_meteo):
    print("Fusion des jeux de données...")

    df_merged = pd.merge(df_elec, df_gaz, on='Date', how='inner')
    df_final = pd.merge(df_merged, df_meteo, on='Date', how='inner')

    df_final = df_final.dropna().reset_index(drop=True)
    print("Taille du dataset final :", df_final.shape)

    return df_final
