import pandas as pd
import numpy as np

def load_and_clean_data(chemin_elec, chemin_gaz, chemin_meteo):
    """
    Charge les données d'Électricité, de Gaz et de Météo, 
    gère les valeurs manquantes et formate les dates.
    """
    print("Extraction et nettoyage des données brutes...")
    
    # 1. ÉLECTRICITÉ (Daily)
    # Dans ton Colab : import du CSV, tri, interpolation et formatage de la date
    df_elec = pd.read_csv(chemin_elec)
    df_elec = df_elec.sort_index()
    
    # Remplir les trous avec interpolation linéaire (comme dans ton EDA)
    df_elec = df_elec.interpolate(method='linear')
    df_elec['Date'] = pd.to_datetime(df_elec['Date'])
    
    # Calcul de la moyenne journalière (cellule 14 de ton notebook)
    daily_avg_elec = df_elec.groupby('Date')['Price (EUR/MWhe)'].mean().reset_index()
    
    # 2. GAZ (TTF)
    df_gas = pd.read_csv(chemin_gaz)
    
    # Nettoyage des virgules en points (si nécessaire) et conversion en nombres
    colonnes_prix = ['Dernier ((EUR/MWh)', 'Ouv.', ' Plus Haut', 'Plus Bas']
    for col in colonnes_prix:
        if df_gas[col].dtype == 'object':
            df_gas[col] = df_gas[col].str.replace(',', '.').astype(float)
            
    df_gas['Date'] = pd.to_datetime(df_gas['Date'], format='%d/%m/%Y')
    df_gas = df_gas.sort_values('Date').interpolate(method='linear')
    
    # 3. MÉTÉO (JSON)
    # Extraction des données journalières (cellule 7 de ton notebook)
    df_meteo_raw = pd.read_json(chemin_meteo)
    
    # Reformatage du dictionnaire imbriqué en DataFrame
    if isinstance(df_meteo_raw["daily"].iloc[0], list):
        df_meteo = pd.DataFrame(df_meteo_raw["daily"].tolist())
    else:
        df_meteo = pd.DataFrame(df_meteo_raw["daily"].to_dict())
    
    # Renommer "time" en "Date" pour que tous les fichiers aient la même clé de fusion
    if 'time' in df_meteo.columns:
        df_meteo['Date'] = pd.to_datetime(df_meteo['time'])
        df_meteo = df_meteo.drop('time', axis=1)
    
    return daily_avg_elec, df_gas, df_meteo


def merge_datasets(df_elec, df_gaz, df_meteo):
    """
    Fusionne les 3 jeux de données sur la colonne 'Date'.
    """
    print("Fusion des jeux de données...")
    
    # Fusionner Électricité et Gaz
    df_merged = pd.merge(df_elec, df_gaz, on='Date', how='inner')
    
    # Ajouter la Météo
    df_final = pd.merge(df_merged, df_meteo, on='Date', how='inner')
    
    # Supprimer les éventuelles premières/dernières lignes avec des NaN 
    df_final = df_final.dropna().reset_index(drop=True)
    
    return df_final
