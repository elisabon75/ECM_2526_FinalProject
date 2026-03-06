import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Prend le dataframe fusionné et calcule les nouvelles variables :
    rendements, volatilité historique, et lags.
    """
    print("Création des variables (Feature Engineering)...")
    
    # On s'assure que les données sont bien triées chronologiquement
    df = df.sort_values('Date').reset_index(drop=True)
    
    # --- 1. Rendements Logarithmiques ---
    # Le logarithme permet de lisser les variations de prix
    # Note : Vérifie bien que 'Price (EUR/MWhe)' et 'Dernier ((EUR/MWh)' sont les bons noms de colonnes
    df['Elec_Log_Returns'] = np.log(df['Price (EUR/MWhe)'] / df['Price (EUR/MWhe)'].shift(1))
    df['Gas_Log_Returns'] = np.log(df['Dernier ((EUR/MWh)'] / df['Dernier ((EUR/MWh)'].shift(1))
    
    # --- 2. Volatilité Historique (Cible proxy) ---
    # On calcule l'écart-type (standard deviation) sur une fenêtre glissante (ex: 30 jours)
    fenetre = 30
    df['Elec_Volatility_30d'] = df['Elec_Log_Returns'].rolling(window=fenetre).std()
    
    # --- 3. Création des Lags (Mémoire du marché) ---
    # On donne au modèle les rendements des 1, 2 et 3 jours précédents
    for i in range(1, 4):
        df[f'Elec_Log_Returns_Lag_{i}'] = df['Elec_Log_Returns'].shift(i)
        df[f'Gas_Log_Returns_Lag_{i}'] = df['Gas_Log_Returns'].shift(i)
        
    # --- 4. Nettoyage final ---
    # Les fonctions shift() et rolling() ont inévitablement créé des NaN au tout début du dataset.
    # On doit les supprimer pour que le Machine Learning fonctionne.
    df_features = df.dropna().reset_index(drop=True)
    
    print(f"Feature Engineering terminé. {len(df_features)} lignes conservées.")
    
    return df_features
