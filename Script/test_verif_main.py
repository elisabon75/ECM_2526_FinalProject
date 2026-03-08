import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

chemin_fichier = "../Data/volatility_dashboard_data.csv"

try:
    df = pd.read_csv(chemin_fichier)

    if 'Real_Volatility' in df.columns:
        y_true = df['Real_Volatility']
        resultats = []

        # Le CSV ne contient DÉJÀ QUE le jeu de test, on boucle sur chaque modèle
        for col in df.columns:
            if col not in ['Date', 'Real_Volatility'] and not col.startswith('Unnamed'):

                # Reproduction exacte de la logique de calcul de ton Notebook
                y_p = df[col]

                # On crée un masque pour supprimer les NaN communs (Exactement comme dans ton Colab)
                mask = ~np.isnan(y_p) & ~np.isnan(y_true)

                y_t_final = y_true[mask]
                y_p_final = y_p[mask]

                if len(y_t_final) > 0:
                    mae = mean_absolute_error(y_t_final, y_p_final)
                    rmse = np.sqrt(mean_squared_error(y_t_final, y_p_final))
                    resultats.append({'Modèle': col, 'MAE': mae, 'RMSE': rmse})

        if resultats:
            # On trie pour avoir le même ordre d'affichage que le Kaggle
            df_results = pd.DataFrame(resultats).sort_values(by='RMSE').reset_index(drop=True)

            print("\n--- CLASSEMENT FINAL TOUS MODÈLES (IDENTIQUE KAGGLE) ---")
            df_results['MAE'] = df_results['MAE'].apply(lambda x: f"{x:.6f}")
            df_results['RMSE'] = df_results['RMSE'].apply(lambda x: f"{x:.6f}")
            print(df_results.to_string(index=False, justify='right'))
            print("------------------------------------------------------")

except Exception as e:
    print(f"❌ Erreur : {e}")
