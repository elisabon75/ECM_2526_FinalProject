import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Imports pour le Deep Learning (LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_garch_model(df):
    """
    Entraîne un modèle GARCH-X pour prédire la volatilité.
    """
    print("Entraînement du modèle GARCH-X...")
    returns = df['Elec_Log_Returns'] * 100 
    exog = df[['Gas_Log_Returns']] 
    
    am = arch_model(returns, x=exog, mean='ARX', lags=1, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    
    predicted_volatility = res.conditional_volatility / 100
    print("✅ Modèle GARCH entraîné.")
    
    return predicted_volatility


def train_random_forest(df):
    """
    Entraîne un modèle Random Forest.
    """
    print("Entraînement du modèle Random Forest...")
    target = 'Elec_Volatility_30d'
    features = [col for col in df.columns if col not in [target, 'Elec_Log_Returns', 'Date']]
    
    train_size = int(len(df) * 0.8)
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]
    
    X_train, y_train = train_data[features], train_data[target]
    X_test, y_test = test_data[features], test_data[target]
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    print(f"✅ Random Forest entraîné (RMSE: {rmse:.4f})")
    
    return rf_model, rf_predictions, test_data.index


def train_lstm_model(df):
    """
    Entraîne un réseau de neurones récurrents (LSTM).
    """
    print("Entraînement du modèle LSTM (Deep Learning)...")
    target = 'Elec_Volatility_30d'
    features = [col for col in df.columns if col not in [target, 'Elec_Log_Returns', 'Date']]
    
    train_size = int(len(df) * 0.8)
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]
    
    X_train, y_train = train_data[features].values, train_data[target].values
    X_test, y_test = test_data[features].values, test_data[target].values
    
    # 1. Mise à l'échelle (Crucial pour les réseaux de neurones)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Reshape en 3D pour le LSTM : [samples, time_steps, features]
    # Ici on utilise un time_step de 1 pour simplifier l'ingestion directe
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # 3. Création de l'architecture du réseau
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])))
    model.add(Dense(1)) # Couche de sortie (1 seule valeur à prédire : la volatilité)
    
    model.compile(optimizer='adam', loss='mse')
    
    # 4. Entraînement
    # verbose=0 permet de cacher la barre de progression pour ne pas polluer l'affichage final
    model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, verbose=0)
    
    # 5. Prédictions
    lstm_predictions = model.predict(X_test_reshaped, verbose=0).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))
    print(f"✅ LSTM entraîné (RMSE: {rmse:.4f})")
    
    return model, lstm_predictions, test_data.index
