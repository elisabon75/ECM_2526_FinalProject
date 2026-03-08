import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


def train_garch_model(df):
    print("Entraînement du modèle GARCH (Mod1)...")
    returns = df['Elec_Log_Returns']
    am = arch_model(returns, mean='Constant', vol='Garch', p=1, q=1, dist='normal', rescale=False)
    res = am.fit(disp='off')
    return res.conditional_volatility


def train_arx_model(df):
    print("Entraînement du modèle ARX-GARCH (Mod3)...")

    # On s'assure que les données sont propres
    # Dans le Notebook, l'ARX est souvent très sensible à l'échelle
    returns = df['Elec_Log_Returns']


    # On essaie d'utiliser le rendement du Gaz décalé d'un jour (J-1) car c'est souvent ainsi que les modèles ARX sont codés en recherche
    exog = df[['Gas_Log_Returns']].shift(1).fillna(0)

    try:
        am = arch_model(
            returns,
            x=exog,
            mean='ARX',
            lags=1,
            vol='Garch',
            p=1,
            q=1,
            dist='normal',
            rescale=False
        )
        res = am.fit(disp='off')
        return res.conditional_volatility
    except Exception as e:
        print(f" Erreur ARX, tentative sans shift : {e}")
        # Si le shift ne marche pas, on revient à la version brute
        am = arch_model(returns, x=df[['Gas_Log_Returns']], mean='ARX', lags=1, vol='Garch', p=1, q=1, rescale=False)
        res = am.fit(disp='off')
        return res.conditional_volatility


def get_baseline_models(df):
    print("Calcul des modèles de référence (Naïf et MA7)...")
    naive_preds = df['Real_Volatility'].shift(1)
    ma7_preds = df['Real_Volatility'].rolling(window=7).mean()
    return naive_preds, ma7_preds


def train_random_forest(df):
    print("Entraînement du modèle Random Forest...")
    exog_vars = ['temperature_2m_mean', 'precipitation_sum', 'windspeed_10m_mean', 'Gas_Log_Returns']
    lags = [f'lag_ret_{i}' for i in range(1, 6)]
    features = lags + exog_vars + ['day_of_week']
    target = 'Real_Volatility'

    split = int(len(df) * 0.8)
    X_train, X_test = df[features].iloc[:split], df[features].iloc[split:]
    y_train, y_test = df[target].iloc[:split], df[target].iloc[split:]

    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    return rf_model, rf_predictions, y_test.index


def train_lstm_model(df):
    print("Entraînement du modèle LSTM...")
    exog_vars = ['temperature_2m_mean', 'precipitation_sum', 'windspeed_10m_mean', 'Gas_Log_Returns']
    lags = [f'lag_ret_{i}' for i in range(1, 6)]
    features = lags + exog_vars + ['day_of_week']
    target = 'Real_Volatility'

    X = df[features]
    y = df[target]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    def create_sequences(X, y, lookback=7):
        Xs, ys = [], []
        for i in range(len(X) - lookback):
            Xs.append(X[i:(i + lookback)])
            ys.append(y[i + lookback])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback=7)

    split = int(len(X_seq) * 0.8)
    X_train_dl, X_test_dl = X_seq[:split], X_seq[split:]
    y_train_dl, y_test_dl = y_seq[:split], y_seq[split:]

    model = Sequential([
        Input(shape=(7, X_train_dl.shape[2])),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_dl, y_train_dl, epochs=50, batch_size=16, verbose=0)

    preds_scaled = model.predict(X_test_dl, verbose=0)
    preds = scaler_y.inverse_transform(preds_scaled).flatten()

    test_index = df.index[7 + split: 7 + split + len(X_test_dl)]
    return model, preds, test_index
