# ECM_2526
# Energy Price Volatility Prediction  
### Final Project – Machine Learning & Data Science  
École Centrale Méditerranéenne

---

# Project Overview

Energy markets have become increasingly volatile in recent years. Electricity and gas prices fluctuate due to weather conditions, geopolitical tensions, production constraints, and macroeconomic shocks.

Unlike many commodities, **electricity cannot be stored economically at large scale**. Market actors must therefore constantly decide whether to:

- buy on the **spot market**
- secure **future contracts**
- increase their **hedging exposure**

This project focuses on **predicting volatility rather than predicting exact prices**.

Anticipating volatility helps decision-makers:

- avoid extreme market instability
- optimize purchasing strategies
- reduce financial risk
- improve energy procurement timing

The project combines **statistical models, machine learning, and deep learning** to build a volatility forecasting system and a decision-support dashboard.

---

# Teaching & Supervision

This project was developed as part of a **Machine Learning course at École Centrale Méditerranéenne**.


---

# Project Team

### Students

### [Elisa Bon](https://www.linkedin.com/in/elisa-bon-298651299/)
Machine Learning Student – École Centrale Méditerranéenne

### [Alexis Moisdon](https://www.linkedin.com/in/alexis-moisdon-b09062249/)
Machine Learning Student – École Centrale Méditerranéenne

### [Coralie Brouillet](https://www.linkedin.com/in/coralie-brouillet/)
Machine Learning Student – École Centrale Méditerranéenne

---

# Repository

Main project repository:

**ECM_2526_FinalProject**

Reference notebook used as inspiration for deep learning models:

https://www.kaggle.com/code/dimitriosroussis/electricity-price-forecasting-with-dnns-eda

---

# Problem Statement

Energy traders face a critical decision every day:

**Should we buy electricity now on the spot market or lock in a future contract?**

If **high volatility** is expected:
- securing a **future contract** can reduce risk.

If **low volatility** is expected:
- buying on the **spot market** may be cheaper.

The objective of the project is therefore to **predict future volatility levels** to support these strategic decisions.

---

# Objectives

The goal is to build a **complete machine learning pipeline** capable of forecasting volatility.

### Main goals

- Build a **data pipeline** integrating multiple sources
- Predict volatility at several horizons:
  - 1 day
  - 1 week
  - 1 month
- Compare **statistical models, ML models and deep learning models**
- Deliver a **dashboard generating volatility alerts**

### Performance targets

- Improve prediction metrics by **15% vs baseline**
- Achieve **>70% precision in volatility alerts**
- Demonstrate **risk reduction via backtesting**

---

# Data Sources

Multiple data sources are integrated.

### Energy data

- Electricity prices (EPEX Spot)
- Gas prices
- Oil prices

### External variables

- Weather data  
  temperature  
  humidity  
  anomalies

- Renewable production

### Calendar variables

- day of week
- month
- weekends
- public holidays

---

# Feature Engineering

Time-series transformations are applied.

### Temporal features

- Lag variables  
  1 day  
  7 days  
  30 days

- Rolling statistics  
  rolling mean  
  rolling variance  
  rolling standard deviation

### Financial transformations

- Log returns
- Historical volatility

### Dimensionality reduction

- PCA (Principal Component Analysis)

---

# Models

Several modeling approaches are compared.

## Baseline Models

- Naive model  
  price(t) = price(t−1)

- Moving averages  
  7 days  
  30 days

- ARIMA

---

## Statistical Models

- **GARCH (1,1)**  
for volatility prediction

---

## Machine Learning Models

- **XGBoost**
- **Random Forest**

---

## Deep Learning Models

Inspired by the Kaggle notebook.

- LSTM
- Stacked LSTM
- CNN
- CNN-LSTM
- Time Distributed MLP
- Encoder-Decoder (Seq2Seq)

All deep learning models use:

- multivariate time series
- ~25 previous time steps
- Adam optimizer

---

# Validation Strategy

Chronological split:

- Train: 70%
- Validation: 15%
- Test: 15%

Additional evaluation:

- Time-series cross validation with rolling windows

### Metrics

- RMSE
- MAE
- MSE (volatility)
- Alert precision
- Backtest P&L

---

# Volatility Alert System

Predicted volatility is converted into **risk levels**.

| Risk Level | Meaning |
|------------|--------|
| Stable | Low expected volatility |
| Moderate | Medium market uncertainty |
| High | Strong market instability |

Alerts are generated using:

- predicted volatility thresholds
- z-score detection
- anomaly detection (Isolation Forest)

---

# System Architecture

The project follows a full **data science lifecycle**:

1. Data collection  
2. Data preprocessing  
3. Feature engineering  
4. Model training  
5. Volatility prediction  
6. Alert generation  
7. Visualization dashboard

A visualization interface is planned via **Replit** and an API layer.

---

# Limitations

The system is a **decision support tool**.

It does not:

- perform automated trading
- optimize full trading portfolios
- eliminate financial risk

Performance depends on:

- data quality
- market conditions
- model assumptions

---

# Future Work

Possible extensions:

- Transformer-based time-series models
- Real-time data pipelines
- Reinforcement learning for trading strategies
- Integration with professional energy trading platforms

---

# Conclusion

This project shows how **machine learning can help anticipate market instability** in energy markets.

Instead of predicting exact prices, the system focuses on **volatility prediction**, which is often more useful for **risk management and strategic energy purchasing decisions**.


Course given by:

### [Sitraka Matthieu FORLER](https://www.linkedin.com/in/sitraka-matthieu-forler/)

Senior Data Scientist & AI Architect  
Professor of Applied Machine Learning
