<div align="center">

# ⚡ Energy Price Volatility Prediction
### ECM_2026_DDEFI | Final Project – Machine Learning & Data Science
**École Centrale Méditerranéenne**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-179A9A?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

*Anticipating market instability through Statistical Models, Machine Learning, and Deep Learning.*

</div>

---

## 📖 Project Overview

Energy markets have become increasingly volatile in recent years. Electricity and gas prices fluctuate rapidly due to weather conditions, geopolitical tensions, production constraints, and macroeconomic shocks.

> **Crucial context:** Unlike many commodities, **electricity cannot be stored economically at a large scale**. 

Market actors must constantly decide whether to buy on the **spot market**, secure **future contracts**, or increase their **hedging exposure**.

This project focuses on **predicting volatility rather than predicting exact prices**. Anticipating volatility helps decision-makers:
- 🛡️ Avoid extreme market instability
- 📉 Reduce financial risk
- 🛒 Optimize purchasing strategies and energy procurement timing



---

## 🎯 Problem Statement

Energy traders face a critical decision every day: **Should we buy electricity now on the spot market, or lock in a future contract?**

* 🔴 **If HIGH volatility is expected:** Securing a future contract can reduce risk.
* 🟢 **If LOW volatility is expected:** Buying on the spot market may be cheaper.

The primary objective of this project is to build a complete machine learning pipeline to **predict future volatility levels** and support these strategic, high-stakes decisions.

---

## 🚀 Objectives & Performance Targets

### Main Goals
* 🔄 **Build a robust data pipeline** integrating multiple temporal and external sources.
* 🔮 **Predict volatility** at multiple horizons: `1 day`, `1 week`, and `1 month`.
* 📊 **Benchmark models:** Compare Statistical, Machine Learning, and Deep Learning approaches.
* 🖥️ **Deliver a dashboard** that generates actionable volatility alerts.

### Performance Targets
* 📈 Improve prediction metrics by **15% vs. baseline models**.
* 🎯 Achieve **>70% precision** in volatility alerts.
* 🛡️ Demonstrate concrete **risk reduction via backtesting**.

---

## 🏗️ System Architecture

The project follows a full, end-to-end data science lifecycle. A visualization interface is planned via **Replit** and an API layer.

1. `Data Collection` ➔ 2. `Preprocessing` ➔ 3. `Feature Engineering` ➔ 4. `Model Training` ➔ 5. `Volatility Prediction` ➔ 6. `Alert Generation` ➔ 7. `Dashboard`



---

## 🗄️ Data Sources

To capture the complex dynamics of the energy market, we integrate multiple datasets:

| Category | Variables Included |
| :--- | :--- |
| **⚡ Energy Data** | Electricity prices (EPEX Spot), Gas prices, Oil prices |
| **🌦️ External Variables** | Weather data (temperature, humidity, anomalies), Renewable production |
| **📅 Calendar Variables**| Day of week, Month, Weekends, Public holidays |

---

## ⚙️ Feature Engineering

Time-series transformations are critical for exposing patterns to our models.

* **Temporal Features:** Lag variables (`1d`, `7d`, `30d`), Rolling statistics (mean, variance, standard deviation).
* **Financial Transformations:** Log returns, Historical volatility.
* **Dimensionality Reduction:** Principal Component Analysis (PCA) to manage feature bloat.

---

## 🧠 Modeling Approaches

We compare a wide variety of approaches, scaling from simple baselines to advanced neural networks.

### 1️⃣ Baseline & Statistical Models
* **Naive Model:** $price(t) = price(t-1)$
* **Moving Averages:** 7-day and 30-day windows
* **ARIMA**
* **GARCH (1,1):** Specifically for baseline volatility prediction.

### 2️⃣ Machine Learning
* **XGBoost**
* **Random Forest**

### 3️⃣ Deep Learning 
*Inspired by this [Kaggle Reference Notebook](https://www.kaggle.com/code/dimitriosroussis/electricity-price-forecasting-with-dnns-eda).*

* **LSTM** & **Stacked LSTM**
* **CNN** & **CNN-LSTM**
* **Time Distributed MLP**
* **Encoder-Decoder (Seq2Seq)**

> **Note:** All deep learning models utilize multivariate time series, approximately 25 previous time steps, and the Adam optimizer.

---

## ⚖️ Validation Strategy & Metrics

**Chronological Split:**
* **Train:** `70%` | **Validation:** `15%` | **Test:** `15%`
* *Additional evaluation uses time-series cross-validation with rolling windows.*

**Key Evaluation Metrics:**
* `RMSE`, `MAE`, `MSE` (focusing on volatility)
* Alert Precision
* Backtest P&L (Profit & Loss)

---

## 🚨 Volatility Alert System

Predicted volatility is converted into distinct **risk levels** using z-score detection and anomaly detection (Isolation Forest).

| Status | Risk Level | Meaning |
| :---: | :--- | :--- |
| 🟢 | **Stable** | Low expected volatility. Favorable for spot market purchasing. |
| 🟡 | **Moderate** | Medium market uncertainty. |
| 🔴 | **High** | Strong market instability. Consider future contracts/hedging. |

---

## ⚠️ Limitations

This system is designed as a **decision support tool**, not an autonomous agent. 
* ❌ It does **not** perform automated trading.
* ❌ It does **not** optimize full trading portfolios or entirely eliminate financial risk.
* 📉 System performance is heavily dependent on real-time data quality and extreme, unprecedented market shocks.

---

## 🔮 Future Work
- [ ] Integration of **Transformer-based** time-series models.
- [ ] Deployment of real-time streaming data pipelines.
- [ ] Application of **Reinforcement Learning** for dynamic trading strategies.
- [ ] Direct integration with professional energy trading APIs.

---

## 👨‍🏫 Teaching & Supervision

This project was developed as part of a **Machine Learning course** at École Centrale Méditerranéenne.

**Course given by:** 🎓 **[Sitraka Matthieu FORLER](https://www.linkedin.com/in/sitraka-matthieu-forler/)** *Senior Data Scientist & AI Architect | Professor of Applied Machine Learning*

## 👥 Project Team

* **[Elisa Bon](https://www.linkedin.com/in/elisa-bon-298651299/)** – Machine Learning Student
* **[Alexis Moisdon](https://www.linkedin.com/in/alexis-moisdon-b09062249/)** – Machine Learning Student
* **[Coralie Brouillet](https://www.linkedin.com/in/coralie-brouillet/)** – Machine Learning Student

---
*Main repository: **ECM_2526_FinalProject***

Course given by:



### [Sitraka Matthieu FORLER](https://www.linkedin.com/in/sitraka-matthieu-forler/)



Senior Data Scientist & AI Architect  

Professor of Applied Machine Learning
