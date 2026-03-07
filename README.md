<div align="center">

# ⚡ Energy Price Volatility Prediction
### ECM_2026_DDEFI | Final Project – Machine Learning & Data Science
**École Centrale Méditerranéen**

<p align="center">
  <a href="https://www.kaggle.com/code/alexismoisdon/projet-data-du-meilleur-groupe">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" width="400" alt="Kaggle Badge">
  </a>
</p>

*Ce projet vise à anticiper les périodes d'instabilité sur les marchés de l'électricité et du gaz pour optimiser les décisions d'achat d'énergie.*

</div>

##  Vue d'ensemble et Problématique

L’électricité ne peut pas être stockée à grande échelle. Les acteurs doivent jongler entre le marché **spot** (jour le jour) et les **contrats à terme**. Notre solution ne prédit pas le prix exact, mais la **volatilité** pour aider à choisir la meilleure stratégie :
- 🔴 **Forte volatilité anticipée :** Sécuriser un contrat à terme pour limiter les risques.
- 🟢 **Faible volatilité anticipée :** Acheter sur le marché spot pour optimiser les coûts.

##  Système d'Alerte de Volatilité

Nous convertissons les prédictions numériques en niveaux de risque actionnables via des détections de seuils (Z-score) et des algorithmes d'isolation (Isolation Forest).

| Statut | Niveau de Risque | Signification |
| :---: | :--- | :--- |
| 🟢 | **Stable** | Volatilité faible. Opportunité d'achat sur le marché spot. |
| 🟡 | **Modéré** | Incertitude moyenne sur le marché. |
| 🔴 | **Élevé** | Forte instabilité. Recommandation de couverture (hedging). |

##  Architecture du Système

Le projet suit un cycle de vie complet de Data Science, industrialisé via un pipeline automatisé :
1. `Collecte` ➔ 2. `Nettoyage` ➔ 3. `Feature Engineering` ➔ 4. `Entraînement` ➔ 5. `Prédiction` ➔ 6. `Génération d'Alertes` ➔ 7. `Dashboard`

## Project Team

* **[Elisa Bon](https://www.linkedin.com/in/elisa-bon-298651299/)** – Machine Learning Student
* **[Alexis Moisdon](https://www.linkedin.com/in/alexis-moisdon-b09062249/)** – Machine Learning Student
* **[Coralie Brouillet](https://www.linkedin.com/in/coralie-brouillet/)** – Machine Learning Student

## Détails Techniques & Modélisation

### Feature Engineering
* **Variables Temporelles :** Lags (1j, 7j, 30j), statistiques glissantes (moyenne, écart-type).
* **Transformations Financières :** Rendements logarithmiques (Log-returns) et volatilité historique.
* **Réduction de dimension :** Analyse en Composantes Principales (PCA) pour le débruitage.

### Modèles Comparés
* **Statistiques :** Naïve, Moyennes Mobiles, ARIMA, **GARCH (1,1)**.
* **Machine Learning :** XGBoost, Random Forest.
* **Deep Learning :** LSTM, Stacked LSTM, CNN, CNN-LSTM, Encoder-Decoder.

##  Stratégie de Validation & Performance

* **Découpage Chronologique :** Train (70%), Validation (15%), Test (15%).
* **Objectifs :** Amélioration de **15%** par rapport aux modèles baselines et précision des alertes supérieure à **70%**.
* **Backtesting :** Évaluation des gains réels (P&L) via une simulation historique.

##  Perspectives (Future Work)

- [ ] Intégration de modèles de séries temporelles basés sur les **Transformers**.
- [ ] Déploiement de pipelines de données en temps réel (Streaming).
- [ ] Utilisation de l'**Apprentissage par Renforcement** pour des stratégies de trading dynamique.

##  Installation et Utilisation Rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer le pipeline complet
python Script/main.py
```

## Teaching & Supervision

This project was developed as part of a **Machine Learning course** at École Centrale Méditerranéen.

**Course given by:** 🎓 **[Sitraka Matthieu FORLER](https://www.linkedin.com/in/sitraka-matthieu-forler/)** *Senior Data Scientist & AI Architect | Professor of Applied Machine Learning*


---
*Main repository: **ECM_2526_FinalProject***

