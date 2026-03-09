<div align="center">

# ⚡ Energy Price Volatility Prediction
### ECM_2026_DDEFI | Final Project – Machine Learning & Data Science
**École Centrale Méditerranée**

<p align="center">
  <a href="https://www.kaggle.com/code/alexismoisdon/projet-data-du-meilleur-groupe">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" width="400" alt="Kaggle Badge">
  </a>
</p>

*Ce projet vise à anticiper les périodes d'instabilité sur les marchés de l'électricité et du gaz pour optimiser les décisions d'achat d'énergie.*

</div>

---

##  Contenu de ce Repository

Ce GitHub est organisé pour montrer le passage d'une phase de recherche (Notebook) à une solution industrialisée (Scripts).

###  1. Dossier `Data/` (Les Datafiles)
Ce dossier contient la base de notre projet : les données brutes et les résultats.
- **Données sources :** Prix EPEX Spot, cours du Gaz TTF et données Météo France.

###  2. Dossier `Notebooks/` (Recherche & Exploration)
C'est ici que se trouve notre **code Google Colab (`.ipynb`)**. 
- Il contient toute l'analyse exploratoire des données (EDA), les tests de modèles et les visualisations graphiques. C'est le laboratoire de recherche du projet.

### 3. Dossier `Script/` (Industrialisation & Automatisation)

C'est la partie la plus critique pour la mise en production. Nous avons transformé le code exploratoire du Notebook en une architecture de fichiers Python (`.py`) entièrement modulaires.

**Pourquoi ?** Pour automatiser de bout en bout le traitement de la donnée. Au lieu de relancer des cellules de code manuellement, l'exécution d'un seul script (`main.py`) orchestre l'ensemble du pipeline :
* Extraction et nettoyage des données brutes.
* Feature engineering (calcul des rendements, lags, volatilité réelle).
* Entraînement et prédiction des modèles (Économétrie, Machine Learning, Deep Learning) avec fixation des variables aléatoires pour garantir une reproductibilité stricte.
* Exportation unifiée des prédictions dans un format prêt à l'emploi.

**Avantage :** Cette architecture rend le projet robuste, reproductible à l'identique, et prêt à absorber de nouvelles données en temps réel. Elle intègre également une couche de restitution visuelle via un dashboard interactif (`dashboard.py`), prouvant ainsi la capacité du projet à s'insérer dans une application métier concrète et utilisable par un client final.

###  4. Dossier `Presentation/` (Gestion de Projet)
Ce dossier regroupe tous les documents stratégiques et théoriques du projet :
- **Cahier des Charges & Problématique :** Définition des objectifs SMART et du périmètre.
- **TD Product Management :** Analyse de la proposition de valeur ("Pourquoi le ML ?", "Quelle valeur pour le client ?").
- **Slides PowerPoint :** Le support de présentation finale pour notre soutenance.

###  Analyse Interactive sur Kaggle

Pour une lecture fluide et visuelle de notre étude, nous avons mis en place un Notebook Kaggle. Vous y trouverez l'intégralité de notre démarche exploratoire, les courbes d'apprentissage des modèles et les graphiques de prédiction interactifs.

**Cliquez sur le bouton ci-dessus pour accéder au Notebook :**

---

## Résumé du projet

###  Vue d'ensemble et Problématique

L’électricité ne peut pas être stockée à grande échelle. Les acteurs doivent jongler entre le marché **spot** (jour le jour) et les **contrats à terme**. Notre solution ne prédit pas le prix exact, mais la **volatilité** pour aider à choisir la meilleure stratégie :
- 🔴 **Forte volatilité anticipée :** Sécuriser un contrat à terme pour limiter les risques.
- 🟢 **Faible volatilité anticipée :** Acheter sur le marché spot pour optimiser les coûts.

###  Système d'Alerte de Volatilité

Nous convertissons les prédictions numériques en niveaux de risque actionnables via des détections de seuils (Z-score) et des algorithmes d'isolation (Isolation Forest).

| Statut | Niveau de Risque | Signification |
| :---: | :--- | :--- |
| 🟢 | **Stable** | Volatilité faible. Opportunité d'achat sur le marché spot. |
| 🟡 | **Modéré** | Incertitude moyenne sur le marché. |
| 🔴 | **Élevé** | Forte instabilité. Recommandation de couverture (hedging). |

###  Architecture du Système

Le projet suit un cycle de vie complet de Data Science, industrialisé via un pipeline automatisé :
1. `Collecte` ➔ 2. `Nettoyage` ➔ 3. `Feature Engineering` ➔ 4. `Entraînement` ➔ 5. `Prédiction` ➔ 6. `Génération d'Alertes` ➔ 7. `Dashboard`

### Project Team

* **[Elisa Bon](https://www.linkedin.com/in/elisa-bon-298651299/)** – Machine Learning Student
* **[Alexis Moisdon](https://www.linkedin.com/in/alexis-moisdon-b09062249/)** – Machine Learning Student
* **[Coralie Brouillet](https://www.linkedin.com/in/coralie-brouillet/)** – Machine Learning Student

### Détails Techniques & Modélisation

#### Feature Engineering
* **Variables Temporelles :** Lags (1j, 7j, 30j), statistiques glissantes (moyenne, écart-type).
* **Transformations Financières :** Rendements logarithmiques (Log-returns) et volatilité historique.
* **Réduction de dimension :** Analyse en Composantes Principales (PCA) pour le débruitage.

#### Modèles Comparés
* **Statistiques :** Naïve, Moyennes Mobiles, ARIMA, **GARCH (1,1)**.
* **Machine Learning :** Random Forest.
* **Deep Learning :** LSTM.

###  Stratégie de Validation & Performance

* **Découpage Chronologique :** Train (70%), Validation (15%), Test (15%).
* **Objectifs :** Amélioration de **15%** par rapport aux modèles baselines et précision des alertes supérieure à **70%**.
* **Backtesting :** Évaluation des gains réels (P&L) via une simulation historique.

###  Perspectives (Future Work)

- [ ] Intégration de modèles de séries temporelles basés sur les **Transformers**.
- [ ] Déploiement de pipelines de données en temps réel (Streaming).
- [ ] Utilisation de l'**Apprentissage par Renforcement** pour des stratégies de trading dynamique.

###  Installation et Utilisation Rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer le pipeline complet
python Script/main.py

# 3. Démarrer la visualisation interactive (Dashboard)
streamlit run Script/dashboard.py
```
### Teaching & Supervision

This project was developed as part of a **Machine Learning course** at École Centrale Méditerranéen.

**Course given by:** 🎓 **[Sitraka Matthieu FORLER](https://www.linkedin.com/in/sitraka-matthieu-forler/)** *Senior Data Scientist & AI Architect | Professor of Applied Machine Learning*


---
*Main repository: **ECM_2526_FinalProject***

