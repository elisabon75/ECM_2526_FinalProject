# CAHIER DES CHARGES

Cahier des charges dans ce document : 
https://docs.google.com/document/d/11R6FXaHx754gDY3hBgc99tFLFk6EPxcaFEOVWOAqtkc/edit?usp=sharing

**Projet :** Prédiction de la volatilité des prix de l’énergie pour la prise de décision d’achat  
**Équipe :** Alexis MOISDON, Coralie BROUILLET, Elisa BON  
**Ressources :** [Lien d'inspiration Kaggle](https://www.kaggle.com/code/dimitriosroussis/electricity-price-forecasting-with-dnns-eda) | **GitHub :** `ECM_2526_FinalProject`

## I – Contexte

L’électricité et le gaz sont devenus, au cours des dernières décennies, des commodités échangées sur des marchés libres, soumis à de fortes fluctuations. La dérégulation, combinée aux aléas météorologiques et géopolitiques (tensions sur le gaz, indisponibilités nucléaires, vagues de froid, canicules), rend les prix particulièrement instables.

L’un des défis majeurs du secteur est que l’électricité ne peut pas être stockée de manière économique à grande échelle. Les acteurs (fournisseurs, traders, industriels, investisseurs) doivent donc acheter l’énergie au bon moment, en jonglant entre :
* Le marché **spot** (achat au jour le jour).
* Les **contrats à terme** (prix fixé pour une date future).
* La **couverture (hedging)** pour limiter les risques.

Dans ce contexte, anticiper la volatilité future devient un avantage compétitif majeur : non pas pour prédire le prix exact, mais pour identifier les périodes où le marché va être instable, afin d'ajuster les décisions d’achat :
>  **Si forte volatilité anticipée :** sécuriser un contrat à terme.  
>  **Si faible volatilité :** acheter sur le spot, souvent moins cher.

## II – Problématique

Comment prévoir les périodes de forte volatilité des prix de l’électricité et du gaz, à court et moyen terme, pour permettre à un trader de :
1. Choisir intelligemment entre spot et marché à terme.
2. Ajuster son niveau de couverture.
3. Éviter les périodes de risque extrême.
4. Optimiser le timing des achats.

**L’objectif final :** La réduction du risque financier et l’optimisation des coûts dans un marché où chaque mauvaise décision peut coûter des millions.

## III – Objectifs (S.M.A.R.T)

**Objectif global :** Développer un système complet de prévision de volatilité, reposant sur des séries temporelles multivariées, intégrant des modèles statistiques et deep learning, et offrant un tableau de bord simple pour les décideurs.

| Critère | Description |
| :--- | :--- |
| **S**pécifique | Construire un pipeline automatisé (prix, météo, gaz). Prévoir la volatilité sur 1 jour, 1 semaine, 1 mois. Développer un dashboard avec alertes. |
| **M**esurable | Améliorer les métriques (MAE/RMSE) d’au moins 15% vs baseline. Précision des alertes > 70%. Réduire le risque d’achat de 20% (backtest). |
| **A**tteignable | Utilisation de modèles éprouvés : Baselines, GARCH, XGBoost, LSTM. |
| **R**éaliste | Données disponibles publiquement (EPEX, Météo France, Gaz TTF). |
| **T**emporel | MVP en 2 semaines. Version avancée (modélisation/backtest) en 8–10 semaines. |

## IV – Périmètre et Limites

**Périmètre inclus**
* Prévision de la volatilité (pas du prix exact).
* Horizon à court/moyen terme.
* Approche multivariée (prix, météo, pétrole/gaz).
* Comparaison de modèles : Statistiques (GARCH, ARIMA), Machine Learning (XGBoost, RF), Deep Learning (LSTM, CNN-LSTM).

** Limites**
* Pas de trading automatique haute fréquence.
* Pas d’optimisation financière avancée (gestion de portefeuille complet).
* Qualité dépendante de la disponibilité des données brutes.
* Le système ne supprime pas le risque, il l’évalue.
  
## V – Mise en œuvre et réalisation

### 1. Collecte et préparation des données
* **Sources :** EPEX Spot (électricité), Prix du gaz/pétrole, Météo (température, anomalies), variables calendaires (jours fériés, week-ends).
* **Feature Engineering :** * Transformations temporelles (Lags : 1/7/30 jours, Moyennes glissantes, Rendements logarithmiques).
  * Approche multivariée (intégration gaz/météo).
  * Réduction de dimensionnalité (PCA) pour débruiter si nécessaire.

### 2. Modélisation
* **Baselines :** Naïve ($Prix_{t} = Prix_{t-1}$), Moyenne mobile (7/30 jours), ARIMA.
* **Modèles Statistiques :** GARCH(1,1) spécialisé dans la variance conditionnelle.
* **Machine Learning :** XGBoost, Random Forest.
* **Deep Learning :** LSTM.

### 3. Validation
* **Split chronologique :** Train (70%), Validation (15%), Test (15%). Time-series cross-validation.
* **Métriques :** RMSE, MAE pour les prix. MSE pour la volatilité. Précision des alertes de risque.

## 🎯 Vision Produit (Product Management TD)

**What is the background?** Le contexte de notre projet est la forte volatilité des prix de l’électricité et du gaz en Europe. Avec la dérégulation des marchés, la dépendance à la météo, les problèmes géopolitiques et les limites du nucléaire ou des renouvelables, les prix bougent beaucoup d’un jour à l’autre. Comme l’électricité ne se stocke pas facilement, les acteurs sont obligés de décider en permanence s’ils achètent sur le marché spot, s’ils prennent des contrats à terme ou s’ils se couvrent davantage. D’où l’idée : plutôt que de prédire le prix exact, on cherche à anticiper les périodes où la volatilité va être forte.

**What is the value proposition? Why? What kind of problems it solves?** Le but est d’aider un trader ou un acheteur d’énergie à mieux décider quand acheter et comment se couvrir. Si on sait qu’une période très volatile arrive, il peut sécuriser un contrat à terme ou augmenter son hedge. Si au contraire on anticipe une période calme, il peut continuer à acheter sur le spot et payer moins cher. Le système que l’on développe permet donc de réduire le risque financier et d’optimiser le coût d’achat dans un marché très instable.

**Do you really need ML?** On ne pense pas qu’une simple heuristique soit suffisante. Des choses comme une moyenne mobile ou une volatilité historique donnent une première idée, mais elles ne prennent pas bien en compte la complexité du marché : l’impact de la météo, des prix du gaz, de la production renouvelable, etc. C’est pour ça qu’on utilise des modèles plus avancés, comme GARCH, XGBoost ou des LSTM, qui peuvent exploiter plusieurs séries temporelles en même temps et capturer des relations non linéaires. Les heuristiques servent surtout de baseline pour comparer et montrer qu’on fait mieux.

**What are the different objectives?** 1. Récupérer et préparer les données.
2. Créer des features pertinentes (rendements, lags).
3. Entraîner et comparer les modèles sur divers horizons (1j, 1s, 1m).
4. Transformer ces prévisions en niveaux de risque (faible, moyen, fort) affichés dans un dashboard.

**What is your solution for addressing the problem?** Une chaîne complète de la donnée (ETL) à la modélisation, jusqu'à la visualisation. Les modèles sont évalués sur leurs performances (MAE/RMSE), et les prévisions génèrent un système d'alerte. Le tout sera exposé via un dashboard interactif pour être directement actionnable par un décideur.
