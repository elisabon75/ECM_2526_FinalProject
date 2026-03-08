# Dossier Script : Pipeline de Données et Modélisation

Ce dossier contient l'ensemble du code source industrialisé du projet. Il a pour but de transformer l'analyse exploratoire (réalisée dans le dossier `Notebook`) en un pipeline de données modulaire, automatisé et reproductible.

L'objectif de cette architecture est de pouvoir traiter de nouvelles données de manière fluide, de l'extraction brute jusqu'à la génération des prévisions de volatilité, sans intervention manuelle.

## Architecture du code

Le code a été découpé en plusieurs modules spécifiques pour respecter les bonnes pratiques de développement :

* **`data_preprocessing.py`** : Gère le chargement des données brutes (Électricité, Gaz, Météo), le nettoyage (gestion des valeurs manquantes par interpolation) et la fusion des différents jeux de données sur un index temporel commun.
* **`feature_engineering.py`** : Contient les fonctions permettant de créer les variables explicatives nécessaires aux modèles (calcul des rendements, volatilité historique, création de lags).
* **`models.py`** : Regroupe les fonctions d'entraînement et de prédiction pour nos différents modèles (GARCH, Random Forest, LSTM).
* **`main.py`** : C'est le script principal (le "chef d'orchestre"). Il importe les fonctions des autres modules et exécute le pipeline complet de bout en bout.
* **`test_verif.py`** : Script utilitaire permettant de calculer et d'afficher le classement final des modèles (via les métriques MAE et RMSE) en s'assurant que l'évaluation se fait strictement sur la même période de test pour chaque algorithme.
* **`dashboard.py`** : Application web interactive (développée avec Streamlit) permettant de visualiser les courbes de prédiction superposées à la volatilité réelle, de cibler des périodes spécifiques, et d'afficher les scores de performance dynamiquement.

## Prérequis et Installation

Il est recommandé d'utiliser un IDE tel que PyCharm et de travailler dans un environnement virtuel isolé. 

Avant de lancer les scripts, assurez-vous d'installer l'ensemble des dépendances requises. Placez-vous à la racine du projet dans votre terminal et exécutez :

```bash
pip install -r requirements.txt
```

## Comment exécuter le pipeline ?

Une fois l'environnement configuré, l'exécution se fait en deux étapes :

**1. Entraînement des modèles et génération des prévisions**

Pour lancer l'intégralité du traitement des données, entraîner les algorithmes et générer le fichier de résultats, exécutez la commande suivante :

```bash
python Script/main.py
```

**2. Visualisation des résultats (Dashboard)**

Pour analyser visuellement les performances des modèles de manière interactive, démarrez l'interface web avec la commande suivante :

```bash
streamlit run Script/dashboard.py
```

Cette commande ouvrira automatiquement le tableau de bord dans votre navigateur internet par défaut.
