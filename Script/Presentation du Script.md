# Dossier Script : Pipeline de Données et Modélisation

Ce dossier contient l'ensemble du code source industrialisé du projet. Il a pour but de transformer l'analyse exploratoire (réalisée dans le dossier `Notebook`) en un pipeline de données modulaire, automatisé et reproductible.

L'objectif de cette architecture est de pouvoir traiter de nouvelles données de manière fluide, de l'extraction brute jusqu'à la génération des prévisions de volatilité, sans intervention manuelle.

## Architecture du code

Le code a été découpé en plusieurs modules spécifiques pour respecter les bonnes pratiques de développement :

* **`data_preprocessing.py`** : Gère le chargement des données brutes (Électricité, Gaz, Météo), le nettoyage (gestion des valeurs manquantes par interpolation) et la fusion des différents jeux de données sur un index temporel commun.
* **`feature_engineering.py`** : Contient les fonctions permettant de créer les variables explicatives nécessaires aux modèles (calcul des rendements, volatilité historique, création de lags).
* **`models.py`** : Regroupe les fonctions d'entraînement et de prédiction pour nos différents modèles (GARCH, Random Forest, LSTM).
* **`main.py`** : C'est le script principal (le "chef d'orchestre"). Il importe les fonctions des autres modules et exécute le pipeline complet de bout en bout. 

## Comment exécuter le pipeline ?

Pour lancer l'intégralité du traitement des données et générer les nouvelles prévisions, placez-vous à la racine du projet dans votre terminal et exécutez la commande suivante :

```bash
python Script/main.py
