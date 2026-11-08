# QUANT LAB — Analyse et Trading Quantitatif de l'ETF XLV

## 1. Présentation du projet

QUANT LAB est une application Web de trading quantitatif réalisée dans le cadre du module Trading Intelligent à la Faculté des Sciences de Rabat.

Le projet consiste à collecter et analyser des données financières réelles, entraîner différents modèles de prévision, évaluer les risques et transformer les prévisions en décisions de trading simulées.

Notre équipe travaille sur la variante **B05 — XLV (Health Care Select Sector SPDR Fund)**, représentant le secteur américain de la santé.

## 2. Identification de l'équipe

- **Membre 1 :** EL OTMANI Mouad
- **Membre 2 :** Seyid Ahmed SALEM

## 3. Question scientifique

Notre question de recherche est :

**Les modèles complexes apportent-ils quelque chose face à des références simples sur un secteur relativement défensif ?**

L'expérience consiste à tester la stabilité de l'avantage prévisionnel dans plusieurs sous-périodes du test.

L'objectif est de déterminer si les modèles complexes améliorent réellement les prévisions et les décisions de trading par rapport aux modèles simples.

## 4. Actif financier étudié

| Élément | Valeur |
|---|---|
| Ticker principal | XLV |
| Nom | Health Care Select Sector SPDR Fund |
| Marché | États-Unis |
| Secteur | Santé |
| Benchmark | SPY |
| Fréquence | Journalière |
| Capital initial | 100 000 USD virtuels |

## 5. Méthodologie

Le projet utilise plusieurs familles de modèles :

- Références simples : Buy and Hold, Cash, modèle naïf et moyenne mobile.
- Économétrie : ARIMA, SARIMA et SARIMAX.
- Gestion du risque : ARCH-LM, GARCH(1,1), VaR et Expected Shortfall.
- Machine Learning : Ridge et Random Forest.
- Deep Learning : MLP.

Les modèles seront comparés selon leurs performances prédictives et leur utilité pour les décisions de trading.

## 6. Fonctionnalités de l'application

L'application Streamlit comprend six espaces :

1. Market Data : collecte et visualisation des données.
2. Model Laboratory : entraînement et comparaison des modèles.
3. Risk Management : analyse de la volatilité et du risque.
4. Trading Strategy : génération des signaux de trading.
5. Backtesting : simulation et évaluation des stratégies.
6. Paper Trading : suivi des transactions fictives.

## 7. Technologies utilisées

- Python 3.11+
- Streamlit
- Pandas et NumPy
- Matplotlib et Plotly
- Statsmodels et ARCH
- Scikit-learn
- PyTorch ou TensorFlow
- yfinance
- Git et GitHub

## 8. Répartition des tâches

**EL OTMANI Mouad :**
- Collecte et exploration des données financières.
- Préparation et nettoyage des données.
- Développement des modèles ARIMA, SARIMA et SARIMAX.
- Développement des modèles de Machine Learning et Deep Learning.
- Participation à l'intégration de l'application.

**Seyid Ahmed SALEM :**
- Analyse statistique des rendements.
- Développement du module GARCH et gestion des risques.
- Développement de la stratégie de trading.
- Développement du moteur de Backtesting.
- Développement du module Paper Trading.

**Travail commun :**
- Développement de l'interface Streamlit.
- Tests et validation des résultats.
- Intégration des différents modules.
- Rédaction du rapport et préparation de la soutenance.

## 9. Cadre de simulation

Le projet utilise un capital initial de 100 000 USD virtuels.

Aucune transaction réelle, aucun effet de levier et aucune vente à découvert ne sont autorisés.

Les décisions sont prises après la clôture du marché et exécutées au plus tôt à l'ouverture de la séance suivante.

## 10. Installation et lancement

Installation des dépendances :

```bash
pip install -r requirements.txt
```

Lancement de l'application :

```bash
streamlit run app.py
```

## 11. Encadrement

- Module : Trading Intelligent
- Projet : QUANT LAB
- Enseignant : M. Tawfik Masrour
- Établissement : Faculté des Sciences, Université Mohammed V de Rabat.
- Année universitaire : 2026–2027
  
