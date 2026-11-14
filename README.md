# QUANT LAB — Analyse et Trading Quantitatif de l'ETF XLV

## 1. Présentation du projet

QUANT LAB est une application Web de trading quantitatif réalisée dans le cadre du module **Trading Intelligent** à la Faculté des Sciences de Rabat.

Le projet consiste à collecter et analyser des données financières réelles, entraîner différents modèles de prévision, évaluer les risques et transformer les prévisions en décisions de trading simulées.

Notre équipe travaille sur la variante **B05 — XLV (Health Care Select Sector SPDR Fund)**, représentant le secteur américain de la santé.

L'application sera développée avec Python et Streamlit. Elle permettra de suivre les données du marché, de comparer plusieurs modèles, de tester des stratégies et de réaliser des simulations de trading sans utiliser d'argent réel.

## 2. Identification de l'équipe

- **Membre 1 :** EL OTMANI Mouad
- **Membre 2 :** Seyid Ahmed SALEM

## 3. Question scientifique

Notre question de recherche est :

**Les modèles complexes apportent-ils quelque chose face à des références simples sur un secteur relativement défensif ?**

L'expérience consiste à tester la stabilité de l'avantage prévisionnel dans plusieurs sous-périodes du test.

L'objectif est de déterminer si les modèles complexes améliorent réellement les prévisions et les décisions de trading par rapport aux modèles simples.

Nous chercherons notamment à répondre aux questions suivantes :

- Les modèles de Machine Learning et de Deep Learning sont-ils plus performants que les modèles classiques ?
- Les améliorations observées sont-elles statistiquement et économiquement significatives ?
- Les performances des modèles restent-elles stables sur différentes périodes ?
- La complexité supplémentaire des modèles est-elle justifiée par leurs résultats ?

## 4. Actif financier étudié

| Élément | Valeur |
|---|---|
| Ticker principal | XLV |
| Nom | Health Care Select Sector SPDR Fund |
| Marché | États-Unis |
| Secteur | Santé |
| Benchmark | SPY |
| Fréquence principale | Journalière |
| Capital initial | 100 000 USD virtuels |

**XLV :** actif principal étudié, représentant le secteur de la santé.

**SPY :** benchmark représentant le marché américain à travers le S&P 500. Il permettra de comparer le comportement de XLV avec celui du marché global.

## 5. Architecture et gestion des données

### 5.1. Sources de données

Les données financières seront récupérées principalement depuis Yahoo Finance à l'aide de la bibliothèque Python `yfinance`.

Les deux instruments étudiés seront :

- XLV : actif principal du projet.
- SPY : benchmark de comparaison.

La période historique proposée commence en janvier 2018 et se termine à la dernière date complète disponible au moment de la collecte.

### 5.2. Données récupérées

Les données journalières comprendront les variables suivantes :

| Variable | Description |
|---|---|
| Date | Date de la séance boursière |
| Open | Prix d'ouverture |
| High | Prix maximum |
| Low | Prix minimum |
| Close | Prix de clôture |
| Adj Close | Prix ajusté des dividendes et opérations sur titres |
| Volume | Volume des transactions |

Les données seront conservées avec leur ticker, leur fréquence, leur source et leur date de récupération.

### 5.3. Actualisation des données du marché

L'application intégrera un espace permettant de récupérer et d'actualiser les dernières données disponibles.

Le fonctionnement prévu sera le suivant :

1. Récupération des données historiques de XLV et SPY.
2. Actualisation des dernières observations disponibles auprès de la source.
3. Vérification de la date et de la qualité des nouvelles données.
4. Mise à jour des fichiers de données.
5. Recalcul des indicateurs nécessaires.
6. Affichage des dernières informations dans l'interface Streamlit.

Un bouton **Refresh Data** permettra de lancer manuellement l'actualisation.

L'application pourra également afficher la dernière cotation disponible et les dernières données journalières.

**Remarque importante :** l'utilisation de Yahoo Finance avec yfinance ne garantit pas un flux boursier professionnel en temps réel. Les données peuvent être différées et leur disponibilité dépend de la source et de l'intervalle demandé. Le projet utilisera donc les dernières données effectivement disponibles. Une véritable alimentation intrajournalière en temps réel nécessiterait une source spécialisée compatible.

Les données actualisées ne seront pas automatiquement utilisées pour modifier les résultats du test final déjà réalisé. Chaque expérience conservera une copie datée de ses données afin de garantir la reproductibilité.

### 5.4. Préparation et nettoyage des données

Avant leur utilisation, les données passeront par plusieurs opérations :

- Vérification des valeurs manquantes.
- Détection des doublons.
- Vérification de l'ordre chronologique.
- Contrôle des valeurs incohérentes.
- Synchronisation des dates de XLV et SPY.
- Calcul des rendements journaliers.
- Calcul des indicateurs financiers nécessaires aux modèles.

Le rendement journalier sera calculé à partir des prix ajustés :

\[
R_t = \frac{P_t}{P_{t-1}} - 1
\]

où \(P_t\) représente le prix ajusté à la date \(t\).

### 5.5. Organisation du stockage des données

Les données seront organisées de la manière suivante :

```text
data/
├── raw/
│   ├── xlv.csv
│   └── spy.csv
│
├── processed/
│   ├── xlv_clean.csv
│   ├── spy_clean.csv
│   └── dataset_final.csv
│
└── snapshots/
    ├── 2026-11-20/
    └── 2026-11-27/
```

- `raw/` : données originales récupérées depuis la source.
- `processed/` : données nettoyées et préparées.
- `snapshots/` : copies datées utilisées pour les expérimentations.

Les formats CSV ou Parquet seront utilisés pour le stockage.

### 5.6. Architecture du flux de données

```text
Yahoo Finance
      |
      v
XLV et SPY
      |
      v
Data Collection (yfinance)
      |
      v
Data Cleaning
      |
      v
Feature Engineering
      |
      v
Data Storage (CSV / Parquet)
      |
      v
Dataset d'entraînement et de test
      |
      v
Model Laboratory
      |
      v
Risk Management
      |
      v
Trading Strategy
      |
      v
Backtesting
      |
      v
Paper Trading
```

## 6. Méthodologie et modèles étudiés

Le projet utilise plusieurs familles de modèles.

### 6.1. Modèles de référence

Les modèles simples serviront de références pour évaluer l'intérêt des modèles plus complexes.

- Buy and Hold.
- Cash.
- Prévision naïve.
- Moyenne historique.
- Moyenne mobile.

### 6.2. Modèles économétriques

- ARIMA.
- SARIMA, si une structure saisonnière pertinente est identifiée.
- SARIMAX avec au moins une variable explicative externe retardée.

### 6.3. Modèles de gestion du risque

- Test ARCH-LM.
- GARCH(1,1).
- Prévision de la volatilité conditionnelle.
- Value at Risk (VaR).
- Expected Shortfall (ES).

### 6.4. Modèles de Machine Learning

- Ridge Regression.
- Random Forest.
- Gradient Boosting, si les ressources et le calendrier le permettent.

### 6.5. Modèles de Deep Learning

- MLP (Multi-Layer Perceptron).
- LSTM ou TCN en extension éventuelle, si le temps et les ressources le permettent.

## 7. Démonstration et comparaison des modèles

Cette partie constitue un élément central du projet. L'application devra permettre de présenter les résultats de chaque modèle et de comparer leurs performances dans des conditions identiques.

### 7.1. Objectif de la comparaison

La comparaison doit répondre à notre question scientifique : les modèles complexes apportent-ils une amélioration réelle par rapport aux références simples sur XLV ?

Pour cela, les modèles seront évalués sur les mêmes données, les mêmes horizons de prévision et les mêmes périodes de test.

### 7.2. Tableau comparatif des modèles

| Famille | Modèle | Rôle |
|---|---|---|
| Baseline | Naïve | Référence de prévision |
| Baseline | Buy and Hold | Référence de stratégie |
| Baseline | Moyenne mobile | Prévision simple |
| Économétrie | ARIMA | Prévision de série temporelle |
| Économétrie | SARIMAX | Prévision avec variable externe |
| Risque | GARCH(1,1) | Prévision de volatilité |
| Machine Learning | Ridge | Régression régularisée |
| Machine Learning | Random Forest | Apprentissage non linéaire |
| Deep Learning | MLP | Apprentissage par réseau de neurones |

### 7.3. Métriques de comparaison prédictive

Les modèles de prévision seront évalués avec plusieurs indicateurs :

- **MAE :** mesure l'erreur absolue moyenne.
- **RMSE :** mesure l'erreur quadratique moyenne.
- **Directional Accuracy :** mesure la proportion de directions correctement prédites.

Les résultats seront présentés sous forme de tableaux et de graphiques.

### 7.4. Comparaison des performances financières

Les stratégies générées à partir des prévisions seront comparées à l'aide des indicateurs suivants :

- Rendement total.
- Rendement annualisé.
- Volatilité annualisée.
- Sharpe Ratio.
- Maximum Drawdown.
- Nombre de transactions.
- Exposition au marché.
- Coûts de transaction.

La comparaison inclura les coûts de transaction et le slippage.

### 7.5. Démonstration dans l'application

La section Model Laboratory permettra de :

1. Sélectionner un modèle.
2. Choisir la période d'étude.
3. Lancer l'entraînement et les prévisions.
4. Afficher les valeurs prédites et les valeurs réelles.
5. Visualiser les erreurs de prévision.
6. Comparer les modèles dans un tableau de résultats.

Des graphiques permettront notamment de présenter :

- Les prix réels et les prix prévus.
- Les rendements réels et les rendements prévus.
- Les erreurs de chaque modèle.
- La comparaison des métriques.
- La stabilité des résultats entre les différentes périodes de test.

La section Backtesting permettra ensuite de comparer les performances des stratégies associées aux modèles.

**Important :** GARCH sera principalement évalué comme modèle de volatilité et de risque. Il ne sera pas comparé directement à ARIMA ou Ridge avec les mêmes métriques de prévision du prix.

## 8. Protocole d'expérimentation

Afin de garantir une comparaison scientifique et équitable, nous allons suivre un protocole expérimental commun à tous les modèles.

### 8.1. Constitution du dataset

Les données seront récupérées pour XLV et SPY, puis nettoyées et synchronisées.

La période proposée est de janvier 2018 à octobre 2026, sous réserve de disponibilité des données.

### 8.2. Séparation chronologique

Nous proposons le découpage suivant :

| Ensemble | Période | Utilisation |
|---|---|---|
| Train | 2018–2022 | Entraînement des modèles |
| Validation | 2023–2024 | Sélection des paramètres |
| Test final | 2025–2026 | Évaluation finale |

Le test final sera subdivisé en plusieurs périodes afin de vérifier la stabilité des résultats :

- Sous-période 1 : janvier à juin 2025.
- Sous-période 2 : juillet à décembre 2025.
- Sous-période 3 : janvier à juin 2026.
- Sous-période 4 : juillet à octobre 2026.

Les périodes exactes dépendront des données disponibles et des séances de marché complètes.

### 8.3. Entraînement et validation

Chaque modèle sera entraîné uniquement avec les données historiques disponibles à la date considérée.

Les paramètres seront sélectionnés à partir des données d'entraînement et de validation.

Les transformations, normalisations et sélections de variables seront ajustées sur les données d'entraînement uniquement, puis appliquées aux données suivantes.

### 8.4. Évaluation hors échantillon

Les modèles seront évalués sur les périodes de test qu'ils n'ont pas utilisées pour leur entraînement ou leur réglage.

Nous utiliserons une approche chronologique de type Walk-Forward lorsque cela est adapté :

1. Entraîner le modèle sur les données passées.
2. Produire une prévision pour la période suivante.
3. Comparer la prévision avec la valeur réellement observée.
4. Avancer dans le temps.
5. Répéter le processus.

Cette méthode permet de simuler les conditions réelles dans lesquelles un modèle aurait été utilisé.

### 8.5. Comparaison équitable

Pour garantir une comparaison fiable :

- Les modèles utiliseront les mêmes périodes d'évaluation.
- Les mêmes variables cibles seront utilisées lorsque cela est pertinent.
- Les horizons de prévision seront identiques.
- Les modèles simples seront toujours inclus comme références.
- Les paramètres seront réglés sans utiliser le test final.
- Les résultats seront calculés après prise en compte des coûts pour les stratégies de trading.

### 8.6. Protocole de trading simulé

Les signaux de trading seront générés après la clôture de la séance.

Les ordres seront exécutés au plus tôt à l'ouverture de la séance suivante.

Les hypothèses initiales de coûts seront :

- Frais de transaction : 0,05 % par côté.
- Slippage : 0,05 % par côté.
- Coût total estimé : 0,10 % par côté.

Une analyse de sensibilité sera également effectuée en doublant les coûts pour mesurer leur impact sur les performances.

### 8.7. Analyse de la stabilité des modèles

La particularité de notre variante B05 est l'étude de la stabilité de l'avantage prédictif.

Nous allons donc comparer les résultats obtenus sur les quatre sous-périodes de test.

Un modèle ne sera pas considéré comme meilleur uniquement parce qu'il obtient le meilleur résultat global.

Nous allons également vérifier :

- Si son avantage apparaît dans plusieurs sous-périodes.
- Si ses performances restent relativement stables.
- Si son amélioration est significative sur le plan financier.
- Si les résultats restent intéressants après les coûts de transaction.

### 8.8. Reproductibilité

Chaque expérimentation devra conserver :

- La date de récupération des données.
- La période étudiée.
- La configuration du modèle.
- Les paramètres utilisés.
- Les métriques obtenues.
- Les prévisions et résultats de test.
- La version du code utilisée.

Les expériences importantes seront documentées afin de pouvoir reproduire les résultats.

## 9. Fonctionnalités de l'application

L'application Streamlit comprend six espaces :

1. **Market Data :** collecte, actualisation et visualisation des données.
2. **Model Laboratory :** entraînement et comparaison des modèles.
3. **Risk Management :** analyse de la volatilité et du risque.
4. **Trading Strategy :** génération des signaux de trading.
5. **Backtesting :** simulation et évaluation des stratégies.
6. **Paper Trading :** suivi des transactions fictives.

## 10. Cahier des charges et calendrier prévisionnel

Le projet sera réalisé progressivement selon les tâches et les dates prévisionnelles suivantes.

| ID | Tâche | Description | Responsable principal | Date prévue |
|---|---|---|---|---|
| D01 | Cadrage du projet | README, équipe, ticker, problématique, répartition et architecture | Binôme | 16–20 novembre 2026 |
| D02 | Collecte et audit des données | Récupération XLV/SPY, nettoyage, dictionnaire et qualité des données | Mouad | 16–27 novembre 2026 |
| D03 | Modèles économétriques | Baselines, ARIMA, SARIMA, SARIMAX et diagnostics | Mouad | 23 novembre–11 décembre 2026 |
| D04 | Gestion des risques | ARCH-LM, GARCH, volatilité, VaR et ES | Ahmed | 30 novembre–11 décembre 2026 |
| D05 | Machine Learning et Deep Learning | Ridge, Random Forest, MLP et comparaison | Mouad | 7–25 décembre 2026 |
| D06 | Stratégie et Backtesting | Règles de trading, moteur de simulation, coûts et résultats | Ahmed | 14 décembre 2026–8 janvier 2027 |
| D07 | Application Streamlit | Développement et intégration des six espaces | Binôme | 4–15 janvier 2027 |
| D08 | Paper Trading | Simulation sur au moins cinq nouvelles séances de marché | Binôme | 11–22 janvier 2027 |
| D09 | Tests et reproductibilité | Tests, documentation, environnement et vérification des résultats | Binôme | 18–26 janvier 2027 |
| D10 | Rapport et soutenance | Rapport final, présentation et démonstration | Binôme | 25–29 janvier 2027 |

*Ce calendrier est prévisionnel et devra être adapté aux échéances officielles communiquées par l'enseignant.*

### 10.1. Livrables attendus

- D01 : document de cadrage et dépôt GitHub initial.
- D02 : données récupérées, nettoyées et documentées.
- D03 : résultats des modèles économétriques.
- D04 : rapport d'analyse du risque et de la volatilité.
- D05 : résultats des modèles de Machine Learning et Deep Learning.
- D06 : stratégie de trading et résultats du Backtesting.
- D07 : application Streamlit fonctionnelle.
- D08 : preuves des séances de Paper Trading.
- D09 : environnement reproductible et tests.
- D10 : rapport final et présentation du projet.

## 11. Répartition des tâches

### EL OTMANI Mouad

- Collecte et exploration des données financières.
- Préparation et nettoyage des données.
- Développement des modèles ARIMA, SARIMA et SARIMAX.
- Développement des modèles de Machine Learning et Deep Learning.
- Participation à l'intégration de l'application.

### Seyid Ahmed SALEM

- Analyse statistique des rendements.
- Développement du module GARCH et gestion des risques.
- Développement de la stratégie de trading.
- Développement du moteur de Backtesting.
- Développement du module Paper Trading.

### Travail commun

- Développement de l'interface Streamlit.
- Tests et validation des résultats.
- Intégration des différents modules.
- Analyse comparative des modèles.
- Rédaction du rapport et préparation de la soutenance.

## 12. Cadre de simulation

Le projet utilise un capital initial de **100 000 USD virtuels**.

Les règles suivantes seront respectées :

- Aucune transaction réelle.
- Aucun effet de levier.
- Aucune vente à découvert.
- Position longue ou liquidités uniquement.
- Décisions prises après la clôture du marché.
- Exécution des transactions au plus tôt à l'ouverture de la séance suivante.
- Prise en compte des frais et du slippage.

Le Paper Trading servira à simuler les décisions sur de nouvelles séances après le gel de la stratégie.

Au moins cinq nouvelles séances de marché devront être documentées.

## 13. Technologies utilisées

- Python 3.11+
- Streamlit
- Pandas et NumPy
- Matplotlib et Plotly
- Statsmodels et ARCH
- Scikit-learn
- PyTorch ou TensorFlow
- yfinance
- CSV et Parquet
- Git et GitHub
- Pytest

## 14. Organisation du dépôt GitHub

```text
QUANT-LAB-XLV/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── snapshots/
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── arima.py
│   │   ├── sarimax.py
│   │   ├── garch.py
│   │   ├── machine_learning.py
│   │   └── deep_learning.py
│   │
│   ├── risk.py
│   ├── strategy.py
│   ├── backtest.py
│   └── paper.py
│
├── notebooks/
│   └── data_exploration.ipynb
│
├── configs/
├── tests/
├── reports/
└── docs/
```

Les clés API et les informations confidentielles ne devront jamais être publiées dans le dépôt GitHub.

## 15. Installation et lancement

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Lancement de l'application

```bash
streamlit run app.py
```

L'application sera exécutée localement sur le navigateur.

## 16. Critères de validation du projet

Le projet sera considéré comme satisfaisant lorsque les éléments suivants seront validés :

- Les données financières sont correctement récupérées et documentées.
- Le nettoyage et la préparation des données sont reproductibles.
- Les modèles sont entraînés et évalués selon un protocole commun.
- Les comparaisons incluent des références simples.
- Le Data Leakage est évité.
- Les performances sont évaluées sur plusieurs sous-périodes.
- Les stratégies prennent en compte les coûts de transaction.
- Le Backtesting et le Paper Trading fonctionnent correctement.
- L'application Streamlit présente les résultats de manière claire.
- Les expériences peuvent être reproduites à partir du dépôt GitHub.

## 17. Encadrement

- **Module :** Trading Intelligent
- **Projet :** QUANT LAB
- **Enseignant :** M. Tawfik Masrour
- **Établissement :** Faculté des Sciences, Université Mohammed V de Rabat
- **Année universitaire :** 2026–2027
