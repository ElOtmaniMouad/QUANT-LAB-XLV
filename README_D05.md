# D05 — Machine Learning et Deep Learning

Binôme B05 · XLV (benchmark SPY). Le rapport complet est généré par le code :
[`reports/d05_ml_dl.md`](reports/d05_ml_dl.md). Il contient le protocole, les variables, la sélection des hyperparamètres,
les résultats sur la validation et sur le test, les tests statistiques, la stabilité dans les sous-périodes, la parcimonie
et les conclusions.

## Pièces du livrable

Le cahier demande : Ridge/logistique, arbre/boosting, MLP, protocole d'entraînement, hyperparamètres, temps de calcul,
comparatifs avec les mêmes références.

| Pièce | Fichier |
|---|---|
| Paramètres gelés (snapshot, variables, grilles d'hyperparamètres, protocole, graine) | [`configs/ml.yaml`](configs/ml.yaml) |
| Variables explicatives (information connue à la clôture de t-1) | [`src/ml_features.py`](src/ml_features.py) |
| Ridge, forêt aléatoire, gradient boosting, moteur walk-forward, grilles | [`src/models/machine_learning.py`](src/models/machine_learning.py) |
| MLP compact en PyTorch | [`src/models/deep_learning.py`](src/models/deep_learning.py) |
| Pipeline complet (variables → validation → gel → test → rapport) | [`src/ml_pipeline.py`](src/ml_pipeline.py) |
| Hyperparamètres évalués et retenus | [`reports/d05_validation_grille.csv`](reports/d05_validation_grille.csv) |
| Sélection gelée avant le test | [`reports/d05_selection.json`](reports/d05_selection.json) |
| Métriques validation et test, tests de Diebold-Mariano, Holm, Pesaran-Timmermann | `reports/d05_validation_metrics.csv`, `reports/d05_test_metrics.csv` |
| Stabilité dans les sous-périodes T1–T4 | [`reports/d05_test_subperiods.csv`](reports/d05_test_subperiods.csv) |
| Temps de calcul et nombre de paramètres | [`reports/d05_parcimonie.csv`](reports/d05_parcimonie.csv) |
| Importance des variables | [`reports/d05_importances.csv`](reports/d05_importances.csv) |
| Prévisions datées de tous les modèles (validation + test), entrée de D06/D07 | [`reports/d05_predictions.csv`](reports/d05_predictions.csv) |
| Figures | `reports/figures/d05_validation_grille.png`, `d05_test_rmse.png`, `d05_prevision_vs_reel.png`, `d05_stabilite.png`, `d05_importances.png` |
| Tests | [`tests/test_ml.py`](tests/test_ml.py) |

## Commandes

Toutes les commandes se lancent depuis le dossier du projet, avec l'environnement activé.

```bash
pip install -r requirements.txt     # D05 ajoute scikit-learn et PyTorch
```

```bash
python -m pytest tests/test_ml.py   # résultat attendu : 36 passed, ~12 s
```

```bash
python -m src.ml_pipeline           # ~2 minutes (CPU) ; régénère reports/d05_* et reports/figures/d05_*
```

Résultat attendu à l'écran : la liste des modèles retenus par famille, `Sélection gelée -> reports/d05_selection.json`,
puis `Rapport écrit : reports/d05_ml_dl.md`. Pour vérifier la reproductibilité, lancez `git status` : seul
`reports/d05_selection.json` peut apparaître comme modifié (il contient la date d'exécution) ; la durée affichée dans le
rapport et la colonne `secondes_test` de `d05_parcimonie.csv` varient aussi selon la machine. Tous les autres chiffres
doivent être identiques.

## Protocole (identique à D03, pour que la comparaison soit directe)

| Élément | Choix |
|---|---|
| Cible | r_t = ln(P_t / P_{t-1}) sur prix ajusté, prévu à la clôture de t-1 (une séance) ; métriques en % |
| Variables | 20 variables, toutes connues à la clôture de t-1 : 5 rendements retardés de XLV, moyennes et volatilités sur 5/20/60 séances, 3 rendements retardés de SPY et sa volatilité, force relative XLV − SPY, écart aux moyennes mobiles 50/200 jours, volume relatif |
| Normalisation | `StandardScaler` ajusté sur les seules données d'entraînement de chaque ré-estimation |
| Échantillons | train 2018-2022 · validation 2023-2024 · test 2025-01-02 → 2026-10-30 (459 séances) |
| Estimation | fenêtre croissante depuis 2018 ; ré-estimation toutes les 21 séances avec les seules lignes antérieures au bloc |
| Hyperparamètres | grille limitée (5 Ridge, 4 forêt, 4 boosting, 4 MLP), évaluée **uniquement sur la validation** ; choix gelé dans `d05_selection.json` avant la première prévision du test |
| Références communes | marche aléatoire (r̂ = 0), moyenne historique, moyenne mobile 60, ARIMA(3,0,2) et SARIMAX de D03, ré-estimés ici : leurs RMSE reproduisent ceux de D03 à l'identique |
| Tests | Diebold-Mariano contre la marche aléatoire avec correction de **Holm** ; Pesaran-Timmermann sur la direction |
| Graine | `seed` dans `configs/ml.yaml` ; CPU, un seul fil pour PyTorch : mêmes données ⇒ mêmes prévisions |

Le choix d'une **régression** (et non d'une classification) garde la même cible et les mêmes métriques qu'en D03. Le signe de
la prévision donne la direction pour la règle de décision de D06.

## Résultats (test 2025-2026)

| Modèle | RMSE (%) | RMSE / marche aléatoire | Exactitude directionnelle | p ajustée (RMSE) |
|---|---|---|---|---|
| Marche aléatoire (r̂ = 0) | 1,0776 | 1,000 | 51,0 % | — |
| Moyenne historique | 1,0760 | 0,999 | 51,0 % | 1,00 |
| ARIMA(3,0,2) de D03 | 1,0930 | 1,014 | 51,4 % | 1,00 |
| Ridge (α = 10 000) | 1,0748 | 0,997 | 55,8 % | 1,00 |
| Forêt aléatoire | 1,0791 | 1,001 | 51,9 % | 1,00 |
| Gradient boosting | 1,0772 | 1,000 | 52,3 % | 1,00 |
| MLP (PyTorch) | 1,0804 | 1,003 | 52,7 % | 1,00 |

- **Aucun modèle ML/DL ne bat significativement la marche aléatoire** en erreur de prévision. Les écarts de RMSE sont de
  l'ordre de 0,1 point de base par séance, indiscernables du hasard avec 459 séances.
- **Une seule exception à signaler honnêtement : la direction de Ridge** (55,8 %, test de Pesaran-Timmermann p = 0,009).
  Elle n'est plus significative après correction de Holm (p = 0,063), n'existait pas en validation (49,4 %) et n'est accompagnée
  d'aucun gain de RMSE. C'est une hypothèse à confirmer en D06 (backtest net de coûts) et D08 (nouvelles séances), pas une preuve.
- **Variante B05** : l'avantage prévisionnel n'est stable dans aucune sous-période pour aucun modèle ; le meilleur, Ridge, bat la marche
  aléatoire dans 3 sous-périodes sur 4, de très peu.
- **Parcimonie** : de 21 à 1 492 paramètres (7 pour l'ARIMA, 0 pour la marche aléatoire) et jusqu'à 13 s de calcul pour le test, sans gain
  significatif : la complexité n'est pas justifiée sur cet échantillon. Ce résultat prolonge D03 pour la moyenne ; D04 montre un rôle
  plus net pour les modèles de risque.

## Limites

Une seule série et un horizon d'une séance ; 459 séances de test seulement ; grilles volontairement petites (un LSTM ou un TCN,
facultatifs, n'ont pas été explorés) ; le RMSE ne mesure pas la rentabilité, qui se juge en D06 avec frais, glissement et turnover.
Le détail est dans la section 9 du rapport.
