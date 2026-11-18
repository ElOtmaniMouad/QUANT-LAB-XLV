# D03 — Références simples et modèles économétriques (XLV)

> Fichier généré par `python -m src.econometrics`. Ne pas modifier à la main : relancer la commande.

## 1. Cadre

- Snapshot D02 : `data/snapshots/XLV_SPY_1d_20261116.csv` (SHA-256 `d80d6249f3dcebd0…`, vérifié au chargement)
- Cible : rendement log journalier de XLV, r_t = ln(P_t / P_{t-1}), P = prix ajusté (dividendes et fractionnements).
- Horizon : 1 séance. La prévision de r_t est calculée à la clôture de t-1 avec l'information disponible à cette clôture uniquement.
- Échantillons : train 2018-01-02 → 2022-12-30 (1259 séances) · validation 2023-01-03 → 2024-12-31 (502) · test 2025-01-02 → 2026-10-30 (459). L'année 2017 sert seulement d'amorçage.
- Walk-forward : fenêtre croissante depuis janvier 2018, paramètres ré-estimés toutes les 21 séances, état du filtre mis à jour chaque jour.
- Logiciel : statsmodels 0.14.6, pandas 2.3.3, numpy 2.0.2.

## 2. Identification (train 2018-2022 uniquement)

### 2.1 Stationnarité

| serie | ADF_stat | ADF_p | KPSS_stat | KPSS_p | conclusion |
|---|---|---|---|---|---|
| log-prix ln P_t | -0.7801 | 0.8249 | 5.6177 | 0.0100 | non stationnaire |
| rendement log r_t | -11.5344 | 0.0000 | 0.0224 | 0.1000 | stationnaire |

Le log-prix a une racine unitaire, le rendement est stationnaire : on modélise donc r_t avec d = 0, ce qui équivaut à un ARIMA(p, 1, q) sur le log-prix. La constante correspond à la dérive.

### 2.2 Grille ARIMA(p, 0, q) — 8 meilleurs AIC

| p | q | aic | bic | llf | converged | d_aic | d_bic |
|---|---|---|---|---|---|---|---|
| 3 | 2 | 3995.93 | 4031.90 | -1990.97 | True | 0.00 | 1.50 |
| 2 | 2 | 3999.57 | 4030.39 | -1993.78 | True | 3.63 | 0.00 |
| 1 | 3 | 4065.05 | 4095.87 | -2026.52 | True | 69.11 | 65.48 |
| 2 | 3 | 4066.07 | 4102.03 | -2026.03 | True | 70.14 | 71.64 |
| 3 | 3 | 4066.22 | 4107.33 | -2025.11 | True | 70.29 | 76.93 |
| 3 | 1 | 4066.26 | 4097.09 | -2027.13 | True | 70.33 | 66.69 |
| 0 | 3 | 4072.59 | 4098.28 | -2031.30 | True | 76.66 | 67.89 |
| 1 | 2 | 4074.73 | 4100.42 | -2032.37 | True | 78.80 | 70.03 |

Grille complète : `reports/d03_order_grid.csv`. Le bruit blanc ARIMA(0,0,0) a un ΔAIC de 108.2. Les critères voient donc une dépendance linéaire dans le train. Les meilleurs ordres ont des racines AR et MA proches l'une de l'autre et proches du cercle unité (section 6.2) : ce sont des structures fragiles.

### 2.3 D'où vient cette structure ? Robustesse à l'épisode Covid

| echantillon | n | acf_1 | corr_r_SPY(t-1) | gain_AIC_ARIMA(1,0,3) |
|---|---|---|---|---|
| train complet 2018-2022 | 1259 | -0.1271 | -0.1178 | 39.1023 |
| train sans 15/02/2020-30/06/2020 | 1165 | 0.0327 | 0.0240 | 1.6683 |
| 2018-2019 seulement | 503 | 0.0367 | 0.0039 | -1.9796 |
| 2021-2022 seulement | 503 | 0.0328 | 0.0437 | -5.7472 |

Sans la période 2020-02-15 → 2020-06-30 (moins de 8 % du train), l'autocorrélation d'ordre 1 passe de -0.127 à +0.033, la corrélation avec r^SPY(t-1) de -0.118 à +0.024, et le gain d'AIC de l'ARIMA retenu de 39.1 à 1.7. **La dépendance linéaire du train vient presque entièrement du krach et du rebond de 2020.** En dehors de cet épisode, le rendement de XLV se comporte comme un bruit blanc. Les modèles ne sont pas ré-estimés sans 2020 : ce choix serait guidé par le résultat. Cette table sert seulement à interpréter les résultats hors échantillon.

### 2.4 Saisonnalité

- Effet jour de la semaine (régression sur indicatrices, écarts-types robustes) : F = 0.92, p = 0.451. Rendements moyens (%) : lundi -0.085, mardi +0.059, mercredi +0.109, jeudi +0.026, vendredi +0.112.
- Composantes saisonnières hebdomadaires (s = 5) ajoutées à l'ARIMA retenu :

| candidat | AIC_train | BIC_train | LR_vs_ARIMA | LR_p |
|---|---|---|---|---|
| SARIMA(1,0,3)(1,0,0)_5 | 4062.174 | 4098.140 | 4.872 | 0.027 |
| SARIMA(1,0,3)(0,0,1)_5 | 4061.461 | 4097.428 | 5.585 | 0.018 |
| SARIMA(1,0,3)(1,0,1)_5 | 4062.056 | 4103.161 | 6.990 | 0.030 |

Conclusion : la saisonnalité hebdomadaire est **pertinente au sens statistique** sur le train (AIC plus bas et test LR significatif à 5 %). Le SARIMA est quand même évalué hors échantillon, pour vérifier si ce résultat tient sur des données nouvelles.

## 3. Sélection sur la validation (2023-2024)

Les ordres candidats viennent du train (les 3 meilleurs AIC et le meilleur BIC, sans le bruit blanc). Pour chaque famille, on retient le candidat dont la RMSE walk-forward en validation est la plus faible. Le test n'intervient dans aucun choix.

| famille | candidat | RMSE_val_% | non_convergences | note | retenu |
|---|---|---|---|---|---|
| ARIMA | ARIMA(3,0,2) | 0.7278 | 0 | AIC train 3995.9 | False |
| ARIMA | ARIMA(2,0,2) | 0.7282 | 0 | AIC train 3999.6 | False |
| ARIMA | ARIMA(1,0,3) | 0.7230 | 1 | AIC train 4065.0 | True |
| SARIMA | SARIMA(1,0,3)(0,0,1)_5 | 0.7241 | 0 | meilleur AIC train parmi les SARIMA | True |
| SARIMAX | SARIMAX(1,0,3) + r_SPY(t-1) | 0.7231 | 1 |  | False |
| SARIMAX | SARIMAX(0,0,0) + r_SPY(t-1) | 0.7155 | 0 |  | True |
| Référence | Moyenne mobile 5 séances | 0.7785 | 0 |  | False |
| Référence | Moyenne mobile 20 séances | 0.7206 | 0 |  | False |
| Référence | Moyenne mobile 60 séances | 0.7130 | 0 |  | True |

Modèles retenus et gelés (`reports/d03_selection.json`, écrit avant toute prévision du test) :

- **ARIMA** : ARIMA(1,0,3)
- **SARIMA** : SARIMA(1,0,3)(0,0,1)_5
- **SARIMAX** : SARIMAX(0,0,0) + r_SPY(t-1)
- **Moyenne mobile** : 60 séances

Métriques de validation de tous les candidats : `reports/d03_validation_metrics.csv`.

## 4. Résultats sur le test final (2025-01 → 2026-10)

| modele | n | MAE_% | RMSE_% | MAE_prix_USD | DA | R2_oos | ratio_RMSE_zero | DM_vs_zero | DM_p |
|---|---|---|---|---|---|---|---|---|---|
| Marche aléatoire (r̂=0) | 459 | 0.8077 | 1.0776 | 1.1858 | 0.5098 | -0.0030 | 1.0000 | — | — |
| Naïf (r̂=r_{t-1}) | 459 | 1.1907 | 1.5498 | 1.7517 | 0.4815 | -1.0745 | 1.4382 | 7.0350 | 0.0000 |
| Moyenne historique | 459 | 0.8082 | 1.0760 | 1.1869 | 0.5098 | 0.0000 | 0.9985 | -0.9553 | 0.3399 |
| Moyenne mobile 60 | 459 | 0.8209 | 1.0840 | 1.2056 | 0.4989 | -0.0150 | 1.0060 | 1.1029 | 0.2707 |
| ARIMA(1,0,3) | 459 | 0.8052 | 1.0875 | 1.1834 | 0.5251 | -0.0216 | 1.0092 | 0.9963 | 0.3196 |
| SARIMA(1,0,3)(0,0,1)_5 | 459 | 0.8067 | 1.0880 | 1.1860 | 0.5229 | -0.0225 | 1.0097 | 0.9699 | 0.3326 |
| SARIMAX(0,0,0) + r_SPY(t-1) | 459 | 0.8062 | 1.0740 | 1.1842 | 0.5185 | 0.0037 | 0.9966 | -0.4299 | 0.6675 |

Lectures :

- La plus faible RMSE est celle de **SARIMAX(0,0,0) + r_SPY(t-1)**.
- Modèles économétriques dont la RMSE est inférieure à celle de la marche aléatoire : SARIMAX. Aucun de ces écarts n'est significatif à 5 % (Diebold-Mariano).
- Exactitude directionnelle : la référence « toujours hausse » obtient 51.0 % (part des séances haussières). Une DA de 50 % ne veut donc rien dire. Il faut la comparer à ce chiffre.
- R2_oos < 0 : le modèle prévoit moins bien que la simple moyenne historique des rendements.
- ratio_RMSE_zero et DM_vs_zero : un ratio < 1 et une statistique DM négative avec DM_p < 0,05 indiquent une amélioration significative par rapport à la marche aléatoire.

Ré-estimations sur le test : ARIMA 22 (1 non-convergence(s)), SARIMA 22 (2 non-convergence(s)), SARIMAX 22 (0 non-convergence(s)).

Toutes les prévisions (validation et test) sont dans `reports/d03_predictions.csv`. Ce fichier sert d'entrée à la comparaison D05 et aux stratégies D06.

## 5. Stabilité de l'avantage prédictif (variante B05)

Ratio RMSE / marche aléatoire par sous-période (< 1 : le modèle fait mieux), et nombre de sous-périodes (sur 4) où le modèle bat la marche aléatoire et la moyenne historique :

| modele | ratio_T1 | ratio_T2 | ratio_T3 | ratio_T4 | sous_periodes_battant_zero | sous_periodes_battant_moyenne |
|---|---|---|---|---|---|---|
| Marche aléatoire (r̂=0) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 1 |
| Naïf (r̂=r_{t-1}) | 1.5176 | 1.3420 | 1.3933 | 1.4479 | 0 | 0 |
| Moyenne historique | 1.0009 | 0.9964 | 0.9999 | 0.9952 | 3 | 0 |
| Moyenne mobile 60 | 1.0149 | 0.9991 | 1.0102 | 0.9931 | 2 | 1 |
| ARIMA | 0.9986 | 1.0054 | 1.0251 | 1.0126 | 1 | 1 |
| SARIMA | 1.0006 | 1.0016 | 1.0233 | 1.0173 | 0 | 1 |
| SARIMAX | 0.9975 | 0.9983 | 0.9890 | 1.0020 | 3 | 2 |

Détail des métriques par sous-période : `reports/d03_test_subperiods.csv`. Figures : `figures/d03_stabilite.png` et `figures/d03_test_erreurs_cumulees.png`.

Conclusion sur la stabilité : aucun modèle économétrique ne bat la marche aléatoire dans les 4 sous-périodes. Quand un avantage apparaît, il est local et ne se maintient pas.

## 6. Diagnostics des modèles retenus (estimés sur le train)

### 6.1 Coefficients (unités : rendement en %)

| modele | parametre | coef | std_err | z | p_value |
|---|---|---|---|---|---|
| ARIMA(1,0,3) | intercept | 0.0022 | 0.0008 | 2.8276 | 0.0047 |
| ARIMA(1,0,3) | ar.L1 | 0.9511 | 0.0129 | 73.6719 | 0.0000 |
| ARIMA(1,0,3) | ma.L1 | -1.0683 | 0.0209 | -51.1794 | 0.0000 |
| ARIMA(1,0,3) | ma.L2 | 0.2230 | 0.0179 | 12.4392 | 0.0000 |
| ARIMA(1,0,3) | ma.L3 | -0.1389 | 0.0135 | -10.2620 | 0.0000 |
| ARIMA(1,0,3) | sigma2 | 1.4639 | 0.0302 | 48.5032 | 0.0000 |
| SARIMA(1,0,3)(0,0,1)_5 | intercept | 0.0028 | 0.0010 | 2.8271 | 0.0047 |
| SARIMA(1,0,3)(0,0,1)_5 | ar.L1 | 0.9377 | 0.0140 | 67.1495 | 0.0000 |
| SARIMA(1,0,3)(0,0,1)_5 | ma.L1 | -1.0489 | 0.0223 | -47.1152 | 0.0000 |
| SARIMA(1,0,3)(0,0,1)_5 | ma.L2 | 0.2113 | 0.0185 | 11.4460 | 0.0000 |
| SARIMA(1,0,3)(0,0,1)_5 | ma.L3 | -0.1426 | 0.0138 | -10.3681 | 0.0000 |
| SARIMA(1,0,3)(0,0,1)_5 | ma.S.L5 | 0.0741 | 0.0149 | 4.9651 | 0.0000 |
| SARIMA(1,0,3)(0,0,1)_5 | sigma2 | 1.4577 | 0.0303 | 48.1365 | 0.0000 |
| SARIMAX(0,0,0) + r_SPY(t-1) | intercept | 0.0500 | 0.0358 | 1.3952 | 0.1629 |
| SARIMAX(0,0,0) + r_SPY(t-1) | x1 | -0.1067 | 0.0112 | -9.5087 | 0.0000 |
| SARIMAX(0,0,0) + r_SPY(t-1) | sigma2 | 1.4990 | 0.0282 | 53.1719 | 0.0000 |

### 6.2 Racines (plus petit module, > 1 : stationnaire et inversible)

| modele | ar_min_modulus | ma_min_modulus |
|---|---|---|
| ARIMA(1,0,3) | 1.051 | 1.015 |
| SARIMA(1,0,3)(0,0,1)_5 | 1.066 | 1.019 |
| SARIMAX(0,0,0) + r_SPY(t-1) | inf | inf |

### 6.3 Tests sur les résidus

| modele | test | statistique | p_value | h0 | verdict |
|---|---|---|---|---|---|
| ARIMA(1,0,3) | Ljung-Box (résidus, 5 retards) | 6.6216 | 0.0101 | pas d'autocorrélation | H0 rejetée |
| ARIMA(1,0,3) | Ljung-Box (résidus, 10 retards) | 90.3936 | 0.0000 | pas d'autocorrélation | H0 rejetée |
| ARIMA(1,0,3) | Ljung-Box (résidus, 20 retards) | 124.8593 | 0.0000 | pas d'autocorrélation | H0 rejetée |
| ARIMA(1,0,3) | Ljung-Box (résidus², 20 retards) | 1904.7130 | 0.0000 | pas d'effet ARCH | H0 rejetée |
| ARIMA(1,0,3) | Jarque-Bera | 3108.4753 | 0.0000 | résidus gaussiens | H0 rejetée |
| ARIMA(1,0,3) | ARCH-LM (5 retards) | 399.2395 | 0.0000 | pas d'effet ARCH | H0 rejetée |
| SARIMA(1,0,3)(0,0,1)_5 | Ljung-Box (résidus, 5 retards) | 1.9247 | — | pas d'autocorrélation | non applicable |
| SARIMA(1,0,3)(0,0,1)_5 | Ljung-Box (résidus, 10 retards) | 84.7735 | 0.0000 | pas d'autocorrélation | H0 rejetée |
| SARIMA(1,0,3)(0,0,1)_5 | Ljung-Box (résidus, 20 retards) | 115.4752 | 0.0000 | pas d'autocorrélation | H0 rejetée |
| SARIMA(1,0,3)(0,0,1)_5 | Ljung-Box (résidus², 20 retards) | 1918.9471 | 0.0000 | pas d'effet ARCH | H0 rejetée |
| SARIMA(1,0,3)(0,0,1)_5 | Jarque-Bera | 2842.6945 | 0.0000 | résidus gaussiens | H0 rejetée |
| SARIMA(1,0,3)(0,0,1)_5 | ARCH-LM (5 retards) | 393.7591 | 0.0000 | pas d'effet ARCH | H0 rejetée |
| SARIMAX(0,0,0) + r_SPY(t-1) | Ljung-Box (résidus, 5 retards) | 21.4208 | 0.0007 | pas d'autocorrélation | H0 rejetée |
| SARIMAX(0,0,0) + r_SPY(t-1) | Ljung-Box (résidus, 10 retards) | 129.6345 | 0.0000 | pas d'autocorrélation | H0 rejetée |
| SARIMAX(0,0,0) + r_SPY(t-1) | Ljung-Box (résidus, 20 retards) | 173.2688 | 0.0000 | pas d'autocorrélation | H0 rejetée |
| SARIMAX(0,0,0) + r_SPY(t-1) | Ljung-Box (résidus², 20 retards) | 2051.4058 | 0.0000 | pas d'effet ARCH | H0 rejetée |
| SARIMAX(0,0,0) + r_SPY(t-1) | Jarque-Bera | 3640.5421 | 0.0000 | résidus gaussiens | H0 rejetée |
| SARIMAX(0,0,0) + r_SPY(t-1) | ARCH-LM (5 retards) | 411.6631 | 0.0000 | pas d'effet ARCH | H0 rejetée |

Lecture : les résidus ne sont pas gaussiens (queues épaisses) et ont un fort effet ARCH. Le test de Ljung-Box sur les résidus rejette encore aux retards 10 et 20. Ce test suppose une variance constante et rejette trop souvent en présence d'effet ARCH : ce rejet ne justifie donc pas d'ajouter des retards ARMA. Leur variance conditionnelle n'est pas constante. Les intervalles de prévision d'un ARIMA à variance constante sont donc mal calibrés. C'est le point de départ du livrable D04 (GARCH, VaR, ES). Voir `figures/d03_residus.png`.

## 7. Réponse provisoire à la question scientifique

Sur XLV, en prévision ponctuelle du rendement à une séance :

1. Les critères d'information du train trouvent une structure ARMA et un effet retardé de SPY. Ces deux résultats disparaissent quand on retire le krach de 2020 (section 2.3).
2. Hors échantillon, le meilleur modèle économétrique (SARIMAX(0,0,0) + r_SPY(t-1)) a un ratio RMSE / marche aléatoire de 0.9966, soit un gain de 0.34 %. Ce gain n'est pas significatif (Diebold-Mariano p = 0.67), et le modèle ne bat la marche aléatoire que dans 3 sous-périodes sur 4. Le gain n'est donc pas stable. La moyenne historique fait presque aussi bien (ratio 0.9985).
3. La SARIMA n'améliore pas l'ARIMA hors échantillon (ratio 1.0097 contre 1.0092). La saisonnalité hebdomadaire détectée sur le train ne sert donc pas à prévoir.
4. L'information économétrique utile est surtout dans la **variance** (effet ARCH très net), pas dans la moyenne conditionnelle.
5. Ces conclusions portent sur la précision statistique. L'intérêt économique, après coûts, sera mesuré en D06 à partir de `d03_predictions.csv`.

_Durée d'exécution : 43 s._
