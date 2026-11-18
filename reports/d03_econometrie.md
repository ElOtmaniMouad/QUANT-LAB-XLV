# D03 — Références simples et modèles économétriques (XLV)

> Fichier généré par `python -m src.econometrics`. Ne pas modifier à la main : relancer la commande.

## 1. Cadre

- Snapshot D02 : `data/snapshots/XLV_SPY_1d_20261116.csv` (SHA-256 `d80d6249f3dcebd0…`, vérifié au chargement)
- Cible : rendement log journalier de XLV, r_t = ln(P_t / P_{t-1}), P = prix ajusté (dividendes et fractionnements).
- Horizon : 1 séance. La prévision de r_t est calculée à la clôture de t-1 avec l'information disponible à cette clôture uniquement.
- Variable externe de SARIMAX : x_t = r^SPY_{t-1}, le rendement de SPY de la **veille**. Pour prévoir r_t à la clôture de t-1, on connaît déjà r^SPY_{t-1} : aucune valeur future de SPY n'est utilisée. Le rendement de SPY du jour t, qui n'est connu qu'à la clôture de t, n'entre jamais dans le modèle. En walk-forward, la prévision de la séance t ne lit que les lignes ≤ t de x et les lignes ≤ t-1 de y. Les tests `test_dataset_alignment` et `test_walk_forward_no_lookahead` le vérifient.
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

### 2.2 Grille ARIMA(p, 0, q) et diagnostics — 8 meilleurs AIC

Pour chaque ordre, on calcule aussi sur le train les diagnostics qui décident de l'admissibilité (section 3) : test de Ljung-Box robuste à l'effet ARCH sur les résidus (10 retards), plus petit module des racines AR et MA, et plus petite distance entre une racine AR et une racine MA (compensation).

| p | q | aic | bic | d_aic | LB_robuste_p | ar_min_modulus | ma_min_modulus | root_cancellation | admissible | motif |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 3995.93 | 4031.90 | 0.00 | 0.34 | 1.07 | 1.20 | 0.15 | True | — |
| 2 | 2 | 3999.57 | 4030.39 | 3.63 | 0.38 | 1.06 | 1.15 | 0.11 | True | — |
| 1 | 3 | 4065.05 | 4095.87 | 69.11 | 0.00 | 1.05 | 1.02 | 0.04 | False | racines AR/MA qui se compensent; résidus autocorrélés (LB robuste) |
| 2 | 3 | 4066.07 | 4102.03 | 70.14 | 0.00 | 1.05 | 1.02 | 0.03 | False | racines AR/MA qui se compensent; résidus autocorrélés (LB robuste) |
| 3 | 3 | 4066.22 | 4107.33 | 70.29 | 0.00 | 1.04 | 1.01 | 0.03 | False | racines AR/MA qui se compensent; résidus autocorrélés (LB robuste) |
| 3 | 1 | 4066.26 | 4097.09 | 70.33 | 0.00 | 1.05 | 1.01 | 0.03 | False | racines AR/MA qui se compensent; résidus autocorrélés (LB robuste) |
| 0 | 3 | 4072.59 | 4098.28 | 76.66 | 0.00 | inf | 2.27 | inf | False | résidus autocorrélés (LB robuste) |
| 1 | 2 | 4074.73 | 4100.42 | 78.80 | 0.00 | 4.14 | 3.13 | 4.65 | False | résidus autocorrélés (LB robuste) |

Grille complète : `reports/d03_order_grid.csv`. Le bruit blanc ARIMA(0,0,0) a un ΔAIC de 108.2. Les critères voient donc une dépendance linéaire dans le train. Seuls 2 ordres sur 16 sont admissibles : ARIMA(3,0,2), ARIMA(2,0,2). Les autres gardent une autocorrélation dans leurs résidus, même avec le test robuste, et plusieurs ont des racines AR et MA qui se compensent presque : leurs paramètres sont redondants.

### 2.3 D'où vient cette structure ? Robustesse à l'épisode Covid

| echantillon | n | acf_1 | corr_r_SPY(t-1) | gain_AIC_ARIMA(3,0,2) |
|---|---|---|---|---|
| train complet 2018-2022 | 1259 | -0.1271 | -0.1178 | 108.2166 |
| train sans 15/02/2020-30/06/2020 | 1165 | 0.0327 | 0.0240 | 2.4452 |
| 2018-2019 seulement | 503 | 0.0367 | 0.0039 | -0.0318 |
| 2021-2022 seulement | 503 | 0.0328 | 0.0437 | -2.5462 |

Sans la période 2020-02-15 → 2020-06-30 (moins de 8 % du train), l'autocorrélation d'ordre 1 passe de -0.127 à +0.033, la corrélation avec r^SPY(t-1) de -0.118 à +0.024, et le gain d'AIC de l'ARIMA retenu de 108.2 à 2.4. **La dépendance linéaire du train vient presque entièrement du krach et du rebond de 2020.** En dehors de cet épisode, le rendement de XLV se comporte comme un bruit blanc. Les modèles ne sont pas ré-estimés sans 2020 : ce choix serait guidé par le résultat. Cette table sert seulement à interpréter les résultats hors échantillon.

### 2.4 Saisonnalité

Choix de la période saisonnière s = 5 :
- les données sont des **séances de bourse**, pas des jours calendaires : une semaine compte 5 séances ;
- s = 12 correspond à une saison annuelle en données **mensuelles**. En journalier, cela voudrait dire un cycle de 12 séances, qui ne correspond à aucun phénomène de marché connu ;
- un cycle mensuel (~21 séances) ou annuel (~252 séances) serait possible en théorie. Mais un SARIMA avec s = 252 coûte très cher à estimer, et il n'existe pas de mécanisme connu qui produirait ces cycles dans les rendements de XLV. Ces effets de calendrier (fin de mois, janvier…) se testent plus simplement avec des indicatrices, si besoin dans D05 ;
- le seul cycle plausible à priori est donc l'effet jour de la semaine, qu'on teste directement :

- Effet jour de la semaine (régression sur indicatrices, écarts-types robustes) : F = 0.92, p = 0.451. Rendements moyens (%) : lundi -0.085, mardi +0.059, mercredi +0.109, jeudi +0.026, vendredi +0.112.
- Composantes saisonnières hebdomadaires (s = 5) ajoutées à l'ARIMA retenu :

| candidat | AIC_train | BIC_train | LR_vs_ARIMA | LR_p | LB_robuste_p | admissible | motif |
|---|---|---|---|---|---|---|---|
| SARIMA(3,0,2)(1,0,0)_5 | 3993.587 | 4034.692 | 4.344 | 0.037 | 0.343 | True | — |
| SARIMA(3,0,2)(0,0,1)_5 | 3993.079 | 4034.183 | 4.852 | 0.028 | 0.354 | True | — |
| SARIMA(3,0,2)(1,0,1)_5 | 3990.927 | 4037.170 | 9.004 | 0.011 | 0.248 | False | racines AR/MA qui se compensent |

Conclusion : la saisonnalité hebdomadaire est **pertinente au sens statistique** sur le train (AIC plus bas et test LR significatif à 5 %). Le SARIMA est quand même évalué hors échantillon, pour vérifier si ce résultat tient sur des données nouvelles.

## 3. Sélection sur la validation (2023-2024)

Un modèle n'est pas choisi seulement parce que sa RMSE est faible. La règle comporte deux étapes, fixées dans `configs/models.yaml` (section `selection`) :

1. **Admissibilité, sur le train.** Un candidat est écarté si son estimation ne converge pas ; si une racine AR ou MA a un module inférieur à 1.01 ; si une racine AR et une racine MA sont à moins de 0.05 l'une de l'autre (elles se compensent) ; ou si le Ljung-Box robuste (10 retards) rejette l'absence d'autocorrélation des résidus au seuil de 0.05.
2. **Précision, sur la validation.** Parmi les candidats admissibles (pour l'ARIMA : les 3 meilleurs AIC et le meilleur BIC, sans le bruit blanc), on retient celui dont la RMSE walk-forward est la plus faible. Si aucun candidat d'une famille n'est admissible, on le signale et on garde le meilleur RMSE.

On utilise le Ljung-Box **robuste** (Lobato, Nankervis et Savin, 2001), pas le classique. Le test classique suppose une variance constante. Sur des rendements avec un fort effet ARCH, il rejette presque toujours, même quand les résidus ne sont pas autocorrélés : il éliminerait tous les modèles. Le test classique reste affiché en section 6.3 pour comparaison. Le jeu de test 2025-2026 n'intervient dans aucun choix.

| famille | candidat | AIC_train | LB_robuste_p | racine_min | compensation | admissible | motif | RMSE_val_% | note | retenu |
|---|---|---|---|---|---|---|---|---|---|---|
| ARIMA | ARIMA(3,0,2) | 3995.9313 | 0.3387 | 1.0713 | 0.1469 | True | — | 0.7278 |  | True |
| ARIMA | ARIMA(2,0,2) | 3999.5656 | 0.3845 | 1.0621 | 0.1077 | True | — | 0.7282 |  | False |
| SARIMA | SARIMA(3,0,2)(0,0,1)_5 | 3993.0788 | 0.3540 | 1.0765 | 0.1811 | True | — | 0.7280 | meilleur AIC train parmi les SARIMA admissibles | True |
| SARIMAX | SARIMAX(3,0,2) + r_SPY(t-1) | 3997.7975 | 0.3405 | 1.0715 | 0.1484 | True | — | 0.7284 |  | True |
| SARIMAX | SARIMAX(0,0,0) + r_SPY(t-1) | 4088.5400 | 0.0012 | inf | inf | False | résidus autocorrélés (LB robuste) | 0.7155 |  | False |
| Référence | Moyenne mobile 5 séances | — | — | — | — | — | — | 0.7785 | pas de résidus : pas de diagnostic | False |
| Référence | Moyenne mobile 20 séances | — | — | — | — | — | — | 0.7206 | pas de résidus : pas de diagnostic | False |
| Référence | Moyenne mobile 60 séances | — | — | — | — | — | — | 0.7130 | pas de résidus : pas de diagnostic | True |

À noter : SARIMAX(0,0,0) + r_SPY(t-1) a une RMSE de validation plus faible (0.7155 %) que le modèle retenu (0.7284 %), mais il est écarté par les diagnostics (résidus autocorrélés (LB robuste)). Cela montre que la règle ne choisit pas seulement sur la RMSE.

Modèles retenus et gelés (`reports/d03_selection.json`, écrit avant toute prévision du test) :

- **ARIMA** : ARIMA(3,0,2)
- **SARIMA** : SARIMA(3,0,2)(0,0,1)_5
- **SARIMAX** : SARIMAX(3,0,2) + r_SPY(t-1)
- **Moyenne mobile** : 60 séances

Métriques de validation de tous les candidats : `reports/d03_validation_metrics.csv`.

## 4. Résultats sur le test final (2025-01 → 2026-10)

| modele | n | MAE_% | RMSE_% | MAE_prix_USD | DA | R2_oos | ratio_RMSE_zero | DM_vs_zero | DM_p |
|---|---|---|---|---|---|---|---|---|---|
| Marche aléatoire (r̂=0) | 459 | 0.8077 | 1.0776 | 1.1858 | 0.5098 | -0.0030 | 1.0000 | — | — |
| Naïf (r̂=r_{t-1}) | 459 | 1.1907 | 1.5498 | 1.7517 | 0.4815 | -1.0745 | 1.4382 | 7.0350 | 0.0000 |
| Moyenne historique | 459 | 0.8082 | 1.0760 | 1.1869 | 0.5098 | 0.0000 | 0.9985 | -0.9553 | 0.3399 |
| Moyenne mobile 60 | 459 | 0.8209 | 1.0840 | 1.2056 | 0.4989 | -0.0150 | 1.0060 | 1.1029 | 0.2707 |
| ARIMA(3,0,2) | 459 | 0.8137 | 1.0930 | 1.1955 | 0.5142 | -0.0319 | 1.0143 | 1.3549 | 0.1761 |
| SARIMA(3,0,2)(0,0,1)_5 | 459 | 0.8117 | 1.0944 | 1.1925 | 0.5185 | -0.0345 | 1.0156 | 1.4384 | 0.1510 |
| SARIMAX(3,0,2) + r_SPY(t-1) | 459 | 0.8141 | 1.0931 | 1.1961 | 0.5076 | -0.0321 | 1.0144 | 1.3647 | 0.1730 |

Lectures :

- La plus faible RMSE est celle de **Moyenne historique**.
- **Aucun modèle économétrique ne bat la marche aléatoire** en RMSE sur le test.
- Exactitude directionnelle : la référence « toujours hausse » obtient 51.0 % (part des séances haussières). Une DA de 50 % ne veut donc rien dire. Il faut la comparer à ce chiffre.
- R2_oos < 0 : le modèle prévoit moins bien que la simple moyenne historique des rendements.
- ratio_RMSE_zero et DM_vs_zero : un ratio < 1 et une statistique DM négative avec DM_p < 0,05 indiquent une amélioration significative par rapport à la marche aléatoire.

Ré-estimations sur le test : ARIMA 22 (0 non-convergence(s)), SARIMA 22 (0 non-convergence(s)), SARIMAX 22 (0 non-convergence(s)).

Toutes les prévisions (validation et test) sont dans `reports/d03_predictions.csv`. Ce fichier sert d'entrée à la comparaison D05 et aux stratégies D06.

## 5. Stabilité de l'avantage prédictif (variante B05)

Ratio RMSE / marche aléatoire par sous-période (< 1 : le modèle fait mieux), et nombre de sous-périodes (sur 4) où le modèle bat la marche aléatoire et la moyenne historique :

| modele | ratio_T1 | ratio_T2 | ratio_T3 | ratio_T4 | sous_periodes_battant_zero | sous_periodes_battant_moyenne |
|---|---|---|---|---|---|---|
| Marche aléatoire (r̂=0) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 1 |
| Naïf (r̂=r_{t-1}) | 1.5176 | 1.3420 | 1.3933 | 1.4479 | 0 | 0 |
| Moyenne historique | 1.0009 | 0.9964 | 0.9999 | 0.9952 | 3 | 0 |
| Moyenne mobile 60 | 1.0149 | 0.9991 | 1.0102 | 0.9931 | 2 | 1 |
| ARIMA | 1.0228 | 1.0061 | 1.0149 | 1.0079 | 0 | 0 |
| SARIMA | 1.0241 | 1.0091 | 1.0180 | 1.0053 | 0 | 0 |
| SARIMAX | 1.0234 | 1.0062 | 1.0122 | 1.0102 | 0 | 0 |

Détail des métriques par sous-période : `reports/d03_test_subperiods.csv`. Figures : `figures/d03_stabilite.png` et `figures/d03_test_erreurs_cumulees.png`.

Conclusion sur la stabilité : aucun modèle économétrique ne bat la marche aléatoire dans les 4 sous-périodes. Quand un avantage apparaît, il est local et ne se maintient pas.

## 6. Diagnostics des modèles retenus (estimés sur le train)

### 6.1 Coefficients (unités : rendement en %)

| modele | parametre | coef | std_err | z | p_value |
|---|---|---|---|---|---|
| ARIMA(3,0,2) | intercept | 0.1511 | 0.1135 | 1.3313 | 0.1831 |
| ARIMA(3,0,2) | ar.L1 | -1.6369 | 0.0444 | -36.8967 | 0.0000 |
| ARIMA(3,0,2) | ar.L2 | -0.7058 | 0.0602 | -11.7187 | 0.0000 |
| ARIMA(3,0,2) | ar.L3 | 0.0832 | 0.0235 | 3.5477 | 0.0004 |
| ARIMA(3,0,2) | ma.L1 | 1.5850 | 0.0392 | 40.4454 | 0.0000 |
| ARIMA(3,0,2) | ma.L2 | 0.6958 | 0.0369 | 18.8757 | 0.0000 |
| ARIMA(3,0,2) | sigma2 | 1.3833 | 0.0323 | 42.7674 | 0.0000 |
| SARIMA(3,0,2)(0,0,1)_5 | intercept | 0.1475 | 0.1049 | 1.4065 | 0.1596 |
| SARIMA(3,0,2)(0,0,1)_5 | ar.L1 | -1.6120 | 0.0496 | -32.5023 | 0.0000 |
| SARIMA(3,0,2)(0,0,1)_5 | ar.L2 | -0.6742 | 0.0663 | -10.1708 | 0.0000 |
| SARIMA(3,0,2)(0,0,1)_5 | ar.L3 | 0.0946 | 0.0257 | 3.6795 | 0.0002 |
| SARIMA(3,0,2)(0,0,1)_5 | ma.L1 | 1.5600 | 0.0447 | 34.8795 | 0.0000 |
| SARIMA(3,0,2)(0,0,1)_5 | ma.L2 | 0.6652 | 0.0429 | 15.4945 | 0.0000 |
| SARIMA(3,0,2)(0,0,1)_5 | ma.S.L5 | -0.0740 | 0.0231 | -3.2033 | 0.0014 |
| SARIMA(3,0,2)(0,0,1)_5 | sigma2 | 1.3779 | 0.0329 | 41.9065 | 0.0000 |
| SARIMAX(3,0,2) + r_SPY(t-1) | intercept | 0.1514 | 0.1133 | 1.3363 | 0.1815 |
| SARIMAX(3,0,2) + r_SPY(t-1) | x1 | 0.0171 | 0.0367 | 0.4665 | 0.6409 |
| SARIMAX(3,0,2) + r_SPY(t-1) | ar.L1 | -1.6507 | 0.0567 | -29.1162 | 0.0000 |
| SARIMAX(3,0,2) + r_SPY(t-1) | ar.L2 | -0.7304 | 0.0857 | -8.5256 | 0.0000 |
| SARIMAX(3,0,2) + r_SPY(t-1) | ar.L3 | 0.0707 | 0.0396 | 1.7846 | 0.0743 |
| SARIMAX(3,0,2) + r_SPY(t-1) | ma.L1 | 1.5830 | 0.0401 | 39.4452 | 0.0000 |
| SARIMAX(3,0,2) + r_SPY(t-1) | ma.L2 | 0.6941 | 0.0380 | 18.2757 | 0.0000 |
| SARIMAX(3,0,2) + r_SPY(t-1) | sigma2 | 1.3832 | 0.0323 | 42.7662 | 0.0000 |

### 6.2 Racines (plus petit module, > 1 : stationnaire et inversible)

| modele | ar_min_modulus | ma_min_modulus | root_cancellation |
|---|---|---|---|
| ARIMA(3,0,2) | 1.071 | 1.199 | 0.147 |
| SARIMA(3,0,2)(0,0,1)_5 | 1.077 | 1.226 | 0.181 |
| SARIMAX(3,0,2) + r_SPY(t-1) | 1.071 | 1.200 | 0.148 |

### 6.3 Tests sur les résidus

| modele | test | statistique | p_value | h0 | verdict |
|---|---|---|---|---|---|
| ARIMA(3,0,2) | Ljung-Box (résidus, 5 retards) | 4.8944 | — | pas d'autocorrélation | non applicable |
| ARIMA(3,0,2) | Ljung-Box (résidus, 10 retards) | 17.5445 | 0.0036 | pas d'autocorrélation | H0 rejetée |
| ARIMA(3,0,2) | Ljung-Box (résidus, 20 retards) | 31.9376 | 0.0066 | pas d'autocorrélation | H0 rejetée |
| ARIMA(3,0,2) | Ljung-Box robuste ARCH (5 retards) | 1.3995 | — | pas d'autocorrélation | non applicable |
| ARIMA(3,0,2) | Ljung-Box robuste ARCH (10 retards) | 5.6795 | 0.3387 | pas d'autocorrélation | H0 non rejetée |
| ARIMA(3,0,2) | Ljung-Box robuste ARCH (20 retards) | 13.1391 | 0.5916 | pas d'autocorrélation | H0 non rejetée |
| ARIMA(3,0,2) | Ljung-Box (résidus², 20 retards) | 1838.5335 | 0.0000 | pas d'effet ARCH | H0 rejetée |
| ARIMA(3,0,2) | Jarque-Bera | 1375.2576 | 0.0000 | résidus gaussiens | H0 rejetée |
| ARIMA(3,0,2) | ARCH-LM (5 retards) | 369.9787 | 0.0000 | pas d'effet ARCH | H0 rejetée |
| SARIMA(3,0,2)(0,0,1)_5 | Ljung-Box (résidus, 5 retards) | 0.6550 | — | pas d'autocorrélation | non applicable |
| SARIMA(3,0,2)(0,0,1)_5 | Ljung-Box (résidus, 10 retards) | 13.1368 | 0.0106 | pas d'autocorrélation | H0 rejetée |
| SARIMA(3,0,2)(0,0,1)_5 | Ljung-Box (résidus, 20 retards) | 26.2494 | 0.0241 | pas d'autocorrélation | H0 rejetée |
| SARIMA(3,0,2)(0,0,1)_5 | Ljung-Box robuste ARCH (5 retards) | 0.1754 | — | pas d'autocorrélation | non applicable |
| SARIMA(3,0,2)(0,0,1)_5 | Ljung-Box robuste ARCH (10 retards) | 4.4049 | 0.3540 | pas d'autocorrélation | H0 non rejetée |
| SARIMA(3,0,2)(0,0,1)_5 | Ljung-Box robuste ARCH (20 retards) | 11.2802 | 0.6639 | pas d'autocorrélation | H0 non rejetée |
| SARIMA(3,0,2)(0,0,1)_5 | Ljung-Box (résidus², 20 retards) | 1819.9734 | 0.0000 | pas d'effet ARCH | H0 rejetée |
| SARIMA(3,0,2)(0,0,1)_5 | Jarque-Bera | 1349.3183 | 0.0000 | résidus gaussiens | H0 rejetée |
| SARIMA(3,0,2)(0,0,1)_5 | ARCH-LM (5 retards) | 369.2170 | 0.0000 | pas d'effet ARCH | H0 rejetée |
| SARIMAX(3,0,2) + r_SPY(t-1) | Ljung-Box (résidus, 5 retards) | 4.9520 | — | pas d'autocorrélation | non applicable |
| SARIMAX(3,0,2) + r_SPY(t-1) | Ljung-Box (résidus, 10 retards) | 17.4954 | 0.0037 | pas d'autocorrélation | H0 rejetée |
| SARIMAX(3,0,2) + r_SPY(t-1) | Ljung-Box (résidus, 20 retards) | 31.9222 | 0.0066 | pas d'autocorrélation | H0 rejetée |
| SARIMAX(3,0,2) + r_SPY(t-1) | Ljung-Box robuste ARCH (5 retards) | 1.4155 | — | pas d'autocorrélation | non applicable |
| SARIMAX(3,0,2) + r_SPY(t-1) | Ljung-Box robuste ARCH (10 retards) | 5.6616 | 0.3405 | pas d'autocorrélation | H0 non rejetée |
| SARIMAX(3,0,2) + r_SPY(t-1) | Ljung-Box robuste ARCH (20 retards) | 13.1607 | 0.5899 | pas d'autocorrélation | H0 non rejetée |
| SARIMAX(3,0,2) + r_SPY(t-1) | Ljung-Box (résidus², 20 retards) | 1836.3328 | 0.0000 | pas d'effet ARCH | H0 rejetée |
| SARIMAX(3,0,2) + r_SPY(t-1) | Jarque-Bera | 1379.7662 | 0.0000 | résidus gaussiens | H0 rejetée |
| SARIMAX(3,0,2) + r_SPY(t-1) | ARCH-LM (5 retards) | 370.4125 | 0.0000 | pas d'effet ARCH | H0 rejetée |

Lecture :
- **Autocorrélation** : le Ljung-Box classique rejette aux retards 10 et 20, mais le Ljung-Box robuste ne rejette pas. La dynamique de la moyenne est donc bien captée. Le rejet du test classique vient de l'hétéroscédasticité, pas d'une autocorrélation oubliée.
- **Normalité** : Jarque-Bera rejette nettement. Les résidus ont des queues épaisses (voir le QQ-plot).
- **Effet ARCH** : ARCH-LM et Ljung-Box sur les résidus² rejettent fortement. La variance conditionnelle n'est pas constante. Les intervalles de prévision d'un ARIMA à variance constante sont donc mal calibrés. C'est le point de départ du livrable D04 (GARCH, VaR, ES). Voir `figures/d03_residus.png`.

## 7. Réponse provisoire à la question scientifique

Sur XLV, en prévision ponctuelle du rendement à une séance :

1. Les critères d'information du train trouvent une structure ARMA et un effet retardé de SPY. Ces deux résultats disparaissent quand on retire le krach de 2020 (section 2.3).
2. Les modèles retenus passent les diagnostics : résidus non autocorrélés selon le test robuste, racines bien identifiées. Pourtant, hors échantillon, le meilleur d'entre eux (ARIMA(3,0,2)) a un ratio RMSE / marche aléatoire de 1.0143, soit une perte de 1.43 % (Diebold-Mariano p = 0.18). Il bat la marche aléatoire dans 0 sous-période(s) sur 4. Un bon ajustement sur le train ne garantit donc pas de bonnes prévisions : la structure captée est celle de 2020, qui ne se reproduit pas. La moyenne historique, sans aucun paramètre dynamique, fait mieux (ratio 0.9985).
3. La SARIMA n'améliore pas l'ARIMA hors échantillon (ratio 1.0156 contre 1.0143). La saisonnalité hebdomadaire détectée sur le train ne sert donc pas à prévoir.
4. L'information économétrique utile est surtout dans la **variance** (effet ARCH très net), pas dans la moyenne conditionnelle.
5. Ces conclusions portent sur la précision statistique. L'intérêt économique, après coûts, sera mesuré en D06 à partir de `d03_predictions.csv`.

_Durée d'exécution : 37 s._
