# D03 — Références simples et modèles économétriques

Binôme B05 · XLV (benchmark SPY). Le rapport complet est généré par le code :
[`reports/d03_econometrie.md`](reports/d03_econometrie.md). Il contient l'identification, la sélection, les
résultats du test, la stabilité et les diagnostics.

## Pièces du livrable

| Pièce | Fichier |
|---|---|
| Paramètres gelés (cible, grilles, walk-forward, sous-périodes) | [`configs/models.yaml`](configs/models.yaml) |
| Jeu de modélisation (cible r_t, variable externe r^SPY_{t-1}) | [`src/features.py`](src/features.py) |
| Références : marche aléatoire, naïf, moyenne historique, moyenne mobile | [`src/models/baseline.py`](src/models/baseline.py) |
| ARIMA, SARIMA, SARIMAX, walk-forward, diagnostics | [`src/models/arima.py`](src/models/arima.py) |
| Métriques (MAE, RMSE, DA, R²_oos, Diebold-Mariano) | [`src/evaluation.py`](src/evaluation.py) |
| Pipeline complet (identification → validation → gel → test) | [`src/econometrics.py`](src/econometrics.py) |
| Rapport | [`reports/d03_econometrie.md`](reports/d03_econometrie.md) |
| Sélection gelée avant le test | [`reports/d03_selection.json`](reports/d03_selection.json) |
| Prévisions validation + test (entrée de D05/D06) | [`reports/d03_predictions.csv`](reports/d03_predictions.csv) |
| Tableaux | `reports/d03_order_grid.csv`, `d03_validation_metrics.csv`, `d03_test_metrics.csv`, `d03_test_subperiods.csv`, `d03_coefficients.csv`, `d03_diagnostics.csv`, `d03_robustesse_covid.csv` |
| Figures | `reports/figures/d03_identification.png`, `d03_residus.png`, `d03_test_erreurs_cumulees.png`, `d03_stabilite.png` |
| Tests | [`tests/test_models.py`](tests/test_models.py) |

## Commandes

```bash
pip install -r requirements.txt   # ajoute statsmodels et scipy
python -m src.econometrics        # ~1 min, sans réseau, à partir du snapshot gelé D02
python -m pytest                  # 68 tests (46 D02 + 22 D03), sans réseau
```

## Protocole

- **Cible** : rendement log journalier de XLV (prix ajusté). Horizon : 1 séance. La prévision de r_t est faite à la
  clôture de t-1.
- **Train 2018-2022** : stationnarité (ADF/KPSS), ACF/PACF, effet jour de la semaine, grille AIC/BIC des ARIMA(p,0,q)
  avec p, q ≤ 3, et composantes saisonnières hebdomadaires (s = 5).
- **Admissibilité (train)** : un candidat n'est comparé que si ses diagnostics sont bons. Il faut que l'estimation
  converge, que les racines AR et MA soient hors du cercle unité, qu'aucune racine AR ne compense une racine MA, et
  que les résidus ne soient pas autocorrélés selon un **Ljung-Box robuste à l'effet ARCH**.
- **Validation 2023-2024** : parmi les candidats admissibles, on retient celui dont la RMSE walk-forward est la plus
  faible. On choisit ainsi l'ordre ARIMA, la variante SARIMAX et la fenêtre de la moyenne mobile. Le choix est
  **gelé dans `d03_selection.json` avant toute prévision du test**.
- **Test 2025-01 → 2026-10** : une seule passe. Résultats globaux, puis par sous-période T1–T4. Test de
  Diebold-Mariano contre la marche aléatoire.
- **Variable externe** : SARIMAX utilise r^SPY_{t-1}, le rendement de SPY de la veille, déjà connu au moment de
  la prévision. Le rendement de SPY du jour même n'est jamais utilisé.
- **Saisonnalité** : s = 5, une semaine de bourse. On n'utilise pas 12, qui correspond à des données mensuelles.
  La justification complète est dans le rapport, section 2.4.
- **Walk-forward** : la fenêtre grandit depuis janvier 2018. Les paramètres sont ré-estimés toutes les 21 séances, et
  le filtre de Kalman intègre une nouvelle observation chaque jour. Les tests vérifient qu'aucune information future
  n'est utilisée (`test_walk_forward_no_lookahead`).

## Résultats (snapshot du 16/11/2026)

Modèles retenus : **ARIMA(3,0,2)**, **SARIMA(3,0,2)(0,0,1)₅**, **SARIMAX(3,0,2) + r_SPY(t-1)**, et une moyenne mobile
sur 60 séances. Sur 16 ordres ARIMA, seuls ARIMA(3,0,2) et ARIMA(2,0,2) passent les diagnostics.

| Test 2025-2026 | RMSE (%) | ratio / marche aléatoire | DA | Diebold-Mariano p | sous-périodes gagnées |
|---|---|---|---|---|---|
| Marche aléatoire (r̂ = 0) | 1.0776 | 1.0000 | 51.0 % | — | — |
| Moyenne historique | 1.0760 | 0.9985 | 51.0 % | 0.34 | 3 / 4 |
| ARIMA(3,0,2) | 1.0930 | 1.0143 | 51.4 % | 0.18 | 0 / 4 |
| SARIMA(3,0,2)(0,0,1)₅ | 1.0944 | 1.0156 | 51.9 % | 0.15 | 0 / 4 |
| SARIMAX(3,0,2) + r_SPY(t-1) | 1.0931 | 1.0144 | 50.8 % | 0.17 | 0 / 4 |

À retenir :

1. **La structure ARMA du train vient du krach de 2020.** Si on retire la période du 15/02 au 30/06/2020, soit moins
   de 8 % du train, l'autocorrélation d'ordre 1 passe de −0,13 à +0,03 et le gain d'AIC de l'ARIMA tombe de 108 à 2,4.
   L'effet de SPY retardé disparaît aussi.
2. **Des modèles aux diagnostics corrects prévoient quand même moins bien que la marche aléatoire.** Les trois modèles
   économétriques retenus ont des résidus non autocorrélés, mais ils perdent environ 1,4 % de RMSE. Ils font moins
   bien que la marche aléatoire dans les 4 sous-périodes du test. Un bon ajustement sur le train ne garantit pas une
   bonne prévision.
3. La saisonnalité hebdomadaire est significative sur le train (test LR, p ≈ 0,03), mais elle n'apporte rien hors
   échantillon.
4. Les résidus ont de fortes queues épaisses et un effet ARCH très marqué (ARCH-LM p < 0,001). L'information
   exploitable est dans la **variance**, ce qui prépare le livrable D04 (GARCH, VaR, ES).

Ces résultats confirment l'hypothèse de la variante B05 en prévision statistique : sur un secteur défensif, la
complexité économétrique n'apporte pas d'avantage stable. L'effet économique, après coûts, sera mesuré en D06.

## Corrections après la revue du binôme (18/11/2026)

| Remarque | Correction |
|---|---|
| 1. Fuite temporelle | Le code utilisait déjà r^SPY_{t-1}. C'est maintenant écrit explicitement dans le rapport (section 1), et les tests le vérifient. |
| 2. Séparation des données | Déjà chronologique (2018-2022 / 2023-2024 / 2025-2026). Le test n'intervient dans aucun choix. |
| 3. Saisonnalité | Le choix de s = 5 est maintenant justifié dans le rapport, ainsi que le rejet de 12, 21 et 252 (section 2.4). |
| 4. SARIMAX et variables futures | La disponibilité de r^SPY_{t-1} au moment de la prévision est maintenant écrite dans le rapport. |
| 5. Diagnostic insuffisant | Les diagnostics font maintenant partie de la sélection : admissibilité (Ljung-Box robuste, racines, convergence) **avant** la comparaison des RMSE. |

Effet de la correction 5 : avec l'ancienne règle, fondée seulement sur la RMSE, on retenait ARIMA(1,0,3) et
SARIMAX(0,0,0) + r_SPY(t-1). Ces deux modèles sont maintenant écartés :
- ARIMA(1,0,3) a des racines AR et MA qui se compensent (distance 0,04) et des résidus autocorrélés (p = 0,0005) ;
- SARIMAX(0,0,0) a des résidus autocorrélés (p = 0,001).

Sur le test, l'ancien SARIMAX avait un ratio de 0,9966 (p = 0,67), c'est-à-dire un gain non significatif. Les modèles
admissibles font environ 1,4 % moins bien que la marche aléatoire. La conclusion ne change pas : aucun modèle
économétrique ne bat significativement la marche aléatoire. Elle est même plus nette.

## Limites

- Une seule cible (le rendement à 1 séance) et une seule variable externe (r^SPY retardé). On n'a pas utilisé de
  calendrier d'événements ni de données macroéconomiques.
- Les seuils d'admissibilité (module des racines ≥ 1,01, distance de compensation ≥ 0,05, seuil de 5 % pour le
  Ljung-Box robuste) sont des conventions. Ils ont été fixés dans `configs/models.yaml` après la revue, et ils
  s'appliquent sans exception à tous les candidats.
- Le Ljung-Box classique rejette encore sur les résidus des modèles retenus, mais pas la version robuste : ce rejet
  vient de l'effet ARCH. On n'a donc pas ajouté de retards ARMA, ce qui aurait été du sur-ajustement.
