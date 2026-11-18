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
python -m pytest                  # 64 tests (46 D02 + 18 D03), sans réseau
```

## Protocole

- **Cible** : rendement log journalier de XLV (prix ajusté). Horizon : 1 séance. La prévision de r_t est faite à la
  clôture de t-1.
- **Train 2018-2022** : stationnarité (ADF/KPSS), ACF/PACF, effet jour de la semaine, grille AIC/BIC des ARIMA(p,0,q)
  avec p, q ≤ 3, et composantes saisonnières hebdomadaires (s = 5).
- **Validation 2023-2024** : prévisions walk-forward. On choisit l'ordre ARIMA, la variante SARIMAX et la fenêtre de
  la moyenne mobile. Le choix est **gelé dans `d03_selection.json` avant toute prévision du test**.
- **Test 2025-01 → 2026-10** : une seule passe. Résultats globaux, puis par sous-période T1–T4. Test de
  Diebold-Mariano contre la marche aléatoire.
- **Walk-forward** : la fenêtre grandit depuis janvier 2018. Les paramètres sont ré-estimés toutes les 21 séances, et
  le filtre de Kalman intègre une nouvelle observation chaque jour. Les tests vérifient qu'aucune information future
  n'est utilisée (`test_walk_forward_no_lookahead`).

## Résultats (snapshot du 16/11/2026)

Modèles retenus : **ARIMA(1,0,3)**, **SARIMA(1,0,3)(0,0,1)₅**, **SARIMAX(0,0,0) + r_SPY(t-1)**, et une moyenne mobile
sur 60 séances.

| Test 2025-2026 | RMSE (%) | ratio / marche aléatoire | DA | Diebold-Mariano p |
|---|---|---|---|---|
| Marche aléatoire (r̂ = 0) | 1.0776 | 1.0000 | 51.0 % | — |
| Moyenne historique | 1.0760 | 0.9985 | 51.0 % | 0.34 |
| ARIMA(1,0,3) | 1.0875 | 1.0092 | 52.5 % | 0.32 |
| SARIMA(1,0,3)(0,0,1)₅ | 1.0880 | 1.0097 | 52.3 % | 0.33 |
| SARIMAX + r_SPY(t-1) | 1.0740 | 0.9966 | 51.9 % | 0.67 |

À retenir :

1. **La structure ARMA du train vient du krach de 2020.** Si on retire la période du 15/02 au 30/06/2020, soit moins
   de 8 % du train, l'autocorrélation d'ordre 1 passe de −0,13 à +0,03 et le gain d'AIC de l'ARIMA tombe de 39 à 1,7.
   L'effet de SPY retardé disparaît aussi.
2. **Hors échantillon, aucun modèle économétrique ne bat significativement la marche aléatoire.** Le meilleur
   (SARIMAX) gagne 0,3 % de RMSE (p = 0,67), et seulement dans 3 sous-périodes sur 4. L'ARIMA et la SARIMA font moins
   bien que la marche aléatoire.
3. La saisonnalité hebdomadaire est significative sur le train (test LR, p ≈ 0,02), mais elle n'apporte rien hors
   échantillon.
4. Les résidus ont de fortes queues épaisses et un effet ARCH très marqué (ARCH-LM p < 0,001). L'information
   exploitable est dans la **variance**, ce qui prépare le livrable D04 (GARCH, VaR, ES).

Ces résultats confirment l'hypothèse de la variante B05 en prévision statistique : sur un secteur défensif, la
complexité économétrique n'apporte pas d'avantage stable. L'effet économique, après coûts, sera mesuré en D06.

## Limites

- Une seule cible (le rendement à 1 séance) et une seule variable externe (r^SPY retardé). On n'a pas utilisé de
  calendrier d'événements ni de données macroéconomiques.
- Le test de Ljung-Box rejette encore sur les résidus. Ce test suppose une variance constante et rejette trop souvent
  en présence d'effet ARCH. On n'a donc pas ajouté de retards ARMA, ce qui aurait été du sur-ajustement.
- 1 à 2 ré-estimations sur 22 n'ont pas formellement convergé (ARIMA/SARIMA, voir le rapport). Les paramètres obtenus
  restent stationnaires et inversibles, grâce aux contraintes imposées.
