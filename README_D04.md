# D04 — Module de risque

Binôme B05 · XLV (benchmark SPY). Le rapport complet est généré par le code :
[`reports/d04_risque.md`](reports/d04_risque.md). Il contient le cadre et les unités, l'effet ARCH, les GARCH estimés,
leurs diagnostics, la sélection, les backtests de la VaR et de l'ES, la stabilité et les conclusions.

## Pièces du livrable

Le cahier demande : ARCH-LM, GARCH(1,1), volatilité prévue, diagnostics, VaR/ES et tableaux de vérification.

| Pièce | Fichier |
|---|---|
| Paramètres gelés (snapshot, modèles, références, niveaux de VaR, walk-forward) | [`configs/risk.yaml`](configs/risk.yaml) |
| Test ARCH-LM (rendements et résidus de l'ARIMA D03) | [`reports/d04_arch_tests.csv`](reports/d04_arch_tests.csv) |
| GARCH(1,1)-N, GARCH(1,1)-t, GJR-GARCH(1,1)-t | [`src/models/garch.py`](src/models/garch.py) → [`reports/d04_garch_train.csv`](reports/d04_garch_train.csv), [`reports/d04_coefficients.csv`](reports/d04_coefficients.csv) |
| Diagnostics des résidus standardisés | [`reports/d04_diagnostics.csv`](reports/d04_diagnostics.csv), [`reports/d04_tests_lr.csv`](reports/d04_tests_lr.csv) |
| VaR, ES, références simples, backtests | [`src/risk.py`](src/risk.py) |
| Pipeline complet (effet ARCH → estimation → validation → gel → test) | [`src/risk_management.py`](src/risk_management.py) |
| Sélection gelée avant le test | [`reports/d04_selection.json`](reports/d04_selection.json) |
| Volatilité prévue, VaR et ES datées (validation + test, entrée de D06/D07) | [`reports/d04_risk_forecasts.csv`](reports/d04_risk_forecasts.csv) |
| Tableaux de vérification | `reports/d04_test_volatilite.csv`, `d04_test_var.csv`, `d04_test_es.csv`, `d04_test_subperiodes.csv`, `d04_validation_*.csv` |
| Figures | `reports/figures/d04_effet_arch.png`, `d04_garch_train.png`, `d04_volatilite_prevue.png`, `d04_var_backtest.png`, `d04_stabilite.png` |
| Tests | [`tests/test_risk.py`](tests/test_risk.py) |

## Commandes

Toutes les commandes se lancent depuis le dossier du projet, avec l'environnement activé.

```bash
git pull origin main
pip install -r requirements.txt     # D04 ajoute la bibliothèque arch
```

```bash
python -m pytest                    # résultat attendu : 91 passed (46 D02 + 22 D03 + 23 D04), ~4 s
```

```bash
python -m src.risk_management       # ~3 s ; régénère reports/d04_* et reports/figures/d04_*
```

Résultat attendu à l'écran : `Modèle GARCH retenu (...) : GARCH(1,1)-t`, puis `Rapport écrit : reports/d04_risque.md`.

Pour vérifier la reproductibilité, lancez `git status` : seul `reports/d04_selection.json` peut apparaître comme modifié,
car il contient la date d'exécution. Tous les CSV de résultats doivent être identiques.

## Unités

| Grandeur | Unité et convention |
|---|---|
| r_t | rendement log de XLV **en %** : 100 · ln(P_t / P_{t-1}), prix ajusté |
| σ_t | volatilité prévue de r_t, en % par séance (× √252 pour l'annualiser) |
| VaR_t | nombre **positif**, en % de la position longue : P(perte > VaR) = 1 − niveau, avec perte = −r_t |
| ES_t | perte moyenne au-delà de la VaR, en %, toujours ≥ VaR |
| VaR en USD | C · (1 − exp(−VaR/100)), avec C = 100 000 USD (capital du cahier) |
| Exception | la perte réalisée de la séance t dépasse la VaR prévue à la clôture de t-1 |

## Protocole

- **Snapshot figé** : `XLV_SPY_1d_20261116.csv`, le même que D02 et D03. Il est écrit dans `configs/risk.yaml`, donc
  un snapshot plus récent dans `data/snapshots/` ne change pas les résultats.
- **Train 2018-2022** : tests ARCH-LM et Ljung-Box sur r² et sur les résidus de l'ARIMA(3,0,2) de D03. Estimation de
  trois modèles à moyenne constante : GARCH(1,1) gaussien (exigé par le cahier), GARCH(1,1) Student-t, et GJR-GARCH(1,1)
  Student-t (asymétrie). Tests du rapport de vraisemblance pour la loi t et pour l'asymétrie.
- **Admissibilité (train)** : convergence, paramètres de variance ≥ 0, persistance α + β + γ/2 < 0,999, et plus d'effet
  ARCH significatif dans les résidus standardisés (ARCH-LM et Ljung-Box sur z²).
- **Validation 2023-2024** : volatilité prévue en walk-forward. Parmi les modèles admissibles, on retient celui qui a la
  plus petite perte **QLIKE** = ln σ̂² + r²/σ̂². Ce choix est **gelé dans `d04_selection.json` avant le test**.
- **Références simples**, sans paramètre choisi sur les données : EWMA RiskMetrics (λ = 0,94), loi normale à moyenne et
  écart-type constants (tout l'historique), simulation historique sur 250 séances. L'année 2017 sert à les amorcer.
- **Test 2025-01 → 2026-10**, une seule passe :
  - volatilité : QLIKE, test de Diebold-Mariano contre EWMA et contre la volatilité constante, régression de
    Mincer-Zarnowitz, et proxy de Parkinson (amplitude haut-bas) pour vérifier le classement ;
  - VaR 95 % et 99 % : nombre d'exceptions, tests de Kupiec et de Christoffersen, zone de Bâle, perte quantile ;
  - ES : ratio perte / ES, test de McNeil-Frey, perte FZ0 ;
  - stabilité sur les sous-périodes T1–T4 (expérience de la variante B05).
- **Walk-forward** : fenêtre croissante depuis 2018, paramètres ré-estimés toutes les 21 séances, variance
  conditionnelle mise à jour chaque jour. Les tests `test_garch_walk_forward_no_lookahead`,
  `test_ewma_recursion_and_no_lookahead` et `test_historical_var_es_values_and_no_lookahead` vérifient qu'aucune
  information future n'est utilisée.
- **Contrainte de position proposée pour D06** : fraction investie w = min(1, 2 % / (1 − exp(−VaR99/100))), pour que la
  VaR 99 % de la position ne dépasse pas 2 % du capital. Elle est calculée dans `d04_risk_forecasts.csv` mais n'est pas
  encore utilisée : la stratégie sera définie en D06.

## Résultats (snapshot du 16/11/2026)

Effet ARCH : 6 tests sur 6 significatifs (p < 0,001), y compris sur les résidus de l'ARIMA de D03. Les trois GARCH sont
admissibles ; leurs résidus standardisés n'ont plus d'effet ARCH (ARCH-LM p > 0,94). Modèle retenu sur la validation :
**GARCH(1,1)-t** (persistance 0,972, ν = 6,4).

| Test 2025-2026 | QLIKE | écart vs EWMA (DM p) | écart vs constante (DM p) | exceptions VaR 95 % / 99 % | zone 99 % |
|---|---|---|---|---|---|
| GARCH(1,1)-N | 1,145 | −0,029 (0,24) | −0,006 (0,88) | 18 / 6 | vert |
| **GARCH(1,1)-t** (retenu) | 1,145 | −0,029 (0,23) | −0,006 (0,88) | 21 / 3 | vert |
| GJR-GARCH(1,1)-t | 1,156 | −0,018 (0,63) | +0,004 (0,92) | 25 / 3 | vert |
| EWMA (λ = 0,94) | 1,174 | — | +0,023 (0,64) | 15 / 4 | vert |
| Normale constante | 1,151 | −0,023 (0,64) | — | 17 / 7 | vert |
| Simulation historique 250 j | 1,271 | +0,097 (0,12) | +0,119 (0,09) | 23 / **10** | **jaune** |

Exceptions attendues : 22,9 à 95 % et 4,6 à 99 %, sur 459 séances.

À retenir :

1. **L'effet ARCH est très net** et les GARCH l'absorbent. Mais hors échantillon, leur gain sur EWMA ou sur une
   volatilité constante est **faible et non significatif**, et il n'apparaît que dans 2 sous-périodes sur 4 contre EWMA
   (surtout T1, le choc d'avril 2025).
2. La volatilité constante semble correcte sur le test, mais elle était de loin la pire sur la validation 2023-2024,
   plus calme : son score dépend de la période.
3. **VaR** : tous les GARCH, EWMA et la normale constante passent les tests de Kupiec et de Christoffersen à 95 % et à
   99 %. La **simulation historique est rejetée à 99 %** (10 exceptions, Kupiec p = 0,028) : sa fenêtre de 250 séances
   réagit trop lentement au choc d'avril 2025. La loi de Student rend la VaR 99 % plus prudente (3 exceptions contre 6
   pour la loi normale).
4. L'asymétrie GJR est significative sur le train (LR p ≈ 1e-5), mais elle **prévoit moins bien** hors échantillon,
   en validation comme en test.
5. Réponse côté risque à la question B05 : la structure GARCH existe, mais sa complexité n'apporte qu'un avantage
   modeste et instable face à des références simples comme EWMA. Son intérêt principal est une VaR qui réagit vite en
   période de stress, ce que la simulation historique ne fait pas.

## Limites

- Une seule série et un horizon d'une séance ; pas de VaR multi-jours ni de portefeuille.
- r² est un proxy bruité de la variance réalisée, et le proxy de Parkinson ignore l'écart de la nuit.
- À 99 %, environ 4,6 exceptions sont attendues sur le test : les tests de la VaR 99 % et de l'ES ont peu de puissance.
- Les conventions (λ = 0,94, fenêtre de 250 séances, seuils d'admissibilité, budget de VaR de 2 %) sont fixées à
  l'avance dans `configs/risk.yaml` et ne sont pas optimisées.
