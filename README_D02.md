# D02 — Données et audit

Binôme B05 · XLV (benchmark SPY). La documentation complète du livrable — provenance, dictionnaire,
conventions d'ajustement, périodes figées, contrôles et résultats — est dans [`data/README.md`](data/README.md).

## Pièces du livrable

| Pièce demandée | Fichier |
|---|---|
| Script de collecte / mode import | [`src/data.py`](src/data.py) |
| Snapshot daté | [`data/snapshots/XLV_SPY_1d_20261116.csv`](data/snapshots/XLV_SPY_1d_20261116.csv) + manifeste [`.json`](data/snapshots/XLV_SPY_1d_20261116.json) |
| Dictionnaire | [`data/README.md`](data/README.md), section 3 |
| Contrôles qualité | [`src/quality.py`](src/quality.py) → [`reports/d02_data_audit.md`](reports/d02_data_audit.md), [`reports/d02_quality_checks.csv`](reports/d02_quality_checks.csv) |
| Graphique de cohérence | [`reports/figures/d02_coherence.png`](reports/figures/d02_coherence.png) |
| Paramètres gelés | [`configs/data.yaml`](configs/data.yaml) |
| Tests | [`tests/test_data.py`](tests/test_data.py) |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Commandes

```bash
python -m src.data show      # mode DEMO : recharge le snapshot gelé et vérifie son SHA-256 (sans réseau)
python -m src.quality        # relance les 36 contrôles, régénère le rapport et le graphique
python -m pytest             # 46 tests, sans réseau
python -m src.data collect   # (réseau) télécharge à nouveau et gèle un NOUVEAU snapshot daté du jour
```

## Résultat

Snapshot du 16/11/2026 : 2 481 séances par ticker, du 03/01/2017 au 13/11/2026.
**33 contrôles OK · 3 INFO · 0 WARN · 0 FAIL — données utilisables.**

Les anomalies sont signalées dans le rapport et jamais supprimées silencieusement.

## Limite de la source

Yahoo Finance via `yfinance` est une source gratuite, sans garantie, dont les données peuvent être
différées ; ce n'est pas un flux professionnel en temps réel. Le snapshot gelé garantit que le travail
reste reproductible même si la source change ou devient indisponible.
