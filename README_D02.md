# D02 — Données et audit

## Prérequis
Python 3.11+ et une connexion Internet pour la collecte Yahoo Finance.

## Installation
Depuis la racine du dépôt :

```bash
python -m venv .venv
```

Windows PowerShell :
```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS :
```bash
source .venv/bin/activate
```

Puis :
```bash
pip install -r requirements.txt
```

## 1. Collecter XLV et SPY

La date de fin est exclusive. Exemple pour récupérer jusqu'au 14 novembre 2026 inclus :

```bash
python -m src.data --start 2018-01-01 --end 2026-11-15
```

Le script crée :
- `data/raw/xlv.csv` et `data/raw/spy.csv`
- `data/processed/xlv_spy_daily.csv`
- `data/snapshots/YYYY-MM-DD/xlv_spy_daily.csv`
- `data/snapshots/YYYY-MM-DD/manifest.json`

## 2. Auditer les données

```bash
python -m src.audit
```

Le script produit :
- `reports/quality_checks.csv`
- `reports/quality_report.json`
- `reports/coherence_xlv_spy.png`

## 3. Recharger les données

```python
import pandas as pd

df = pd.read_csv(
    "data/processed/xlv_spy_daily.csv",
    parse_dates=["Date"],
    index_col="Date"
)
print(df.head())
```

## Contrôles réalisés
Dates dupliquées ou non triées, valeurs manquantes, prix positifs, cohérence OHLC, volume non négatif, rendements extrêmes à examiner et dates communes XLV/SPY.

Les anomalies sont signalées dans le rapport. Elles ne sont pas supprimées silencieusement : chaque correction éventuelle doit être justifiée et traçable.

## Limite de la source
Yahoo Finance via yfinance fournit les données disponibles par son service, qui peuvent être différées. Ce mécanisme d'actualisation ne doit pas être décrit comme un flux professionnel garanti en temps réel.

## Livrables à conserver pour D02
1. Scripts `src/data.py` et `src/audit.py`.
2. Snapshot daté avec `manifest.json`.
3. `data/data_dictionary.md`.
4. Rapport de contrôles qualité.
5. Graphique `coherence_xlv_spy.png`.
