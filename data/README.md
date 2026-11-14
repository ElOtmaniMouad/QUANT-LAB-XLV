# Données

- `raw/` : téléchargements originaux XLV et SPY.
- `processed/` : dataset synchronisé avec rendements.
- `snapshots/YYYY-MM-DD/` : copie datée du dataset et manifeste d'acquisition.

Ne pas modifier manuellement les fichiers `raw/`. Pour reconstruire les données, relancer le script de collecte. La source, la période et la date d'acquisition sont documentées dans `manifest.json`.
