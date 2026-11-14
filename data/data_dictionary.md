# Dictionnaire des données — QUANT LAB B05

## Source et périmètre

- Source prévue : Yahoo Finance via `yfinance`.
- Instruments : XLV (actif étudié) et SPY (benchmark).
- Fréquence : journalière, une ligne par séance de marché.
- Fuseau : l'index est conservé comme date de séance, sans heure.
- Ajustement : `auto_adjust=False`; les colonnes `Close` et `Adj Close` sont conservées.
- Rendements : rendement simple calculé sur `Adj Close`, soit `P_t / P_(t-1) - 1`.

## Variables

| Variable | Type | Description | Unité / remarque |
|---|---|---|---|
| Date | date | Date de la séance boursière | Format YYYY-MM-DD |
| XLV_Open | numérique | Prix d'ouverture de XLV | USD |
| XLV_High | numérique | Plus haut de XLV pendant la séance | USD |
| XLV_Low | numérique | Plus bas de XLV pendant la séance | USD |
| XLV_Close | numérique | Prix de clôture non ajusté de XLV | USD |
| XLV_Adj Close | numérique | Prix ajusté de XLV | USD |
| XLV_Volume | entier | Volume échangé pour XLV | Nombre de parts |
| SPY_Open | numérique | Prix d'ouverture de SPY | USD |
| SPY_High | numérique | Plus haut de SPY pendant la séance | USD |
| SPY_Low | numérique | Plus bas de SPY pendant la séance | USD |
| SPY_Close | numérique | Prix de clôture non ajusté de SPY | USD |
| SPY_Adj Close | numérique | Prix ajusté de SPY | USD |
| SPY_Volume | entier | Volume échangé pour SPY | Nombre de parts |
| XLV_Return | numérique | Rendement simple journalier de XLV | Décimal; 0.01 = 1 % |
| SPY_Return | numérique | Rendement simple journalier de SPY | Décimal; 0.01 = 1 % |

## Règles de lecture

- Les données `raw/` sont les téléchargements par ticker, avant synchronisation.
- `processed/xlv_spy_daily.csv` contient uniquement les dates communes à XLV et SPY.
- La première observation de rendement est normalement vide, car aucun jour précédent n'est disponible dans le fichier.
- Une valeur extrême de rendement est signalée pour examen, mais n'est pas supprimée automatiquement.
- Les jours sans séance de marché (week-ends et jours fériés) ne sont pas considérés comme des données manquantes.
