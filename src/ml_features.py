"""Variables explicatives des modèles ML/DL (livrable D05).

Une ligne = une séance t à prévoir. TOUTES les variables d'une ligne sont calculées
avec l'information disponible à la CLÔTURE DE t-1 : on calcule d'abord chaque
variable « à la date s » avec les données ≤ s, puis on décale tout le tableau d'une
séance (``shift(1)``). La cible de la ligne t est r_t (``y`` du jeu de modélisation D03).

  * rendements retardés de XLV      r_{t-1}, …, r_{t-5}
  * moyennes et volatilités glissantes de XLV (5, 20, 60 séances)
  * rendements retardés de SPY      r^SPY_{t-1}, …, r^SPY_{t-3}, volatilité 20 séances
  * force relative XLV − SPY        somme de (r^XLV − r^SPY) sur 5 et 20 séances
  * tendance                        ln(P / moyenne mobile des prix) sur 50 et 200 séances
  * volume                          ln(volume / volume moyen 20 séances)

Les rendements sont des rendements logarithmiques sur prix ajusté (dividendes inclus).
Le niveau de ``adj_close`` contient les dividendes futurs, mais tout RAPPORT de deux
prix ajustés de dates passées (rendement, écart à une moyenne mobile) ne dépend que
des dividendes entre ces deux dates : aucune fuite (cf. data/README.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import log_returns, wide


def build_features(df: pd.DataFrame, ml_cfg: dict, target: str = "XLV", bench: str = "SPY") -> pd.DataFrame:
    """Table des variables, indexée par date de séance (séances communes XLV/SPY)."""
    f = ml_cfg["features"]
    prices = wide(df, "adj_close").dropna()
    volume = wide(df, "volume").astype("float64").reindex(prices.index)
    r = log_returns(prices)
    x, s = r[target], r[bench]

    cols: dict[str, pd.Series] = {}
    for k in range(int(f["xlv_return_lags"])):
        cols[f"xlv_r_lag{k + 1}"] = x.shift(k)
    for w in f["xlv_mean_windows"]:
        cols[f"xlv_moy{w}"] = x.rolling(int(w)).mean()
    for w in f["xlv_vol_windows"]:
        cols[f"xlv_vol{w}"] = x.rolling(int(w)).std()
    for k in range(int(f["spy_return_lags"])):
        cols[f"spy_r_lag{k + 1}"] = s.shift(k)
    cols[f"spy_vol{f['spy_vol_window']}"] = s.rolling(int(f["spy_vol_window"])).std()
    for w in f["relative_windows"]:
        cols[f"force_rel{w}"] = (x - s).rolling(int(w)).sum()
    for w in f["trend_ma_windows"]:
        cols[f"tendance_ma{w}"] = np.log(prices[target] / prices[target].rolling(int(w)).mean())
    w = int(f["volume_window"])
    cols[f"volume_rel{w}"] = np.log(volume[target] / volume[target].rolling(w).mean())

    X = pd.DataFrame(cols, index=prices.index).shift(1)     # information connue à la clôture de t-1
    X.index.name = "date"
    return X


def make_xy(ds: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Aligne variables et cible sur les séances du jeu de modélisation (lignes complètes)."""
    Xa = X.reindex(ds.index)
    ok = Xa.notna().all(axis=1) & ds["y"].notna()
    return Xa[ok], ds.loc[ok, "y"]
