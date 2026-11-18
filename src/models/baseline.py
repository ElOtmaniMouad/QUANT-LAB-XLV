"""Modèles de référence (livrable D03).

Chaque fonction reçoit la série des rendements ``y`` (indexée par date, depuis le
début de l'échantillon de modélisation) et renvoie, pour chaque séance t, la
prévision de y_t construite UNIQUEMENT avec y_{t-1}, y_{t-2}, … (décalage d'une
séance systématique). La première valeur est donc NaN.

  * ``zero``       : r̂_t = 0 — marche aléatoire sur le log-prix (P̂_t = P_{t-1}) ;
  * ``naive``      : r̂_t = r_{t-1} — persistance du dernier rendement ;
  * ``hist_mean``  : r̂_t = moyenne de tous les rendements passés (fenêtre croissante) ;
  * ``moving_avg`` : r̂_t = moyenne des k derniers rendements.

Buy & Hold et Cash sont des références de STRATÉGIE (D06), pas de prévision.
"""
from __future__ import annotations

import pandas as pd


def zero(y: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=y.index, name="zero")
    out.iloc[0] = float("nan")
    return out


def naive(y: pd.Series) -> pd.Series:
    return y.shift(1).rename("naive")


def hist_mean(y: pd.Series) -> pd.Series:
    return y.expanding().mean().shift(1).rename("hist_mean")


def moving_avg(y: pd.Series, window: int) -> pd.Series:
    return y.rolling(window, min_periods=window).mean().shift(1).rename(f"ma{window}")


def all_baselines(y: pd.Series, ma_windows) -> pd.DataFrame:
    cols = [zero(y), naive(y), hist_mean(y)] + [moving_avg(y, int(k)) for k in ma_windows]
    return pd.concat(cols, axis=1)
