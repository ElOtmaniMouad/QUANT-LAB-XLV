"""Métriques et tests de comparaison des prévisions (livrable D03, réutilisé en D05).

Toutes les métriques portent sur la prévision à une séance du rendement log.

  * MAE, RMSE         erreurs absolues / quadratiques (exprimées en %) ;
  * MAE_prix          erreur absolue sur le prix ajusté reconstruit P̂_t = P_{t-1}·exp(r̂_t) (USD) ;
  * DA                exactitude directionnelle : part des séances où sign(r̂_t) = sign(r_t),
                      avec la convention r̂_t ≥ 0 ⇒ « hausse » (la référence zéro prévoit donc
                      toujours une hausse et sa DA est la part de séances haussières) ;
  * R2_oos            1 − SSE(modèle) / SSE(moyenne historique) (Campbell & Thompson, 2008) ;
  * ratio_RMSE_zero   RMSE(modèle) / RMSE(marche aléatoire) — < 1 : le modèle fait mieux ;
  * DM                test de Diebold-Mariano (perte quadratique) avec correction
                      Harvey-Leybourne-Newbold, contre la marche aléatoire ; H0 : précision égale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _aligned(actual: pd.Series, pred: pd.Series):
    df = pd.concat({"a": actual, "p": pred}, axis=1).dropna()
    return df["a"].to_numpy(), df["p"].to_numpy(), df.index


def mae(actual, pred) -> float:
    a, p, _ = _aligned(actual, pred)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, pred) -> float:
    a, p, _ = _aligned(actual, pred)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def directional_accuracy(actual, pred) -> float:
    a, p, _ = _aligned(actual, pred)
    return float(np.mean((p >= 0) == (a >= 0)))


def r2_oos(actual, pred, bench) -> float:
    df = pd.concat({"a": actual, "p": pred, "b": bench}, axis=1).dropna()
    sse = np.sum((df["a"] - df["p"]) ** 2)
    sse_b = np.sum((df["a"] - df["b"]) ** 2)
    return float(1.0 - sse / sse_b)


def diebold_mariano(actual, pred_model, pred_bench, h: int = 1) -> tuple[float, float]:
    """Statistique DM-HLN et p-value bilatérale (loi de Student à n-1 ddl).

    d_t = e_model² − e_bench² : une statistique négative signifie que le modèle
    a une erreur quadratique plus faible que la référence.
    """
    df = pd.concat({"a": actual, "m": pred_model, "b": pred_bench}, axis=1).dropna()
    d = (df["a"] - df["m"]) ** 2 - (df["a"] - df["b"]) ** 2
    n = len(d)
    if n < 10 or np.allclose(d.to_numpy(), 0.0):
        return float("nan"), float("nan")
    dc = d - d.mean()
    gamma = [np.sum(dc.to_numpy()[k:] * dc.to_numpy()[:n - k]) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = d.mean() / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    return stat, float(2 * stats.t.sf(abs(stat), df=n - 1))


def evaluate(data: pd.DataFrame, preds: pd.DataFrame, zero_col: str = "zero",
             mean_col: str = "hist_mean") -> pd.DataFrame:
    """Tableau de métriques, une ligne par colonne de ``preds``.

    ``data`` doit contenir ``y``, ``price`` et ``price_prev`` sur les mêmes dates.
    """
    rows = []
    a = data["y"]
    for col in preds.columns:
        p = preds[col]
        ok = p.notna() & a.notna()
        price_hat = data["price_prev"][ok] * np.exp(p[ok])
        dm, dm_p = diebold_mariano(a, p, preds[zero_col])
        rows.append({
            "modele": col,
            "n": int(ok.sum()),
            "MAE_%": 100 * mae(a, p),
            "RMSE_%": 100 * rmse(a, p),
            "MAE_prix_USD": float(np.mean(np.abs(data["price"][ok] - price_hat))),
            "DA": directional_accuracy(a, p),
            "R2_oos": r2_oos(a, p, preds[mean_col]),
            "ratio_RMSE_zero": rmse(a, p) / rmse(a, preds[zero_col]),
            "DM_vs_zero": dm,
            "DM_p": dm_p,
        })
    return pd.DataFrame(rows).set_index("modele")


def evaluate_by_period(data: pd.DataFrame, preds: pd.DataFrame, periods: dict) -> pd.DataFrame:
    """``evaluate`` sur chaque sous-période {nom: (début, fin)} ; index (periode, modele)."""
    frames = []
    for name, (start, end) in periods.items():
        m = (data.index >= start) & (data.index <= end)
        tab = evaluate(data[m], preds[m]).reset_index()
        tab.insert(0, "periode", name)
        frames.append(tab)
    return pd.concat(frames, ignore_index=True).set_index(["periode", "modele"])
