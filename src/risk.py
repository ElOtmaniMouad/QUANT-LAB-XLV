"""Mesures de risque, références simples et backtests (livrable D04, réutilisé en D06 et D07).

Conventions d'unités (identiques dans tout le module) :

  * r_t      rendement log de la séance t en %, r_t = 100 · ln(P_t / P_{t-1}) ;
  * σ_t      écart-type conditionnel prévu de r_t, en % par séance (× √252 : annualisé) ;
  * VaR_t    Value-at-Risk d'une position LONGUE, nombre POSITIF en % de la valeur de la position :
             P(−r_t > VaR_t | information en t-1) = 1 − niveau ;
  * ES_t     Expected Shortfall : perte moyenne quand la perte dépasse la VaR, en %, ES ≥ VaR ;
  * USD      montant perdu sur une position de C dollars : C · (1 − exp(−VaR / 100)).

Une exception (« hit ») se produit à la séance t si la perte réalisée −r_t dépasse VaR_t,
prévue à la clôture de t-1. Toutes les prévisions de ce module n'utilisent que r_{≤t-1}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Quantiles et ES des lois standardisées (variance 1)
# --------------------------------------------------------------------------- #
def std_t_scale(nu: float) -> float:
    """Facteur qui ramène une Student-t(ν) à une variance unité : Z = T · √((ν−2)/ν)."""
    if nu <= 2:
        raise ValueError("ν doit être > 2 pour que la variance existe.")
    return float(np.sqrt((nu - 2.0) / nu))


def std_quantile(p: float, dist: str = "normal", nu: float | None = None) -> float:
    """Quantile d'ordre p (p petit, ex. 0,01) de la loi standardisée de moyenne 0 et variance 1."""
    if dist == "normal":
        return float(stats.norm.ppf(p))
    if dist == "t":
        return float(stats.t.ppf(p, nu) * std_t_scale(nu))
    raise ValueError(f"Loi inconnue : {dist}")


def std_tail_mean(p: float, dist: str = "normal", nu: float | None = None) -> float:
    """E[Z | Z ≤ q_p] pour la loi standardisée (nombre négatif)."""
    if dist == "normal":
        q = stats.norm.ppf(p)
        return float(-stats.norm.pdf(q) / p)
    if dist == "t":
        tq = stats.t.ppf(p, nu)
        tail = -(stats.t.pdf(tq, nu) / p) * (nu + tq ** 2) / (nu - 1.0)
        return float(tail * std_t_scale(nu))
    raise ValueError(f"Loi inconnue : {dist}")


def parametric_var_es(mu, sigma, level: float, dist: str = "normal", nu=None) -> tuple:
    """VaR et ES (positifs, en %) d'une position longue pour r ~ μ + σ·Z.

    ``mu``, ``sigma`` (et ``nu`` pour la loi t) peuvent être des tableaux de même forme.
    """
    p = 1.0 - level
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    if dist == "t":
        nu_arr = np.broadcast_to(np.asarray(nu, dtype=float), sigma.shape)
        q = np.array([std_quantile(p, "t", v) for v in nu_arr.ravel()]).reshape(sigma.shape)
        m = np.array([std_tail_mean(p, "t", v) for v in nu_arr.ravel()]).reshape(sigma.shape)
    else:
        q, m = std_quantile(p), std_tail_mean(p)
    return -(mu + sigma * q), -(mu + sigma * m)


def var_to_usd(var_pct, capital: float = 100_000.0):
    """Perte en USD sur une position de ``capital`` pour une perte log de ``var_pct`` %."""
    return capital * (1.0 - np.exp(-np.asarray(var_pct, dtype=float) / 100.0))


# --------------------------------------------------------------------------- #
# Références simples (prévisions pour la séance t avec r_{≤t-1} uniquement)
# --------------------------------------------------------------------------- #
def ewma_sigma(r: pd.Series, lam: float = 0.94, init_obs: int = 20) -> pd.Series:
    """RiskMetrics : σ²_t = λ σ²_{t-1} + (1 − λ) r²_{t-1}, moyenne supposée nulle.

    Initialisation : moyenne de r² sur les ``init_obs`` premières séances ; la valeur
    prévue pour ces séances d'amorçage est NaN.
    """
    x = r.to_numpy(dtype=float)
    s2 = np.full(len(x), np.nan)
    if len(x) <= init_obs:
        return pd.Series(s2, index=r.index, name="EWMA")
    s2[init_obs] = np.mean(x[:init_obs] ** 2)
    for t in range(init_obs + 1, len(x)):
        s2[t] = lam * s2[t - 1] + (1.0 - lam) * x[t - 1] ** 2
    return pd.Series(np.sqrt(s2), index=r.index, name="EWMA")


def constant_moments(r: pd.Series, min_obs: int = 60) -> tuple[pd.Series, pd.Series]:
    """Moyenne et écart-type de TOUS les rendements passés (fenêtre croissante, décalée d'une séance)."""
    mu = r.expanding(min_periods=min_obs).mean().shift(1)
    sd = r.expanding(min_periods=min_obs).std().shift(1)
    return mu.rename("const_mu"), sd.rename("const_sigma")


def historical_var_es(r: pd.Series, level: float, window: int = 250) -> tuple[pd.Series, pd.Series]:
    """Simulation historique : quantile et moyenne de queue des ``window`` derniers rendements."""
    p = 1.0 - level
    x = r.to_numpy(dtype=float)
    var, es = np.full(len(x), np.nan), np.full(len(x), np.nan)
    for t in range(window, len(x)):
        w = x[t - window:t]
        q = np.quantile(w, p)
        var[t] = -q
        es[t] = -w[w <= q].mean()
    return pd.Series(var, index=r.index), pd.Series(es, index=r.index)


def exposure_limit(var99_pct, max_var_pct_capital: float = 2.0, max_exposure: float = 1.0):
    """Fraction maximale du capital investie pour que la VaR99 de la position reste ≤ le budget.

    La VaR est convertie en perte simple (1 − e^{−VaR/100}) : une position w·C perd
    w·C·(1 − e^{−VaR/100}). Bornée entre 0 et ``max_exposure`` (ni levier ni vente à découvert).
    """
    loss = 1.0 - np.exp(-np.asarray(var99_pct, dtype=float) / 100.0)
    with np.errstate(divide="ignore"):
        w = (max_var_pct_capital / 100.0) / loss
    return np.clip(w, 0.0, max_exposure)


# --------------------------------------------------------------------------- #
# Pertes pour comparer les prévisions
# --------------------------------------------------------------------------- #
def qlike(r, sigma) -> np.ndarray:
    """QLIKE = ln h + r²/h, h = σ² (Patton, 2011) : minimale en espérance pour h = vraie variance.

    Robuste au bruit du proxy r² (contrairement au MSE, dominé par quelques journées extrêmes).
    """
    h = np.asarray(sigma, dtype=float) ** 2
    return np.log(h) + np.asarray(r, dtype=float) ** 2 / h


def mse_var(r, sigma) -> np.ndarray:
    """Erreur quadratique sur la variance : (r² − σ²)²  (unité : %⁴)."""
    return (np.asarray(r, dtype=float) ** 2 - np.asarray(sigma, dtype=float) ** 2) ** 2


def tick_loss(r, var, level: float) -> np.ndarray:
    """Perte quantile (« tick ») de la VaR vue comme quantile q = −VaR de r."""
    p = 1.0 - level
    r, q = np.asarray(r, dtype=float), -np.asarray(var, dtype=float)
    return ((r < q).astype(float) - p) * (q - r)


def fz0_loss(r, var, es, level: float) -> np.ndarray:
    """Perte FZ0 de Fissler-Ziegel (Patton, Ziegel & Chen, 2019), conjointement pour (VaR, ES).

    Avec v = −VaR < 0 et e = −ES ≤ v : L = −1{r≤v}(v−r)/(p·e) + v/e + ln(−e) − 1.
    Plus elle est faible, meilleure est la paire (VaR, ES).
    """
    p = 1.0 - level
    r = np.asarray(r, dtype=float)
    v, e = -np.asarray(var, dtype=float), -np.asarray(es, dtype=float)
    hit = (r <= v).astype(float)
    return -hit * (v - r) / (p * e) + v / e + np.log(-e) - 1.0


def loss_dm_test(loss_model, loss_bench) -> tuple[float, float]:
    """Diebold-Mariano (correction HLN, h = 1) sur deux séries de pertes.

    Statistique < 0 : le modèle a une perte moyenne plus faible que la référence.
    """
    d = np.asarray(loss_model, dtype=float) - np.asarray(loss_bench, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 10 or np.allclose(d, 0.0):
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(np.var(d) / n) * np.sqrt((n - 1) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def mincer_zarnowitz(r, sigma, hac_lags: int = 5) -> dict:
    """Régression r²_t = a + b·σ²_t + u (erreurs HAC de Newey-West) et test de Wald a = 0, b = 1.

    Une prévision de variance sans biais donne a ≈ 0 et b ≈ 1 ; le R² mesure la part de la
    variation du proxy r² expliquée (faible par nature : r² est un proxy très bruité).
    """
    import statsmodels.api as sm
    y = np.asarray(r, dtype=float) ** 2
    h = np.asarray(sigma, dtype=float) ** 2
    ok = ~(np.isnan(y) | np.isnan(h))
    res = sm.OLS(y[ok], sm.add_constant(h[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    w = res.wald_test((np.eye(2), np.array([0.0, 1.0])), scalar=True)
    return {"a": float(res.params[0]), "b": float(res.params[1]), "b_se": float(res.bse[1]),
            "R2": float(res.rsquared), "wald_p": float(w.pvalue)}


# --------------------------------------------------------------------------- #
# Backtests de la VaR et de l'ES
# --------------------------------------------------------------------------- #
def hits(r, var) -> np.ndarray:
    """1 si la perte réalisée −r_t dépasse la VaR prévue, 0 sinon."""
    return (-np.asarray(r, dtype=float) > np.asarray(var, dtype=float)).astype(int)


def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(h, level: float) -> tuple[float, float]:
    """Test de couverture inconditionnelle (Kupiec, 1995) — H0 : taux d'exceptions = 1 − niveau."""
    h = np.asarray(h, dtype=int)
    n, x, p = len(h), int(h.sum()), 1.0 - level
    pi = x / n
    ll0 = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(n - x, 1 - pi) + _xlogy(x, pi)
    lr = -2.0 * (ll0 - ll1)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen(h, level: float) -> dict:
    """Tests d'indépendance et de couverture conditionnelle (Christoffersen, 1998).

    Indépendance — H0 : une exception aujourd'hui ne rend pas plus probable une exception demain.
    Couverture conditionnelle = Kupiec + indépendance, χ²(2).
    """
    h = np.asarray(h, dtype=int)
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    pi0 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_ind = (_xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1))
    ll_0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    lr_ind = max(-2.0 * (ll_0 - ll_ind), 0.0)
    lr_pof, _ = kupiec_pof(h, level)
    lr_cc = lr_pof + lr_ind
    return {"n11": n11, "LR_ind": float(lr_ind), "ind_p": float(stats.chi2.sf(lr_ind, 1)),
            "LR_cc": float(lr_cc), "cc_p": float(stats.chi2.sf(lr_cc, 2))}


def traffic_light(n_hits: int, n: int, level: float) -> str:
    """Zone « feux tricolores » du Comité de Bâle généralisée à n séances.

    Vert si P(X ≤ x) < 95 %, jaune si < 99,99 %, rouge sinon, X ~ Binomiale(n, 1 − niveau).
    Pour n = 250 et 99 % : vert ≤ 4 exceptions, jaune 5–9, rouge ≥ 10.
    """
    c = stats.binom.cdf(n_hits, n, 1.0 - level)
    return "vert" if c < 0.95 else ("jaune" if c < 0.9999 else "rouge")


def es_backtest(r, var, es, sigma) -> dict:
    """Test de McNeil & Frey (2000) sur les séances d'exception.

    Résidu d'exception : (perte − ES) / σ. Si l'ES est bien calibrée, sa moyenne est nulle ;
    une moyenne positive signifie que l'ES sous-estime les pertes extrêmes (test unilatéral).
    Le ratio perte moyenne / ES moyenne vaut ≈ 1 pour une ES correcte.
    """
    r, var, es, sigma = (np.asarray(v, dtype=float) for v in (r, var, es, sigma))
    m = -r > var
    k = int(m.sum())
    if k == 0:
        return {"n_exceptions": 0, "perte_moy_%": float("nan"), "ES_moy_%": float("nan"),
                "ratio_perte_ES": float("nan"), "MF_t": float("nan"), "MF_p": float("nan")}
    loss = -r[m]
    resid = (loss - es[m]) / sigma[m]
    if k >= 2 and resid.std(ddof=1) > 0:
        t = resid.mean() / (resid.std(ddof=1) / np.sqrt(k))
        p = float(stats.t.sf(t, df=k - 1))
    else:
        t, p = float("nan"), float("nan")
    return {"n_exceptions": k, "perte_moy_%": float(loss.mean()), "ES_moy_%": float(es[m].mean()),
            "ratio_perte_ES": float(loss.mean() / es[m].mean()), "MF_t": float(t), "MF_p": p}


def var_backtest_table(r, var: dict, level: float) -> pd.DataFrame:
    """Tableau de vérification de la VaR, une ligne par modèle (``var`` : {nom: série VaR})."""
    rows = []
    for name, v in var.items():
        df = pd.concat({"r": pd.Series(r), "v": pd.Series(v)}, axis=1).dropna()
        h = hits(df["r"], df["v"])
        n, x = len(h), int(h.sum())
        lr, p_pof = kupiec_pof(h, level)
        ch = christoffersen(h, level)
        rows.append({"modele": name, "niveau": level, "n": n, "exceptions": x,
                     "attendues": n * (1 - level), "taux_%": 100 * x / n,
                     "Kupiec_p": p_pof, "independance_p": ch["ind_p"], "couverture_cond_p": ch["cc_p"],
                     "exceptions_consecutives": ch["n11"], "zone_Bale": traffic_light(x, n, level),
                     "VaR_moy_%": float(df["v"].mean()),
                     "tick_loss": float(np.mean(tick_loss(df["r"], df["v"], level)))})
    return pd.DataFrame(rows).set_index("modele")
