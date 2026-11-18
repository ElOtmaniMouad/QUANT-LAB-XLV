"""Modèles ARIMA, SARIMA et SARIMAX sur le rendement log de XLV (livrable D03).

Les trois familles partagent le même moteur (``statsmodels`` SARIMAX, estimation
par maximum de vraisemblance, filtre de Kalman) :

  * ARIMA(p, 0, q)               sur r_t      ≡ ARIMA(p, 1, q) sur ln P_t ;
  * SARIMA(p, 0, q)(P, 0, Q)_5   ajoute un cycle hebdomadaire de 5 séances ;
  * SARIMAX(p, 0, q) + x_t        x_t = r^SPY_{t-1}, connu à la clôture t-1.

Les rendements sont multipliés par 100 (unité : %) avant estimation : l'optimiseur
est bien plus stable qu'avec des variances de l'ordre de 1e-4. Les prévisions
renvoyées sont reconverties en rendement log (÷ 100).

Évaluation hors échantillon (``walk_forward``) : fenêtre croissante depuis le
début du train ; les paramètres sont ré-estimés toutes les ``refit_every``
séances avec les seules données antérieures au bloc ; à l'intérieur du bloc, les
paramètres restent fixes et le filtre intègre chaque nouvelle observation, de
sorte que la prévision de la séance t n'utilise que l'information disponible à
la clôture de t-1.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats

SCALE = 100.0  # rendements estimés en %


@dataclass(frozen=True)
class ArimaSpec:
    """Spécification d'un modèle de la famille ARIMA."""
    name: str
    order: tuple = (0, 0, 0)
    seasonal_order: tuple = (0, 0, 0, 0)
    trend: str = "c"
    exog: bool = False

    @property
    def family(self) -> str:
        if self.exog:
            return "SARIMAX"
        if any(self.seasonal_order[:3]):
            return "SARIMA"
        return "ARIMA"

    @property
    def label(self) -> str:
        p, d, q = self.order
        txt = f"({p},{d},{q})"
        if any(self.seasonal_order[:3]):
            P, D, Q, s = self.seasonal_order
            txt += f"({P},{D},{Q})_{s}"
        if self.exog:
            txt += " + r_SPY(t-1)"
        return f"{self.family}{txt}"

    def to_dict(self) -> dict:
        return {"name": self.name, "family": self.family, "label": self.label,
                "order": list(self.order), "seasonal_order": list(self.seasonal_order),
                "trend": self.trend, "exog": self.exog}


def _model(spec: ArimaSpec, y: np.ndarray, x: Optional[np.ndarray]) -> SARIMAX:
    return SARIMAX(
        y * SCALE,
        exog=(x * SCALE) if spec.exog else None,
        order=spec.order, seasonal_order=spec.seasonal_order, trend=spec.trend,
        enforce_stationarity=True, enforce_invertibility=True,
    )


def fit(spec: ArimaSpec, y, x=None, start_params=None):
    """Estime le modèle sur les tableaux ``y`` (et ``x``) fournis."""
    y = np.asarray(y, dtype=float)
    x = None if x is None else np.asarray(x, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        return _model(spec, y, x).fit(disp=False, start_params=start_params, maxiter=500)


def converged(res) -> bool:
    return bool(res.mle_retvals.get("converged", True)) if res.mle_retvals else True


# --------------------------------------------------------------------------- #
# Identification (train uniquement)
# --------------------------------------------------------------------------- #
def order_grid(y, max_p: int, max_q: int, trend: str = "c",
               rules: Optional[dict] = None) -> pd.DataFrame:
    """AIC / BIC de tous les ARIMA(p, 0, q), p ≤ max_p, q ≤ max_q.

    Avec ``rules`` (section ``selection`` de la configuration), ajoute les
    diagnostics et le verdict d'admissibilité de chaque ordre.
    """
    rows = []
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        spec = ArimaSpec(f"arima_{p}{q}", order=(p, 0, q), trend=trend)
        res = fit(spec, y)
        row = {"p": p, "q": q, "aic": res.aic, "bic": res.bic,
               "llf": res.llf, "converged": converged(res)}
        if rules is not None:
            adm = admissibility(res, rules)
            row.update({k: adm[k] for k in ("ar_min_modulus", "ma_min_modulus", "root_cancellation",
                                            "LB_robuste_p", "admissible", "motif")})
        rows.append(row)
    out = pd.DataFrame(rows)
    out["d_aic"] = out["aic"] - out["aic"].min()
    out["d_bic"] = out["bic"] - out["bic"].min()
    return out.sort_values("aic").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Prévision hors échantillon
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardResult:
    spec: ArimaSpec
    predictions: pd.Series
    n_refits: int
    n_not_converged: int
    params: list = field(default_factory=list)   # (date de ré-estimation, paramètres)


def walk_forward(spec: ArimaSpec, sample: pd.DataFrame, start, end,
                 refit_every: int = 21) -> WalkForwardResult:
    """Prévisions à une séance de ``y`` pour les dates de [start, end].

    ``sample`` : jeu de modélisation (colonnes ``y`` et ``x_spy_lag1``) commençant
    au début du train ; toutes ses lignes antérieures à ``start`` servent
    d'historique d'estimation.
    """
    idx = sample.index
    y = sample["y"].to_numpy(dtype=float)
    x = sample[["x_spy_lag1"]].to_numpy(dtype=float)
    pos = np.flatnonzero((idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end)))
    if len(pos) == 0:
        raise ValueError("Aucune séance à prévoir dans l'intervalle demandé.")
    first, last = int(pos[0]), int(pos[-1])
    if first < 50:
        raise ValueError("Historique d'estimation trop court avant la première prévision.")

    preds = np.full(len(y), np.nan)
    start_params, n_refits, n_bad, params = None, 0, 0, []
    for b in range(first, last + 1, refit_every):
        stop = min(b + refit_every, last + 1)
        res = fit(spec, y[:b], x[:b], start_params=start_params)
        n_refits += 1
        n_bad += int(not converged(res))
        start_params = res.params
        params.append((idx[b], dict(zip(res.model.param_names, np.round(res.params, 6)))))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ext = res.apply(y[:stop] * SCALE, exog=(x[:stop] * SCALE) if spec.exog else None,
                            refit=False)
            # prédiction « in-sample » non dynamique = prévision à une séance
            preds[b:stop] = np.asarray(ext.predict(start=b, end=stop - 1)) / SCALE
    out = pd.Series(preds, index=idx, name=spec.name).iloc[first:last + 1]
    return WalkForwardResult(spec, out, n_refits, n_bad, params)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def coefficient_table(res) -> pd.DataFrame:
    """Coefficients estimés (unités : rendements en %)."""
    return pd.DataFrame({
        "coef": res.params, "std_err": res.bse, "z": res.tvalues, "p_value": res.pvalues,
    }, index=pd.Index(res.model.param_names, name="parametre"))


def n_arma_params(res) -> int:
    m = res.model
    return int(m.k_ar_params + m.k_ma_params + m.k_seasonal_ar_params + m.k_seasonal_ma_params)


def model_residuals(res) -> np.ndarray:
    """Résidus hors période d'initialisation du filtre."""
    return np.asarray(res.resid)[max(res.loglikelihood_burn, 1):]


def robust_ljung_box(resid, lags: int, model_df: int = 0) -> tuple[float, float]:
    """Test portmanteau robuste à l'hétéroscédasticité conditionnelle.

    Q̃ = n Σ_{k=1..m} γ̂_k² / τ̂_k,  γ̂_k = (1/n) Σ e_t e_{t-k},  τ̂_k = (1/n) Σ (e_t e_{t-k})²
    (Lobato, Nankervis & Savin, 2001). Sous H0 « pas d'autocorrélation », Q̃ suit
    approximativement un χ²(m − model_df), y compris quand la variance varie dans le
    temps (effet ARCH). Le Ljung-Box classique suppose une variance constante et
    rejette trop souvent sur des rendements financiers.
    """
    e = np.asarray(resid, dtype=float)
    e = e - e.mean()
    n = len(e)
    q = 0.0
    for k in range(1, lags + 1):
        prod = e[k:] * e[:-k]
        gamma, tau = prod.sum() / n, (prod ** 2).sum() / n
        q += n * gamma ** 2 / tau
    dof = lags - model_df
    return float(q), (float(stats.chi2.sf(q, dof)) if dof > 0 else float("nan"))


def root_checks(res) -> dict:
    """Plus petit module des racines AR / MA et plus petite distance racine AR – racine MA.

    Une racine AR presque égale à une racine MA se simplifie : le modèle contient
    alors des paramètres redondants, mal identifiés et instables hors échantillon.
    """
    ar, ma = np.atleast_1d(res.arroots), np.atleast_1d(res.maroots)
    return {
        "ar_min_modulus": float(np.min(np.abs(ar))) if ar.size else float("inf"),
        "ma_min_modulus": float(np.min(np.abs(ma))) if ma.size else float("inf"),
        "root_cancellation": float(min((abs(a - b) for a in ar for b in ma), default=float("inf"))),
    }


def admissibility(res, rules: dict) -> dict:
    """Diagnostics d'un modèle estimé sur le train et verdict d'admissibilité.

    ``rules`` = section ``selection`` de configs/models.yaml.
    """
    roots = root_checks(res)
    _, p_rlb = robust_ljung_box(model_residuals(res), int(rules["robust_lb_lags"]), n_arma_params(res))
    reasons = []
    if not converged(res):
        reasons.append("non convergé")
    if min(roots["ar_min_modulus"], roots["ma_min_modulus"]) < float(rules["min_root_modulus"]):
        reasons.append("racine sur le cercle unité")
    if roots["root_cancellation"] < float(rules["max_root_cancellation"]):
        reasons.append("racines AR/MA qui se compensent")
    if not np.isnan(p_rlb) and p_rlb < float(rules["alpha"]):
        reasons.append("résidus autocorrélés (LB robuste)")
    return {**roots, "LB_robuste_p": p_rlb, "converged": converged(res),
            "admissible": not reasons, "motif": "; ".join(reasons) or "—"}


def residual_diagnostics(res, lb_lags=(5, 10, 20), arch_lags: int = 5,
                         alpha: float = 0.05) -> pd.DataFrame:
    """Tests sur les résidus standardisés d'un modèle estimé.

    H0 de chaque test = le modèle est adéquat sur ce point :
      Ljung-Box (absence d'autocorrélation), Jarque-Bera (normalité),
      ARCH-LM (absence d'hétéroscédasticité conditionnelle).
    """
    resid = model_residuals(res)
    k = n_arma_params(res)
    rows = []
    lb = acorr_ljungbox(resid, lags=list(lb_lags), model_df=k)
    for lag, r in lb.iterrows():
        rows.append({"test": f"Ljung-Box (résidus, {lag} retards)", "statistique": r["lb_stat"],
                     "p_value": r["lb_pvalue"], "h0": "pas d'autocorrélation"})
    for lag in lb_lags:
        q, p = robust_ljung_box(resid, int(lag), k)
        rows.append({"test": f"Ljung-Box robuste ARCH ({lag} retards)", "statistique": q,
                     "p_value": p, "h0": "pas d'autocorrélation"})
    lb2 = acorr_ljungbox(resid ** 2, lags=[max(lb_lags)])
    rows.append({"test": f"Ljung-Box (résidus², {max(lb_lags)} retards)",
                 "statistique": float(lb2["lb_stat"].iloc[0]),
                 "p_value": float(lb2["lb_pvalue"].iloc[0]), "h0": "pas d'effet ARCH"})
    jb = stats.jarque_bera(resid)
    rows.append({"test": "Jarque-Bera", "statistique": float(jb.statistic),
                 "p_value": float(jb.pvalue), "h0": "résidus gaussiens"})
    lm, lm_p, _, _ = het_arch(resid, nlags=arch_lags)
    rows.append({"test": f"ARCH-LM ({arch_lags} retards)", "statistique": float(lm),
                 "p_value": float(lm_p), "h0": "pas d'effet ARCH"})
    out = pd.DataFrame(rows)
    # Ljung-Box sans degré de liberté restant (retards ≤ nb de paramètres ARMA) : non applicable
    out["verdict"] = np.where(out["p_value"].isna(), "non applicable",
                              np.where(out["p_value"] < alpha, "H0 rejetée", "H0 non rejetée"))
    return out


def root_moduli(res) -> dict:
    """Plus petit module des racines AR / MA (> 1 ⇔ stationnaire / inversible) et compensation."""
    return root_checks(res)
