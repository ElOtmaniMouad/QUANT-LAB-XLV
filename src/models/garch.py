"""Modèles de volatilité conditionnelle GARCH sur le rendement de XLV (livrable D04).

Modèle (moyenne constante, rendements en %) :

    r_t = μ + ε_t,   ε_t = σ_t · z_t,   z_t i.i.d. de moyenne 0 et variance 1
    σ²_t = ω + α ε²_{t-1} + γ ε²_{t-1} 1{ε_{t-1} < 0} + β σ²_{t-1}

  * GARCH(1,1)      γ = 0 (réaction symétrique aux chocs) ;
  * GJR-GARCH(1,1)  γ ≥ 0 : une baisse augmente davantage la volatilité qu'une hausse
                    (effet de levier, Glosten-Jagannathan-Runkle, 1993) ;
  * loi de z_t      normale, ou Student-t standardisée à ν degrés de liberté (queues épaisses).

Persistance : α + β + γ/2 (< 1 : variance inconditionnelle ω / (1 − persistance) finie).
Demi-vie d'un choc de volatilité : ln(0,5) / ln(persistance) séances.

La moyenne est constante : D03 a montré que la dynamique ARMA de la moyenne n'améliore pas
la prévision hors échantillon. Estimation par maximum de vraisemblance (bibliothèque ``arch``).

Évaluation hors échantillon (``walk_forward``) : même protocole qu'en D03. Fenêtre croissante
depuis le début du train, paramètres ré-estimés toutes les ``refit_every`` séances avec les
seules données antérieures au bloc ; à l'intérieur du bloc les paramètres sont fixes et la
variance conditionnelle est mise à jour chaque jour, de sorte que σ_t n'utilise que r_{≤t-1}.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


@dataclass(frozen=True)
class GarchSpec:
    """Spécification d'un modèle de la famille GARCH(1,1)."""
    name: str
    o: int = 0                 # 1 = terme asymétrique GJR
    dist: str = "normal"       # "normal" ou "t"

    def to_dict(self) -> dict:
        return {"name": self.name, "p": 1, "o": self.o, "q": 1, "dist": self.dist, "mean": "Constant"}


def _model(spec: GarchSpec, y: np.ndarray):
    return arch_model(y, mean="Constant", vol="GARCH", p=1, o=spec.o, q=1, dist=spec.dist, rescale=False)


def fit(spec: GarchSpec, y, last_obs: int | None = None, starting_values=None):
    """Estime le modèle sur ``y`` (rendements en %), éventuellement sur y[:last_obs] seulement."""
    y = np.asarray(y, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _model(spec, y).fit(disp="off", last_obs=last_obs, starting_values=starting_values,
                                   options={"maxiter": 500})


def converged(res) -> bool:
    return int(getattr(res, "convergence_flag", 0)) == 0


def persistence(params: pd.Series) -> float:
    return float(params.get("alpha[1]", 0.0) + params.get("beta[1]", 0.0) + 0.5 * params.get("gamma[1]", 0.0))


def summary_row(spec: GarchSpec, res) -> dict:
    """Paramètres et grandeurs interprétables d'un modèle estimé."""
    p = res.params
    pers = persistence(p)
    uncond = p["omega"] / (1.0 - pers) if pers < 1 else float("inf")
    return {
        "modele": spec.name, "mu_%": p["mu"], "omega": p["omega"], "alpha": p["alpha[1]"],
        "gamma": p.get("gamma[1]", np.nan), "beta": p["beta[1]"], "nu": p.get("nu", np.nan),
        "persistance": pers,
        "demi_vie_seances": (np.log(0.5) / np.log(pers)) if 0 < pers < 1 else float("inf"),
        "vol_incond_annuelle_%": float(np.sqrt(uncond * 252)) if np.isfinite(uncond) else float("inf"),
        "logvraisemblance": res.loglikelihood, "AIC": res.aic, "BIC": res.bic,
        "converge": converged(res),
    }


def coefficient_table(spec: GarchSpec, res) -> pd.DataFrame:
    out = pd.DataFrame({"coef": res.params, "std_err": res.std_err, "t": res.tvalues,
                        "p_value": res.pvalues})
    out.index.name = "parametre"
    out = out.reset_index()
    out.insert(0, "modele", spec.name)
    return out


def std_residuals(res) -> np.ndarray:
    z = np.asarray(res.std_resid, dtype=float)
    return z[~np.isnan(z)]


def residual_diagnostics(spec: GarchSpec, res, arch_lags: int = 5, lb_lags: int = 10,
                         alpha: float = 0.05) -> pd.DataFrame:
    """Tests sur les résidus standardisés z_t = ε_t / σ_t d'un modèle estimé.

    H0 de chaque test = le modèle est adéquat sur ce point :
      Ljung-Box sur z (pas d'autocorrélation), Ljung-Box sur z² et ARCH-LM (plus d'effet ARCH),
      Jarque-Bera (z gaussien) et Kolmogorov-Smirnov contre la loi supposée du modèle.
    """
    z = std_residuals(res)
    rows = []
    lb = acorr_ljungbox(z, lags=[lb_lags])
    rows.append({"test": f"Ljung-Box (z, {lb_lags} retards)", "statistique": float(lb["lb_stat"].iloc[0]),
                 "p_value": float(lb["lb_pvalue"].iloc[0]), "h0": "pas d'autocorrélation"})
    lb2 = acorr_ljungbox(z ** 2, lags=[lb_lags])
    rows.append({"test": f"Ljung-Box (z², {lb_lags} retards)", "statistique": float(lb2["lb_stat"].iloc[0]),
                 "p_value": float(lb2["lb_pvalue"].iloc[0]), "h0": "plus d'effet ARCH"})
    lm, lm_p, _, _ = het_arch(z, nlags=arch_lags)
    rows.append({"test": f"ARCH-LM (z, {arch_lags} retards)", "statistique": float(lm),
                 "p_value": float(lm_p), "h0": "plus d'effet ARCH"})
    jb = stats.jarque_bera(z)
    rows.append({"test": "Jarque-Bera (z)", "statistique": float(jb.statistic),
                 "p_value": float(jb.pvalue), "h0": "z gaussien"})
    if spec.dist == "t":
        nu = float(res.params["nu"])
        ks = stats.kstest(z / np.sqrt((nu - 2) / nu), "t", args=(nu,))
        h0 = f"z ~ Student-t({nu:.1f}) standardisée"
    else:
        ks = stats.kstest(z, "norm")
        h0 = "z ~ N(0,1)"
    rows.append({"test": "Kolmogorov-Smirnov (loi du modèle)", "statistique": float(ks.statistic),
                 "p_value": float(ks.pvalue), "h0": h0})
    out = pd.DataFrame(rows)
    out.insert(0, "modele", spec.name)
    out["verdict"] = np.where(out["p_value"] < alpha, "H0 rejetée", "H0 non rejetée")
    return out


def admissibility(spec: GarchSpec, res, rules: dict) -> dict:
    """Le modèle est-il acceptable sur le train ?  (section ``admissibility`` de configs/risk.yaml)

    Conditions : convergence, paramètres de variance positifs, persistance < seuil, et plus
    d'effet ARCH significatif dans les résidus standardisés (ARCH-LM et Ljung-Box sur z²).
    """
    p = res.params
    z = std_residuals(res)
    _, lm_p, _, _ = het_arch(z, nlags=int(rules["arch_lm_lags"]))
    lb_p = float(acorr_ljungbox(z ** 2, lags=[int(rules["robust_lb_lags"])])["lb_pvalue"].iloc[0])
    pers = persistence(p)
    reasons = []
    if not converged(res):
        reasons.append("non convergé")
    if min(p["omega"], p["alpha[1]"], p["beta[1]"], p.get("gamma[1]", 0.0)) < 0:
        reasons.append("paramètre de variance négatif")
    if pers >= float(rules["max_persistence"]):
        reasons.append("persistance ≥ 1 (variance non stationnaire)")
    if lm_p < float(rules["alpha"]) or lb_p < float(rules["alpha"]):
        reasons.append("effet ARCH résiduel")
    return {"persistance": pers, "ARCH_LM_z_p": float(lm_p), "LB_z2_p": lb_p,
            "admissible": not reasons, "motif": "; ".join(reasons) or "—"}


# --------------------------------------------------------------------------- #
# Prévision hors échantillon
# --------------------------------------------------------------------------- #
@dataclass
class GarchForecast:
    spec: GarchSpec
    frame: pd.DataFrame            # colonnes mu, sigma (en %) et nu (loi t), indexées par date
    n_refits: int
    n_not_converged: int
    params: list = field(default_factory=list)   # (date de ré-estimation, paramètres)


def walk_forward(spec: GarchSpec, y: pd.Series, start, end, refit_every: int = 21) -> GarchForecast:
    """Prévisions à une séance de la moyenne et de la volatilité de ``y`` pour les dates de [start, end].

    ``y`` : rendements en %, indexés par date, commençant au début de la fenêtre d'estimation ;
    toutes les lignes antérieures à ``start`` servent d'historique.
    """
    idx = y.index
    x = y.to_numpy(dtype=float)
    pos = np.flatnonzero((idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end)))
    if len(pos) == 0:
        raise ValueError("Aucune séance à prévoir dans l'intervalle demandé.")
    first, last = int(pos[0]), int(pos[-1])
    if first < 250:
        raise ValueError("Historique d'estimation trop court (< 250 séances) avant la première prévision.")

    mu = np.full(len(x), np.nan)
    sig = np.full(len(x), np.nan)
    nu = np.full(len(x), np.nan)
    sv, n_refits, n_bad, params = None, 0, 0, []
    for b in range(first, last + 1, refit_every):
        stop = min(b + refit_every, last + 1)
        # Le modèle ne reçoit que y[:stop] : la séance stop-1 n'est lue que pour prévoir stop (ignoré).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _model(spec, x[:stop]).fit(disp="off", last_obs=b, starting_values=sv,
                                             options={"maxiter": 500})
            # ligne i de la prévision = information jusqu'à i incluse -> distribution de y[i+1]
            f = res.forecast(horizon=1, start=b - 1, reindex=False)
        n_refits += 1
        n_bad += int(not converged(res))
        sv = res.params.to_numpy()
        params.append((idx[b], {k: round(float(v), 6) for k, v in res.params.items()}))
        mu[b:stop] = f.mean.to_numpy()[: stop - b, 0]
        sig[b:stop] = np.sqrt(f.variance.to_numpy()[: stop - b, 0])
        if spec.dist == "t":
            nu[b:stop] = float(res.params["nu"])
    frame = pd.DataFrame({"mu": mu, "sigma": sig, "nu": nu}, index=idx).iloc[first:last + 1]
    return GarchForecast(spec, frame, n_refits, n_bad, params)
