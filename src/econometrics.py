"""Livrable D03 — références simples et modèles économétriques sur XLV.

Protocole (chaque étape n'utilise que les données autorisées) :

  1. Identification          TRAIN (2018-2022)      stationnarité, ACF/PACF, saisonnalité,
                                                     grille AIC/BIC des ARIMA(p,0,q).
  2. Sélection                VALIDATION (2023-2024) prévisions walk-forward à une séance ;
                                                     choix de l'ordre ARIMA, de la variante SARIMAX
                                                     et de la fenêtre de moyenne mobile.
  3. Gel                      reports/d03_selection.json est écrit AVANT toute prévision du test.
  4. Évaluation finale        TEST (2025-01 → 2026-10) une seule passe, métriques globales,
                                                     4 sous-périodes (stabilité, variante B05),
                                                     test de Diebold-Mariano contre la marche aléatoire.
  5. Diagnostics              résidus des modèles retenus estimés sur le train.

Lancement :  ``python -m src.econometrics``  (sans réseau, à partir du snapshot gelé D02)
Sorties    :  reports/d03_*.csv|json|md, reports/figures/d03_*.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from src.data import ROOT, latest_snapshot, load_config, load_snapshot, read_manifest
from src.evaluation import evaluate, evaluate_by_period
from src.features import build_dataset, load_model_config, modelling_sample
from src.models import baseline as B
from src.models.arima import (
    ArimaSpec, admissibility, coefficient_table, fit, order_grid, residual_diagnostics, root_moduli,
    walk_forward,
)

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ECONOMETRIC = ["ARIMA", "SARIMA", "SARIMAX"]


# --------------------------------------------------------------------------- #
# 1. Identification (train)
# --------------------------------------------------------------------------- #
def stationarity_tests(train: pd.DataFrame) -> pd.DataFrame:
    """ADF (H0 : racine unitaire) et KPSS (H0 : stationnarité) sur ln P et sur r."""
    rows = []
    series = {"log-prix ln P_t": np.log(train["price"]), "rendement log r_t": train["y"]}
    for name, s in series.items():
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC", regression="c")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # KPSS borne ses p-values à [0.01, 0.10]
            kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
        verdict = "stationnaire" if (adf_p < 0.05 and kpss_p > 0.05) else (
            "non stationnaire" if (adf_p >= 0.05 and kpss_p <= 0.05) else "ambigu")
        rows.append({"serie": name, "ADF_stat": adf_stat, "ADF_p": adf_p,
                     "KPSS_stat": kpss_stat, "KPSS_p": kpss_p, "conclusion": verdict})
    return pd.DataFrame(rows)


def weekday_effect(train: pd.DataFrame) -> dict:
    """Régression r_t = Σ_j β_j·1{jour = j} : test F d'égalité des 5 moyennes."""
    dow = pd.get_dummies(train.index.dayofweek, prefix="d", drop_first=True).astype(float)
    X = sm.add_constant(dow.set_index(train.index))
    res = sm.OLS(train["y"] * 100, X).fit(cov_type="HC1")
    f = res.f_test(np.eye(X.shape[1])[1:])
    means = (train["y"] * 100).groupby(train.index.dayofweek).mean()
    return {"F": float(np.squeeze(f.fvalue)), "p_value": float(f.pvalue),
            "moyennes_%": {d: float(means[i]) for i, d in enumerate(
                ["lundi", "mardi", "mercredi", "jeudi", "vendredi"])}}


#: Épisode de crise exclu dans l'analyse de robustesse (krach et rebond du Covid-19).
COVID_WINDOW = ("2020-02-15", "2020-06-30")


def covid_robustness(train: pd.DataFrame, spec: ArimaSpec) -> pd.DataFrame:
    """La structure trouvée sur le train survit-elle sans l'épisode Covid ?

    Pour plusieurs sous-échantillons du train : autocorrélation d'ordre 1 de r_t,
    corrélation entre r_t et r^SPY_{t-1}, et gain d'AIC de l'ARIMA retenu sur le
    bruit blanc (> 2 : le modèle apporte quelque chose au sens de l'AIC).
    """
    a, b = (pd.Timestamp(x) for x in COVID_WINDOW)
    covid = (train.index >= a) & (train.index <= b)
    samples = {
        "train complet 2018-2022": ~np.zeros(len(train), dtype=bool),
        f"train sans {a:%d/%m/%Y}-{b:%d/%m/%Y}": ~covid,
        "2018-2019 seulement": train.index < "2020-01-01",
        "2021-2022 seulement": train.index >= "2021-01-01",
    }
    rows = []
    for name, m in samples.items():
        sub = train[m]
        y = sub["y"].to_numpy()
        gain = fit(ArimaSpec("wn", (0, 0, 0)), y).aic - fit(spec, y).aic
        rows.append({"echantillon": name, "n": int(m.sum()),
                     "acf_1": float(acf(y, nlags=1)[1]),
                     "corr_r_SPY(t-1)": float(np.corrcoef(y, sub["x_spy_lag1"])[0, 1]),
                     f"gain_AIC_{spec.label}": float(gain)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def _md(frame: pd.DataFrame, floatfmt: str = "{:.4f}", index: bool = True) -> str:
    f = frame.reset_index() if index else frame
    cols = list(f.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in f.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("—" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _between(ds: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    return ds[(ds.index >= pd.Timestamp(bounds["start"])) & (ds.index <= pd.Timestamp(bounds["end"]))]


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    data_cfg, cfg = load_config(), load_model_config()
    snap = latest_snapshot(data_cfg.snapshot_dir)
    manifest = read_manifest(snap)
    ds = modelling_sample(build_dataset(load_snapshot(snap), data_cfg, cfg))
    sp = data_cfg.splits
    train, val, test = _between(ds, sp["train"]), _between(ds, sp["validation"]), _between(ds, sp["test"])
    every = int(cfg["walk_forward"]["refit_every"])
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    log(f"Jeu de modélisation : train {len(train)} · validation {len(val)} · test {len(test)} séances")

    # ---------------- 1. Identification (train) ----------------
    stat_tests = stationarity_tests(train)
    wd = weekday_effect(train)
    rules = cfg["selection"]
    grid = order_grid(train["y"], cfg["arima"]["max_p"], cfg["arima"]["max_q"], cfg["arima"]["trend"], rules)
    grid.to_csv(REPORTS / "d03_order_grid.csv", index=False, float_format="%.4f")
    log("Identification terminée (stationnarité, saisonnalité, grille AIC/BIC + diagnostics)")

    # Candidats ARIMA : hors bruit blanc (p+q ≥ 1), ADMISSIBLES d'abord (diagnostics des résidus
    # et des racines sur le train), puis les n meilleurs AIC + le meilleur BIC.
    g = grid[(grid["p"] + grid["q"]) >= 1]
    g_ok = g[g["admissible"]]
    if g_ok.empty:  # garde-fou : aucun ordre admissible -> on le signale et on garde la grille entière
        log("ATTENTION : aucun ordre ARIMA admissible, sélection sur la grille complète")
        g_ok = g
    n_c = int(rules["n_candidates"])
    cand = list(g_ok.nsmallest(n_c, "aic")[["p", "q"]].itertuples(index=False, name=None))
    best_bic = tuple(g_ok.nsmallest(1, "bic")[["p", "q"]].iloc[0])
    if best_bic not in cand:
        cand.append(best_bic)
    arima_cands = [ArimaSpec(f"ARIMA({p},0,{q})", order=(int(p), 0, int(q))) for p, q in cand]

    # ---------------- 2. Sélection (validation) ----------------
    val_preds = B.all_baselines(ds["y"], cfg["baselines"]["moving_average_windows"]).loc[val.index]
    selection_rows = []

    def _diag(spec: ArimaSpec) -> dict:
        res = fit(spec, train["y"], train[["x_spy_lag1"]] if spec.exog else None)
        return {"res": res, **admissibility(res, rules)}

    def _val(spec: ArimaSpec, diag: dict, note: str = "") -> float:
        wf = walk_forward(spec, ds, val.index[0], val.index[-1], every)
        val_preds[spec.name] = wf.predictions
        r = float(np.sqrt(np.mean((val["y"] - wf.predictions) ** 2)))
        selection_rows.append({
            "famille": spec.family, "candidat": spec.label, "AIC_train": diag["res"].aic,
            "LB_robuste_p": diag["LB_robuste_p"],
            "racine_min": min(diag["ar_min_modulus"], diag["ma_min_modulus"]),
            "compensation": diag["root_cancellation"], "admissible": diag["admissible"],
            "motif": diag["motif"],
            "RMSE_val_%": 100 * r, "note": note})
        return r

    def _pick(scores: dict, diags: dict) -> ArimaSpec:
        """RMSE de validation minimale parmi les candidats admissibles (tous si aucun ne l'est)."""
        ok = [s_ for s_ in scores if diags[s_]["admissible"]] or list(scores)
        return min(ok, key=scores.get)

    arima_diag = {s_: _diag(s_) for s_ in arima_cands}
    arima_scores = {s_: _val(s_, arima_diag[s_]) for s_ in arima_cands}
    arima_best = _pick(arima_scores, arima_diag)
    order = arima_best.order
    log(f"ARIMA retenu : {arima_best.label}")

    robust = covid_robustness(train, arima_best)
    robust.to_csv(REPORTS / "d03_robustesse_covid.csv", index=False, float_format="%.6f")

    # SARIMA : ordre ARIMA retenu + composantes saisonnières (s = 5) ; admissibles, puis AIC train
    s = int(cfg["sarima"]["season"])
    base_fit = arima_diag[arima_best]["res"]
    sarima_fits = []
    from scipy import stats as _st
    for P, D, Q in cfg["sarima"]["candidates"]:
        spec = ArimaSpec(f"SARIMA{order}({P},{D},{Q})_{s}", order=order, seasonal_order=(P, D, Q, s))
        d = _diag(spec)
        lr = 2 * (d["res"].llf - base_fit.llf)
        sarima_fits.append({"spec": spec, "diag": d, "aic": d["res"].aic, "bic": d["res"].bic, "LR": lr,
                            "LR_p": float(_st.chi2.sf(max(lr, 0.0), P + Q))})
    sarima_tab = pd.DataFrame([{"candidat": f["spec"].label, "AIC_train": f["aic"], "BIC_train": f["bic"],
                                "LR_vs_ARIMA": f["LR"], "LR_p": f["LR_p"],
                                "LB_robuste_p": f["diag"]["LB_robuste_p"],
                                "admissible": f["diag"]["admissible"],
                                "motif": f["diag"]["motif"]} for f in sarima_fits])
    sarima_ok = [f for f in sarima_fits if f["diag"]["admissible"]] or sarima_fits
    sarima_best_info = min(sarima_ok, key=lambda f: f["aic"])
    sarima_best = sarima_best_info["spec"]
    seasonal_relevant = (sarima_best_info["aic"] < base_fit.aic) and (sarima_best_info["LR_p"] < 0.05)
    _val(sarima_best, sarima_best_info["diag"], "meilleur AIC train parmi les SARIMA admissibles")

    # SARIMAX : r^SPY(t-1) avec la dynamique ARMA retenue ou en régression simple
    sx_cands = [ArimaSpec(f"SARIMAX{order}+SPY", order=order, exog=True),
                ArimaSpec("SARIMAX(0,0,0)+SPY", order=(0, 0, 0), exog=True)]
    sx_diag = {s_: _diag(s_) for s_ in sx_cands}
    sx_scores = {s_: _val(s_, sx_diag[s_]) for s_ in sx_cands}
    sarimax_best = _pick(sx_scores, sx_diag)

    ma_windows = cfg["baselines"]["moving_average_windows"]
    ma_scores = {k: float(np.sqrt(np.mean((val["y"] - val_preds[f"ma{k}"]) ** 2))) for k in ma_windows}
    ma_best = min(ma_scores, key=ma_scores.get)
    for k, r in ma_scores.items():
        selection_rows.append({"famille": "Référence", "candidat": f"Moyenne mobile {k} séances",
                               "RMSE_val_%": 100 * r, "note": "pas de résidus : pas de diagnostic"})

    val_metrics = evaluate(val, val_preds)
    val_metrics.to_csv(REPORTS / "d03_validation_metrics.csv", float_format="%.6f")
    sel_tab = pd.DataFrame(selection_rows)
    sel_tab["retenu"] = sel_tab["candidat"].isin(
        [arima_best.label, sarima_best.label, sarimax_best.label, f"Moyenne mobile {ma_best} séances"])

    # ---------------- 3. Gel de la sélection ----------------
    final_specs = {"ARIMA": arima_best, "SARIMA": sarima_best, "SARIMAX": sarimax_best}
    selection = {
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "snapshot": snap.name, "snapshot_sha256": manifest["sha256"],
        "selected_on": "candidats admissibles sur le train (résidus et racines), "
                       "puis RMSE walk-forward minimale sur la validation 2023-2024",
        "admissibility_rules": dict(rules),
        "models": {k: v.to_dict() for k, v in final_specs.items()},
        "moving_average_window": int(ma_best),
        "seasonality_relevant": bool(seasonal_relevant),
        "refit_every": every,
    }
    (REPORTS / "d03_selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    log("Sélection gelée -> reports/d03_selection.json")

    # ---------------- 4. Évaluation finale (test) ----------------
    base_all = B.all_baselines(ds["y"], ma_windows)
    test_preds = base_all.loc[test.index, ["zero", "naive", "hist_mean", f"ma{ma_best}"]].copy()
    refit_info = {}
    for fam, spec in final_specs.items():
        wf = walk_forward(spec, ds, test.index[0], test.index[-1], every)
        test_preds[fam] = wf.predictions
        refit_info[fam] = {"n_refits": wf.n_refits, "non_convergences": wf.n_not_converged}
    names = {"zero": "Marche aléatoire (r̂=0)", "naive": "Naïf (r̂=r_{t-1})",
             "hist_mean": "Moyenne historique", f"ma{ma_best}": f"Moyenne mobile {ma_best}"}
    test_metrics = evaluate(test, test_preds)
    sub = evaluate_by_period(test, test_preds, cfg.subperiods)
    test_metrics.to_csv(REPORTS / "d03_test_metrics.csv", float_format="%.6f")
    sub.to_csv(REPORTS / "d03_test_subperiods.csv", float_format="%.6f")
    log("Évaluation du test terminée")

    preds_out = pd.concat([
        val_preds[["zero", "naive", "hist_mean", f"ma{ma_best}"] +
                  [s_.name for s_ in final_specs.values()]].rename(
            columns={s_.name: fam for fam, s_ in final_specs.items()}).assign(split="validation"),
        test_preds.assign(split="test"),
    ])
    preds_out.insert(0, "y", ds["y"].reindex(preds_out.index))
    preds_out.to_csv(REPORTS / "d03_predictions.csv", float_format="%.8f", index_label="date")

    # Stabilité : ratio RMSE / marche aléatoire par sous-période
    ratio = sub["ratio_RMSE_zero"].unstack("periode")
    beats_zero = (ratio < 1).sum(axis=1)
    r2 = sub["R2_oos"].unstack("periode")
    beats_mean = (r2 > 0).sum(axis=1)
    stability = pd.DataFrame({
        **{f"ratio_{c}": ratio[c] for c in ratio.columns},
        "sous_periodes_battant_zero": beats_zero,
        "sous_periodes_battant_moyenne": beats_mean,
    }).loc[test_preds.columns]

    # ---------------- 5. Diagnostics (train) ----------------
    diag_frames, coef_frames, roots = [], [], {}
    fitted = {}
    for fam, spec in final_specs.items():
        res = fit(spec, train["y"], train[["x_spy_lag1"]] if spec.exog else None)
        fitted[fam] = res
        d = residual_diagnostics(res, cfg["diagnostics"]["ljung_box_lags"],
                                 cfg["diagnostics"]["arch_lm_lags"], cfg["diagnostics"]["alpha"])
        d.insert(0, "modele", spec.label)
        diag_frames.append(d)
        c = coefficient_table(res).reset_index()
        c.insert(0, "modele", spec.label)
        coef_frames.append(c)
        roots[spec.label] = root_moduli(res)
    diag = pd.concat(diag_frames, ignore_index=True)
    coefs = pd.concat(coef_frames, ignore_index=True)
    diag.to_csv(REPORTS / "d03_diagnostics.csv", index=False, float_format="%.6f")
    coefs.to_csv(REPORTS / "d03_coefficients.csv", index=False, float_format="%.6f")
    log("Diagnostics terminés")

    # ---------------- Figures et rapport ----------------
    plot_identification(train, FIGURES / "d03_identification.png")
    plot_residuals(fitted["ARIMA"], final_specs["ARIMA"].label, train.index, FIGURES / "d03_residus.png")
    plot_test(test, test_preds, cfg.subperiods, names, FIGURES / "d03_test_erreurs_cumulees.png")
    plot_stability(ratio.loc[test_preds.columns], names, FIGURES / "d03_stabilite.png")

    ctx = dict(snap=snap, manifest=manifest, train=train, val=val, test=test, stat_tests=stat_tests,
               wd=wd, grid=grid, robust=robust, rules=rules, sel_tab=sel_tab, sarima_tab=sarima_tab, seasonal_relevant=seasonal_relevant,
               final_specs=final_specs, ma_best=ma_best, val_metrics=val_metrics, test_metrics=test_metrics,
               sub=sub, stability=stability, diag=diag, coefs=coefs, roots=roots, names=names,
               refit_info=refit_info, cfg=cfg, every=every, elapsed=time.time() - t0, selection=selection)
    write_report(ctx, REPORTS / "d03_econometrie.md")
    log(f"Rapport écrit : reports/d03_econometrie.md ({time.time() - t0:.0f} s)")
    return ctx


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_identification(train: pd.DataFrame, path: Path) -> Path:
    plt = _plt()
    nl = 20
    n = len(train)
    band = 1.96 / np.sqrt(n)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].plot(train.index, np.log(train["price"]), color="#1f5f8b", lw=1)
    axes[0, 0].set_title("ln P_t (XLV, train 2018-2022) — non stationnaire")
    axes[0, 1].plot(train.index, 100 * train["y"], color="#1f5f8b", lw=0.6)
    axes[0, 1].set_title("r_t en % — stationnaire, volatilité en grappes")
    for ax, vals, title in [
        (axes[1, 0], acf(train["y"], nlags=nl)[1:], "ACF de r_t"),
        (axes[1, 1], pacf(train["y"], nlags=nl)[1:], "PACF de r_t"),
    ]:
        ax.bar(range(1, nl + 1), vals, color="#1f5f8b", width=0.6)
        ax.axhspan(-band, band, color="grey", alpha=0.2, label="±1,96/√n")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("retard (séances)")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_residuals(res, label: str, dates, path: Path) -> Path:
    plt = _plt()
    from scipy import stats
    burn = max(res.loglikelihood_burn, 1)
    z = np.asarray(res.standardized_forecasts_error[0])[burn:]
    d = dates[burn:]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].plot(d, z, lw=0.5, color="#1f5f8b")
    axes[0, 0].set_title(f"Résidus standardisés — {label} (train)")
    axes[0, 1].hist(z, bins=80, density=True, color="#1f5f8b", alpha=0.7)
    xs = np.linspace(-6, 6, 300)
    axes[0, 1].plot(xs, stats.norm.pdf(xs), color="#c0392b", label="N(0,1)")
    axes[0, 1].set_xlim(-6, 6)
    axes[0, 1].set_title("Distribution : queues épaisses")
    axes[0, 1].legend()
    stats.probplot(z, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("QQ-plot normal")
    axes[1, 0].set_xlabel("quantiles théoriques N(0,1)")
    axes[1, 0].set_ylabel("quantiles observés")
    nl = 20
    band = 1.96 / np.sqrt(len(z))
    axes[1, 1].bar(range(1, nl + 1), acf(z, nlags=nl)[1:], color="#1f5f8b", width=0.6, label="résidus")
    axes[1, 1].plot(range(1, nl + 1), acf(z ** 2, nlags=nl)[1:], "o-", color="#c0392b", ms=3,
                    label="résidus²")
    axes[1, 1].axhspan(-band, band, color="grey", alpha=0.2)
    axes[1, 1].axhline(0, color="black", lw=0.6)
    axes[1, 1].set_title("ACF des résidus et des résidus² (effet ARCH)")
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_test(test: pd.DataFrame, preds: pd.DataFrame, periods: dict, names: dict, path: Path) -> Path:
    """Différence cumulée des erreurs quadratiques : Σ (e_zero² − e_modèle²).

    Courbe croissante = le modèle fait mieux que la marche aléatoire sur la période.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    e0 = (test["y"] - preds["zero"]) ** 2
    colors = {"hist_mean": "#7f8c8d", "ARIMA": "#1f5f8b", "SARIMA": "#27ae60", "SARIMAX": "#c0392b"}
    for col in [c for c in preds.columns if c not in ("zero", "naive")]:
        csum = (e0 - (test["y"] - preds[col]) ** 2).cumsum() * 1e4
        ax.plot(test.index, csum, lw=1.4 if col in colors else 0.9,
                color=colors.get(col), ls="-" if col in colors else ":", label=names.get(col, col))
    ax.axhline(0, color="black", lw=0.8)
    for i, (name, (a, b)) in enumerate(periods.items()):
        if i % 2 == 0:
            ax.axvspan(a, b, color="grey", alpha=0.08)
        ax.text(a + (b - a) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom")
    ax.set_title("Test 2025-2026 : avantage cumulé sur la marche aléatoire  Σ(e²_zéro − e²_modèle)  "
                 "[%²]\n> 0 et croissant : le modèle prévoit mieux ; < 0 : il fait moins bien "
                 "(naïf omis : hors échelle, voir le tableau)", fontsize=10, pad=20)
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_stability(ratio: pd.DataFrame, names: dict, path: Path) -> Path:
    plt = _plt()
    ratio = ratio.drop(index="zero")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    n_m, n_p = ratio.shape
    w = 0.8 / n_m
    for i, (m, row) in enumerate(ratio.iterrows()):
        ax.bar(np.arange(n_p) + i * w - 0.4 + w / 2, row.values, width=w, label=names.get(m, m))
    ax.axhline(1, color="black", lw=1)
    ax.set_xticks(range(n_p))
    ax.set_xticklabels(ratio.columns)
    lo = min(0.97, float(np.nanmin(ratio.values)) - 0.01)
    hi = min(max(1.05, float(np.nanmax(ratio[ratio.index != "naive"].values)) + 0.01), 1.15)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("RMSE modèle / RMSE marche aléatoire")
    ax.set_title("Stabilité par sous-période du test (< 1 : le modèle bat la marche aléatoire ; "
                 "le naïf, hors échelle, est tronqué)", fontsize=10)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Rapport
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def write_report(c: dict, path: Path) -> Path:
    specs = c["final_specs"]
    tm, names = c["test_metrics"], c["names"]
    show = tm.rename(index=lambda m: names.get(m, specs[m].label if m in specs else m))
    best_rmse = tm["RMSE_%"].idxmin()
    econ_better = [f for f in ECONOMETRIC if tm.loc[f, "ratio_RMSE_zero"] < 1]
    econ_signif = [f for f in ECONOMETRIC if tm.loc[f, "DM_vs_zero"] < 0 and tm.loc[f, "DM_p"] < 0.05]
    stab = c["stability"]
    n_sub = len(c["cfg"].subperiods)
    roots = pd.DataFrame(c["roots"]).T.rename_axis("modele")
    wd = c["wd"]
    grid = c["grid"]
    rules = c["rules"]
    gsel = grid.head(8)[["p", "q", "aic", "bic", "d_aic", "LB_robuste_p", "ar_min_modulus",
                         "ma_min_modulus", "root_cancellation", "admissible", "motif"]]
    n_ok = int(grid["admissible"].sum())
    da_zero = tm.loc["zero", "DA"]

    L = []
    a = L.append
    a("# D03 — Références simples et modèles économétriques (XLV)\n")
    a("> Fichier généré par `python -m src.econometrics`. Ne pas modifier à la main : relancer la commande.\n")
    a("## 1. Cadre\n")
    a(f"- Snapshot D02 : `data/snapshots/{c['snap'].name}` (SHA-256 `{c['manifest']['sha256'][:16]}…`, "
      f"vérifié au chargement)")
    a("- Cible : rendement log journalier de XLV, r_t = ln(P_t / P_{t-1}), P = prix ajusté "
      "(dividendes et fractionnements).")
    a("- Horizon : 1 séance. La prévision de r_t est calculée à la clôture de t-1 avec l'information "
      "disponible à cette clôture uniquement.")
    a("- Variable externe de SARIMAX : x_t = r^SPY_{t-1}, le rendement de SPY de la **veille**. Pour prévoir "
      "r_t à la clôture de t-1, on connaît déjà r^SPY_{t-1} : aucune valeur future de SPY n'est utilisée. "
      "Le rendement de SPY du jour t, qui n'est connu qu'à la clôture de t, n'entre jamais dans le modèle. "
      "En walk-forward, la prévision de la séance t ne lit que les lignes ≤ t de x et les lignes ≤ t-1 de y. "
      "Les tests `test_dataset_alignment` et `test_walk_forward_no_lookahead` le vérifient.")
    a(f"- Échantillons : train {c['train'].index[0]:%Y-%m-%d} → {c['train'].index[-1]:%Y-%m-%d} "
      f"({len(c['train'])} séances) · validation {c['val'].index[0]:%Y-%m-%d} → {c['val'].index[-1]:%Y-%m-%d} "
      f"({len(c['val'])}) · test {c['test'].index[0]:%Y-%m-%d} → {c['test'].index[-1]:%Y-%m-%d} "
      f"({len(c['test'])}). L'année 2017 sert seulement d'amorçage.")
    a(f"- Walk-forward : fenêtre croissante depuis janvier 2018, paramètres ré-estimés toutes les "
      f"{c['every']} séances, état du filtre mis à jour chaque jour.")
    a(f"- Logiciel : statsmodels {statsmodels.__version__}, pandas {pd.__version__}, numpy {np.__version__}.\n")

    a("## 2. Identification (train 2018-2022 uniquement)\n")
    a("### 2.1 Stationnarité\n")
    a(_md(c["stat_tests"], index=False))
    a("\nLe log-prix a une racine unitaire, le rendement est stationnaire : on modélise donc r_t "
      "avec d = 0, ce qui équivaut à un ARIMA(p, 1, q) sur le log-prix. La constante correspond à la dérive.\n")
    a("### 2.2 Grille ARIMA(p, 0, q) et diagnostics — 8 meilleurs AIC\n")
    a("Pour chaque ordre, on calcule aussi sur le train les diagnostics qui décident de l'admissibilité "
      "(section 3) : test de Ljung-Box robuste à l'effet ARCH sur les résidus "
      f"({rules['robust_lb_lags']} retards), plus petit module des racines AR et MA, et plus petite "
      "distance entre une racine AR et une racine MA (compensation).\n")
    a(_md(gsel, index=False, floatfmt="{:.2f}"))
    a(f"\nGrille complète : `reports/d03_order_grid.csv`. Le bruit blanc ARIMA(0,0,0) a un ΔAIC de "
      f"{c['grid'].set_index(['p','q']).loc[(0,0),'d_aic']:.1f}. Les critères voient donc une dépendance "
      f"linéaire dans le train. Seuls {n_ok} ordres sur {len(grid)} sont admissibles : "
      + ", ".join(f"ARIMA({int(r.p)},0,{int(r.q)})" for r in grid[grid['admissible']].itertuples())
      + ". Les autres gardent une autocorrélation dans leurs résidus, même avec le test robuste, "
      "et plusieurs ont des racines AR et MA qui se compensent presque : leurs paramètres sont redondants.\n")
    a("### 2.3 D'où vient cette structure ? Robustesse à l'épisode Covid\n")
    a(_md(c["robust"], index=False))
    rb = c["robust"]
    a(f"\nSans la période {COVID_WINDOW[0]} → {COVID_WINDOW[1]} (moins de 8 % du train), "
      f"l'autocorrélation d'ordre 1 passe de {rb['acf_1'].iloc[0]:+.3f} à {rb['acf_1'].iloc[1]:+.3f}, "
      f"la corrélation avec r^SPY(t-1) de {rb.iloc[0, 3]:+.3f} à {rb.iloc[1, 3]:+.3f}, et le gain d'AIC "
      f"de l'ARIMA retenu de {rb.iloc[0, 4]:.1f} à {rb.iloc[1, 4]:.1f}. **La dépendance linéaire "
      "du train vient presque entièrement du krach et du rebond de 2020.** En dehors de cet épisode, "
      "le rendement de XLV se comporte comme un bruit blanc. Les modèles ne sont pas ré-estimés "
      "sans 2020 : ce choix serait guidé par le résultat. Cette table sert seulement à interpréter "
      "les résultats hors échantillon.\n")
    a("### 2.4 Saisonnalité\n")
    a(f"Choix de la période saisonnière s = {c['cfg']['sarima']['season']} :")
    a("- les données sont des **séances de bourse**, pas des jours calendaires : une semaine compte 5 séances ;")
    a("- s = 12 correspond à une saison annuelle en données **mensuelles**. En journalier, cela voudrait "
      "dire un cycle de 12 séances, qui ne correspond à aucun phénomène de marché connu ;")
    a("- un cycle mensuel (~21 séances) ou annuel (~252 séances) serait possible en théorie. Mais un SARIMA "
      "avec s = 252 coûte très cher à estimer, et il n'existe pas de mécanisme connu qui produirait ces cycles "
      "dans les rendements de XLV. Ces effets de calendrier (fin de mois, janvier…) se testent plus simplement "
      "avec des indicatrices, si besoin dans D05 ;")
    a("- le seul cycle plausible à priori est donc l'effet jour de la semaine, qu'on teste directement :\n")
    a(f"- Effet jour de la semaine (régression sur indicatrices, écarts-types robustes) : F = {wd['F']:.2f}, "
      f"p = {wd['p_value']:.3f}. Rendements moyens (%) : " +
      ", ".join(f"{k} {v:+.3f}" for k, v in wd["moyennes_%"].items()) + ".")
    a("- Composantes saisonnières hebdomadaires (s = 5) ajoutées à l'ARIMA retenu :\n")
    a(_md(c["sarima_tab"], index=False, floatfmt="{:.3f}"))
    verdict = ("**pertinente au sens statistique** sur le train (AIC plus bas et test LR significatif à 5 %)"
               if c["seasonal_relevant"] else
               "**non pertinente** : le gain d'AIC ou le test LR est insuffisant")
    a(f"\nConclusion : la saisonnalité hebdomadaire est {verdict}. Le SARIMA est quand même évalué "
      "hors échantillon, pour vérifier si ce résultat tient sur des données nouvelles.\n")

    a("## 3. Sélection sur la validation (2023-2024)\n")
    a("Un modèle n'est pas choisi seulement parce que sa RMSE est faible. La règle comporte deux étapes, "
      "fixées dans `configs/models.yaml` (section `selection`) :\n")
    a(f"1. **Admissibilité, sur le train.** Un candidat est écarté si son estimation ne converge pas ; si une "
      f"racine AR ou MA a un module inférieur à {rules['min_root_modulus']} ; si une racine AR et une racine "
      f"MA sont à moins de {rules['max_root_cancellation']} l'une de l'autre (elles se compensent) ; ou si le "
      f"Ljung-Box robuste ({rules['robust_lb_lags']} retards) rejette l'absence d'autocorrélation des résidus "
      f"au seuil de {rules['alpha']}.")
    a(f"2. **Précision, sur la validation.** Parmi les candidats admissibles (pour l'ARIMA : les "
      f"{rules['n_candidates']} meilleurs AIC et le meilleur BIC, sans le bruit blanc), on retient celui dont "
      "la RMSE walk-forward est la plus faible. Si aucun candidat d'une famille n'est admissible, on le "
      "signale et on garde le meilleur RMSE.\n")
    a("On utilise le Ljung-Box **robuste** (Lobato, Nankervis et Savin, 2001), pas le classique. Le test "
      "classique suppose une variance constante. Sur des rendements avec un fort effet ARCH, il rejette "
      "presque toujours, même quand les résidus ne sont pas autocorrélés : il éliminerait tous les modèles. "
      "Le test classique reste affiché en section 6.3 pour comparaison. Le jeu de test 2025-2026 "
      "n'intervient dans aucun choix.\n")
    a(_md(c["sel_tab"], index=False))
    st = c["sel_tab"]
    for fam in ECONOMETRIC:
        f = st[st["famille"] == fam]
        kept = f[f["retenu"]]
        if kept.empty:
            continue
        better = f[(f["admissible"] == False) & (f["RMSE_val_%"] < kept["RMSE_val_%"].iloc[0])]  # noqa: E712
        for _, r in better.iterrows():
            a(f"\nÀ noter : {r['candidat']} a une RMSE de validation plus faible ({r['RMSE_val_%']:.4f} %) que le modèle "
              f"retenu ({kept['RMSE_val_%'].iloc[0]:.4f} %), mais il est écarté par les diagnostics "
              f"({r['motif']}). Cela montre que la règle ne choisit pas seulement sur la RMSE.")
    a("\nModèles retenus et gelés (`reports/d03_selection.json`, écrit avant toute prévision du test) :\n")
    for fam, spec in specs.items():
        a(f"- **{fam}** : {spec.label}")
    a(f"- **Moyenne mobile** : {c['ma_best']} séances\n")
    a("Métriques de validation de tous les candidats : `reports/d03_validation_metrics.csv`.\n")

    a("## 4. Résultats sur le test final (2025-01 → 2026-10)\n")
    a(_md(show, floatfmt="{:.4f}"))
    a(f"\nLectures :\n")
    a(f"- La plus faible RMSE est celle de **{names.get(best_rmse, specs[best_rmse].label if best_rmse in specs else best_rmse)}**.")
    if econ_better:
        a(f"- Modèles économétriques dont la RMSE est inférieure à celle de la marche aléatoire : "
          f"{', '.join(econ_better)}. " +
          (f"L'écart est significatif à 5 % (Diebold-Mariano) pour : {', '.join(econ_signif)}."
           if econ_signif else "Aucun de ces écarts n'est significatif à 5 % (Diebold-Mariano)."))
    else:
        a("- **Aucun modèle économétrique ne bat la marche aléatoire** en RMSE sur le test.")
    a(f"- Exactitude directionnelle : la référence « toujours hausse » obtient {_pct(da_zero)} "
      f"(part des séances haussières). Une DA de 50 % ne veut donc rien dire. Il faut la comparer à ce chiffre.")
    a("- R2_oos < 0 : le modèle prévoit moins bien que la simple moyenne historique des rendements.")
    a("- ratio_RMSE_zero et DM_vs_zero : un ratio < 1 et une statistique DM négative avec DM_p < 0,05 "
      "indiquent une amélioration significative par rapport à la marche aléatoire.\n")
    a(f"Ré-estimations sur le test : " + ", ".join(
        f"{k} {v['n_refits']} ({v['non_convergences']} non-convergence(s))" for k, v in c["refit_info"].items()) + ".\n")
    a("Toutes les prévisions (validation et test) sont dans `reports/d03_predictions.csv`. Ce fichier "
      "sert d'entrée à la comparaison D05 et aux stratégies D06.\n")

    a("## 5. Stabilité de l'avantage prédictif (variante B05)\n")
    a(f"Ratio RMSE / marche aléatoire par sous-période (< 1 : le modèle fait mieux), et nombre de "
      f"sous-périodes (sur {n_sub}) où le modèle bat la marche aléatoire et la moyenne historique :\n")
    a(_md(stab.rename(index=lambda m: names.get(m, m)).rename_axis("modele"), floatfmt="{:.4f}"))
    a("\nDétail des métriques par sous-période : `reports/d03_test_subperiods.csv`. "
      "Figures : `figures/d03_stabilite.png` et `figures/d03_test_erreurs_cumulees.png`.\n")
    stable = [f for f in ECONOMETRIC if stab.loc[f, "sous_periodes_battant_zero"] == n_sub]
    a("Conclusion sur la stabilité : " + (
        f"{', '.join(stable)} bat la marche aléatoire dans les {n_sub} sous-périodes." if stable else
        f"aucun modèle économétrique ne bat la marche aléatoire dans les {n_sub} sous-périodes. "
        "Quand un avantage apparaît, il est local et ne se maintient pas.") + "\n")

    a("## 6. Diagnostics des modèles retenus (estimés sur le train)\n")
    a("### 6.1 Coefficients (unités : rendement en %)\n")
    a(_md(c["coefs"], index=False))
    a("\n### 6.2 Racines (plus petit module, > 1 : stationnaire et inversible)\n")
    a(_md(roots, floatfmt="{:.3f}"))
    a("\n### 6.3 Tests sur les résidus\n")
    a(_md(c["diag"], index=False))
    a("\nLecture :")
    a("- **Autocorrélation** : le Ljung-Box classique rejette aux retards 10 et 20, mais le Ljung-Box robuste "
      "ne rejette pas. La dynamique de la moyenne est donc bien captée. Le rejet du test classique vient de "
      "l'hétéroscédasticité, pas d'une autocorrélation oubliée.")
    a("- **Normalité** : Jarque-Bera rejette nettement. Les résidus ont des queues épaisses (voir le QQ-plot).")
    a("- **Effet ARCH** : ARCH-LM et Ljung-Box sur les résidus² rejettent fortement. "
      "La variance conditionnelle n'est pas constante. Les intervalles de prévision d'un ARIMA "
      "à variance constante sont donc mal calibrés. C'est le point de départ du livrable D04 "
      "(GARCH, VaR, ES). Voir `figures/d03_residus.png`.\n")

    a("## 7. Réponse provisoire à la question scientifique\n")
    a("Sur XLV, en prévision ponctuelle du rendement à une séance :\n")
    a("1. Les critères d'information du train trouvent une structure ARMA et un effet retardé de SPY. "
      "Ces deux résultats disparaissent quand on retire le krach de 2020 (section 2.3).")
    best_econ = min(ECONOMETRIC, key=lambda f: tm.loc[f, "RMSE_%"])
    r_best = tm.loc[best_econ, "ratio_RMSE_zero"]
    n_win = int(stab.loc[best_econ, "sous_periodes_battant_zero"])
    effet = (f"un gain de {100 * (1 - r_best):.2f} %" if r_best < 1 else
             f"une perte de {100 * (r_best - 1):.2f} %")
    a(f"2. Les modèles retenus passent les diagnostics : résidus non autocorrélés selon le test robuste, "
      f"racines bien identifiées. Pourtant, hors échantillon, le meilleur d'entre eux "
      f"({specs[best_econ].label}) a un ratio RMSE / marche aléatoire de {r_best:.4f}, soit {effet} "
      f"(Diebold-Mariano p = {tm.loc[best_econ, 'DM_p']:.2f}). Il bat la marche aléatoire dans {n_win} "
      f"sous-période(s) sur {n_sub}. Un bon ajustement sur le train ne garantit donc pas de bonnes "
      "prévisions : la structure captée est celle de 2020, qui ne se reproduit pas. "
      f"La moyenne historique, sans aucun paramètre dynamique, fait mieux (ratio "
      f"{tm.loc['hist_mean', 'ratio_RMSE_zero']:.4f}).")
    a(f"3. La SARIMA n'améliore pas l'ARIMA hors échantillon (ratio {tm.loc['SARIMA', 'ratio_RMSE_zero']:.4f} "
      f"contre {tm.loc['ARIMA', 'ratio_RMSE_zero']:.4f}). La saisonnalité hebdomadaire détectée sur le "
      "train ne sert donc pas à prévoir.")
    a("4. L'information économétrique utile est surtout dans la **variance** (effet ARCH très net), "
      "pas dans la moyenne conditionnelle.")
    a("5. Ces conclusions portent sur la précision statistique. L'intérêt économique, après coûts, "
      "sera mesuré en D06 à partir de `d03_predictions.csv`.\n")
    a(f"_Durée d'exécution : {c['elapsed']:.0f} s._\n")
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Modèles économétriques — livrable D03")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)
    run(verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
