"""Livrable D04 — module de risque : effet ARCH, GARCH(1,1), volatilité prévue, VaR et ES sur XLV.

Protocole (chaque étape n'utilise que les données autorisées) :

  1. Effet ARCH              TRAIN (2018-2022)       ARCH-LM et Ljung-Box sur r² et sur les résidus
                                                      de l'ARIMA retenu en D03.
  2. Estimation               TRAIN                   GARCH(1,1) normal (exigé), GARCH(1,1)-t,
                                                      GJR-GARCH(1,1)-t ; diagnostics des résidus
                                                      standardisés et règle d'admissibilité.
  3. Sélection                VALIDATION (2023-2024)  volatilité prévue en walk-forward, perte QLIKE ;
                                                      choix du modèle GARCH principal.
  4. Gel                      reports/d04_selection.json est écrit AVANT toute prévision du test.
  5. Évaluation finale        TEST (2025-01 → 2026-10) une seule passe : volatilité prévue confrontée
                                                      aux rendements réalisés, backtests VaR (Kupiec,
                                                      Christoffersen, feux de Bâle) et ES (McNeil-Frey),
                                                      stabilité sur les sous-périodes T1–T4 (variante B05).

Références simples comparées : EWMA RiskMetrics (λ = 0,94), loi normale à moments constants,
simulation historique sur 250 séances. Toutes les prévisions de la séance t n'utilisent que r_{≤t-1}.

Lancement :  ``python -m src.risk_management``  (sans réseau, à partir du snapshot figé D02)
Sorties    :  reports/d04_*.csv|json|md, reports/figures/d04_*.png
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
import yaml
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src import risk as R
from src.data import ROOT, get_ticker, load_config, load_snapshot, read_manifest
from src.econometrics import _md
from src.features import build_dataset, load_model_config
from src.models import garch as G
from src.models.arima import ArimaSpec, fit as arima_fit, model_residuals

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DEFAULT_RISK_CONFIG = ROOT / "configs" / "risk.yaml"

#: identifiants courts des modèles (colonnes du CSV) -> libellés
BASELINES = {"ewma": "EWMA (λ=0,94)", "const": "Normale constante", "hs": "Simulation historique 250 j"}
COLORS = {"garch_n": "#1f5f8b", "garch_t": "#27ae60", "gjr_t": "#8e44ad", "ewma": "#e67e22",
          "const": "#7f8c8d", "hs": "#c0392b"}


def load_risk_config(path: Path | str = DEFAULT_RISK_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _slug(name: str) -> str:
    return {"GARCH(1,1)-N": "garch_n", "GARCH(1,1)-t": "garch_t", "GJR-GARCH(1,1)-t": "gjr_t"}.get(
        name, name.lower().replace("(", "").replace(")", "").replace(",", "").replace("-", "_"))


def _between(s, bounds: dict):
    return s[(s.index >= pd.Timestamp(bounds["start"])) & (s.index <= pd.Timestamp(bounds["end"]))]


def parkinson_proxy(snap: pd.DataFrame, ticker: str = "XLV") -> pd.Series:
    """Variance réalisée de Parkinson (1980) en %² : (ln(H/L))² / (4 ln 2).

    Le rapport H/L ne dépend pas de l'ajustement des dividendes. Ce proxy ne voit que la séance
    (pas l'écart de la nuit) : il sert à vérifier le CLASSEMENT des modèles, pas leur niveau.
    """
    x = get_ticker(snap, ticker)
    return (1e4 * np.log(x["high"] / x["low"]) ** 2 / (4 * np.log(2))).rename("parkinson")


# --------------------------------------------------------------------------- #
# 1. Effet ARCH (train)
# --------------------------------------------------------------------------- #
def arch_effect_tests(r_train: pd.Series, arima_resid: np.ndarray, arima_label: str,
                      alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    series = {"rendement centré r_t − r̄": (r_train - r_train.mean()).to_numpy(),
              f"résidus {arima_label} (D03)": np.asarray(arima_resid)}
    for name, e in series.items():
        for lags in (5, 10):
            lm, p, _, _ = het_arch(e, nlags=lags)
            rows.append({"serie": name, "test": f"ARCH-LM ({lags} retards)", "statistique": float(lm),
                         "p_value": float(p)})
        lb = acorr_ljungbox(e ** 2, lags=[10])
        rows.append({"serie": name, "test": "Ljung-Box sur le carré (10 retards)",
                     "statistique": float(lb["lb_stat"].iloc[0]), "p_value": float(lb["lb_pvalue"].iloc[0])})
    out = pd.DataFrame(rows)
    out["h0"] = "pas d'effet ARCH"
    out["verdict"] = np.where(out["p_value"] < alpha, "effet ARCH (H0 rejetée)", "H0 non rejetée")
    return out


# --------------------------------------------------------------------------- #
# Prévisions de risque d'un modèle sur une période
# --------------------------------------------------------------------------- #
def risk_frame(mu, sigma, levels, dist="normal", nu=None) -> pd.DataFrame:
    out = pd.DataFrame({"mu": mu, "sigma": sigma})
    for lv in levels:
        v, e = R.parametric_var_es(out["mu"], out["sigma"], lv, dist, nu)
        k = int(round(lv * 100))
        out[f"VaR{k}"], out[f"ES{k}"] = v, e
    return out


def forecasts_for(window: pd.Index, r_all: pd.Series, y_est: pd.Series, specs: list, cfg: dict,
                  every: int) -> tuple[dict, dict]:
    """Toutes les prévisions (GARCH en walk-forward + références) pour les dates ``window``."""
    levels = cfg["risk_measures"]["var_levels"]
    out, info = {}, {}
    for spec in specs:
        wf = G.walk_forward(spec, y_est, window[0], window[-1], every)
        f = wf.frame.reindex(window)
        nu = f["nu"].to_numpy() if spec.dist == "t" else None
        out[_slug(spec.name)] = risk_frame(f["mu"], f["sigma"], levels, spec.dist, nu)
        info[_slug(spec.name)] = {"n_refits": wf.n_refits, "non_convergences": wf.n_not_converged,
                                  "derniers_parametres": wf.params[-1][1]}
    b = cfg["baselines"]
    ew = R.ewma_sigma(r_all, b["ewma_lambda"]).reindex(window)
    out["ewma"] = risk_frame(pd.Series(0.0, index=window), ew, levels)
    cm, cs = R.constant_moments(r_all)
    out["const"] = risk_frame(cm.reindex(window), cs.reindex(window), levels)
    hs = pd.DataFrame({"mu": r_all.rolling(b["hist_window"]).mean().shift(1),
                       "sigma": r_all.rolling(b["hist_window"]).std().shift(1)})
    for lv in levels:
        v, e = R.historical_var_es(r_all, lv, b["hist_window"])
        k = int(round(lv * 100))
        hs[f"VaR{k}"], hs[f"ES{k}"] = v, e
    out["hs"] = hs.reindex(window)
    return out, info


def volatility_table(r: pd.Series, fc: dict, labels: dict, proxy: pd.Series) -> pd.DataFrame:
    rows = []
    l_ewma, l_const = R.qlike(r, fc["ewma"]["sigma"]), R.qlike(r, fc["const"]["sigma"])
    for k, f in fc.items():
        ql = R.qlike(r, f["sigma"])
        mz = R.mincer_zarnowitz(r, f["sigma"])
        dm_e = R.loss_dm_test(ql, l_ewma) if k != "ewma" else (np.nan, np.nan)
        dm_c = R.loss_dm_test(ql, l_const) if k != "const" else (np.nan, np.nan)
        rows.append({"modele": labels[k], "QLIKE": float(np.mean(ql)),
                     "ecart_QLIKE_vs_EWMA": float(np.mean(ql - l_ewma)), "DM_vs_EWMA_p": dm_e[1],
                     "ecart_QLIKE_vs_constante": float(np.mean(ql - l_const)), "DM_vs_constante_p": dm_c[1],
                     "QLIKE_Parkinson": float(np.mean(R.qlike(np.sqrt(proxy.reindex(r.index)), f["sigma"]))),
                     "MSE_variance": float(np.mean(R.mse_var(r, f["sigma"]))),
                     "MZ_a": mz["a"], "MZ_b": mz["b"], "MZ_R2": mz["R2"], "MZ_wald_p": mz["wald_p"],
                     "sigma_moy_%": float(f["sigma"].mean())})
    return pd.DataFrame(rows).set_index("modele")


def var_tables(r: pd.Series, fc: dict, labels: dict, levels) -> pd.DataFrame:
    frames = []
    for lv in levels:
        k = int(round(lv * 100))
        t = R.var_backtest_table(r, {labels[m]: f[f"VaR{k}"] for m, f in fc.items()}, lv)
        frames.append(t)
    return pd.concat(frames)


def es_table(r: pd.Series, fc: dict, labels: dict, levels) -> pd.DataFrame:
    rows = []
    for lv in levels:
        k = int(round(lv * 100))
        for m, f in fc.items():
            bt = R.es_backtest(r, f[f"VaR{k}"], f[f"ES{k}"], f["sigma"])
            rows.append({"modele": labels[m], "niveau": lv, **bt,
                         "FZ0": float(np.mean(R.fz0_loss(r, f[f"VaR{k}"], f[f"ES{k}"], lv)))})
    return pd.DataFrame(rows).set_index("modele")


def subperiod_table(r: pd.Series, fc: dict, labels: dict, periods: dict) -> pd.DataFrame:
    rows = []
    for name, (a, b) in periods.items():
        m = (r.index >= a) & (r.index <= b)
        rr = r[m]
        l_ewma = R.qlike(rr, fc["ewma"]["sigma"][m])
        for k, f in fc.items():
            ql = R.qlike(rr, f["sigma"][m])
            h99 = R.hits(rr, f["VaR99"][m])
            h95 = R.hits(rr, f["VaR95"][m])
            rows.append({"periode": name, "modele": labels[k], "n": int(m.sum()),
                         "QLIKE": float(np.mean(ql)), "ecart_QLIKE_vs_EWMA": float(np.mean(ql - l_ewma)),
                         "exceptions_95": int(h95.sum()), "attendues_95": 0.05 * m.sum(),
                         "Kupiec95_p": R.kupiec_pof(h95, 0.95)[1],
                         "exceptions_99": int(h99.sum()), "attendues_99": 0.01 * m.sum(),
                         "Kupiec99_p": R.kupiec_pof(h99, 0.99)[1]})
    return pd.DataFrame(rows).set_index(["periode", "modele"])


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    data_cfg, mcfg, cfg = load_config(), load_model_config(), load_risk_config()
    snap_path = data_cfg.snapshot_dir / cfg["snapshot"]
    manifest = read_manifest(snap_path)
    snap = load_snapshot(snap_path)                       # empreinte SHA-256 vérifiée
    ds = build_dataset(snap, data_cfg, mcfg)              # contient 2017 (amorçage)
    r_all = (100 * ds["y"]).rename("r")                   # rendement log en %
    sp = data_cfg.splits
    y_est = r_all[ds["split"] != "warmup"]                # estimation GARCH : à partir de 2018
    train, val, test = _between(r_all, sp["train"]), _between(r_all, sp["validation"]), _between(r_all, sp["test"])
    every = int(cfg["walk_forward"]["refit_every"])
    levels = cfg["risk_measures"]["var_levels"]
    proxy = parkinson_proxy(snap)
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    log(f"Snapshot {snap_path.name} · train {len(train)} · validation {len(val)} · test {len(test)} séances")

    # ---------------- 1. Effet ARCH (train) ----------------
    d03 = json.loads((REPORTS / "d03_selection.json").read_text(encoding="utf-8"))
    a = d03["models"]["ARIMA"]
    arima_spec = ArimaSpec(a["name"], order=tuple(a["order"]), trend=a["trend"])
    arima_res = arima_fit(arima_spec, train / 100)
    arch_tests = arch_effect_tests(train, model_residuals(arima_res), arima_spec.label)
    arch_tests.to_csv(REPORTS / "d04_arch_tests.csv", index=False, float_format="%.6f")
    log("Effet ARCH testé sur le train")

    # ---------------- 2. Estimation et diagnostics (train) ----------------
    specs = [G.GarchSpec(c["name"], int(c["o"]), c["dist"]) for c in cfg["garch_candidates"]]
    labels = {_slug(s.name): s.name for s in specs} | BASELINES
    fits, summ, coefs, diags, adm = {}, [], [], [], {}
    for s in specs:
        res = G.fit(s, train)
        fits[s.name] = res
        summ.append(G.summary_row(s, res))
        coefs.append(G.coefficient_table(s, res))
        diags.append(G.residual_diagnostics(s, res, cfg["admissibility"]["arch_lm_lags"],
                                            cfg["admissibility"]["robust_lb_lags"], cfg["admissibility"]["alpha"]))
        adm[s.name] = G.admissibility(s, res, cfg["admissibility"])
    summary = pd.DataFrame(summ).set_index("modele")
    summary = summary.join(pd.DataFrame(adm).T[["ARCH_LM_z_p", "LB_z2_p", "admissible", "motif"]])
    coef_tab = pd.concat(coefs, ignore_index=True)
    diag_tab = pd.concat(diags, ignore_index=True)
    lr_asym = 2 * (fits["GJR-GARCH(1,1)-t"].loglikelihood - fits["GARCH(1,1)-t"].loglikelihood)
    lr_t = 2 * (fits["GARCH(1,1)-t"].loglikelihood - fits["GARCH(1,1)-N"].loglikelihood)
    lr_tests = pd.DataFrame([
        {"comparaison": "GJR-GARCH(1,1)-t contre GARCH(1,1)-t", "H0": "γ = 0 (pas d'asymétrie)",
         "LR": lr_asym, "p_value": float(0.5 * stats.chi2.sf(max(lr_asym, 0), 1))},
        {"comparaison": "GARCH(1,1)-t contre GARCH(1,1)-N", "H0": "1/ν = 0 (loi normale)",
         "LR": lr_t, "p_value": float(0.5 * stats.chi2.sf(max(lr_t, 0), 1))},
    ])
    summary.to_csv(REPORTS / "d04_garch_train.csv", float_format="%.6f")
    coef_tab.to_csv(REPORTS / "d04_coefficients.csv", index=False, float_format="%.6f")
    diag_tab.to_csv(REPORTS / "d04_diagnostics.csv", index=False, float_format="%.6f")
    lr_tests.to_csv(REPORTS / "d04_tests_lr.csv", index=False, float_format="%.6f")
    log("GARCH estimés sur le train (paramètres, diagnostics, admissibilité)")

    # ---------------- 3. Sélection (validation) ----------------
    val_fc, _ = forecasts_for(val.index, r_all, y_est, specs, cfg, every)
    val_vol = volatility_table(val, val_fc, labels, proxy)
    val_var = var_tables(val, val_fc, labels, levels)
    ok = [s for s in specs if adm[s.name]["admissible"]] or specs
    chosen = min(ok, key=lambda s: val_vol.loc[s.name, "QLIKE"])
    val_vol.to_csv(REPORTS / "d04_validation_volatilite.csv", float_format="%.6f")
    val_var.to_csv(REPORTS / "d04_validation_var.csv", float_format="%.6f")
    log(f"Modèle GARCH retenu (QLIKE validation minimale parmi les admissibles) : {chosen.name}")

    # ---------------- 4. Gel ----------------
    selection = {
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "snapshot": snap_path.name, "snapshot_sha256": manifest["sha256"],
        "selected_on": "modèles admissibles sur le train (convergence, persistance < 1, plus d'effet ARCH "
                       "dans z_t), puis perte QLIKE minimale sur la validation 2023-2024",
        "admissibility_rules": cfg["admissibility"],
        "candidates": [s.to_dict() for s in specs],
        "admissible": {s.name: bool(adm[s.name]["admissible"]) for s in specs},
        "validation_qlike": {s.name: float(val_vol.loc[s.name, "QLIKE"]) for s in specs},
        "selected_model": chosen.to_dict(),
        "required_model": "GARCH(1,1)-N (toujours rapporté, exigé par le cahier)",
        "baselines": cfg["baselines"], "var_levels": levels, "refit_every": every,
    }
    (REPORTS / "d04_selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    log("Sélection gelée -> reports/d04_selection.json")

    # ---------------- 5. Test final (une passe) ----------------
    test_fc, refit_info = forecasts_for(test.index, r_all, y_est, specs, cfg, every)
    vol_tab = volatility_table(test, test_fc, labels, proxy)
    var_tab = var_tables(test, test_fc, labels, levels)
    es_tab = es_table(test, test_fc, labels, levels)
    sub_tab = subperiod_table(test, test_fc, labels, mcfg.subperiods)
    vol_tab.to_csv(REPORTS / "d04_test_volatilite.csv", float_format="%.6f")
    var_tab.to_csv(REPORTS / "d04_test_var.csv", float_format="%.6f")
    es_tab.to_csv(REPORTS / "d04_test_es.csv", float_format="%.6f")
    sub_tab.to_csv(REPORTS / "d04_test_subperiodes.csv", float_format="%.6f")
    log("Test évalué : volatilité, VaR, ES, sous-périodes")

    # Prévisions datées (entrée de D06/D07) + contrainte de position du modèle retenu
    pl = cfg["position_limit"]
    cap = float(cfg["risk_measures"]["capital_usd"])
    frames = []
    for split, fc, rr in (("validation", val_fc, val), ("test", test_fc, test)):
        cols = {"split": split, "r_%": rr}
        for m, f in fc.items():
            for c in ["sigma"] + [f"{x}{int(round(lv * 100))}" for lv in levels for x in ("VaR", "ES")]:
                cols[f"{m}_{c}"] = f[c]
        sel = fc[_slug(chosen.name)]
        cols["retenu_VaR99_USD"] = R.var_to_usd(sel["VaR99"], cap)
        cols["retenu_exposition_max"] = R.exposure_limit(sel["VaR99"], pl["max_var99_pct_capital"],
                                                         pl["max_exposure"])
        frames.append(pd.DataFrame(cols, index=rr.index))
    out = pd.concat(frames)
    out.to_csv(REPORTS / "d04_risk_forecasts.csv", float_format="%.6f", index_label="date")

    # ---------------- Figures et rapport ----------------
    plot_arch_effect(train, FIGURES / "d04_effet_arch.png")
    plot_garch_train(fits[chosen.name], chosen, train, FIGURES / "d04_garch_train.png")
    plot_volatility(test, test_fc, labels, chosen, FIGURES / "d04_volatilite_prevue.png")
    plot_var(test, test_fc, labels, chosen, FIGURES / "d04_var_backtest.png")
    plot_stability(sub_tab, labels, FIGURES / "d04_stabilite.png")

    ctx = dict(snap_path=snap_path, manifest=manifest, train=train, val=val, test=test, cfg=cfg, mcfg=mcfg,
               arima_spec=arima_spec, arch_tests=arch_tests, summary=summary, coef_tab=coef_tab,
               diag_tab=diag_tab, lr_tests=lr_tests, val_vol=val_vol, val_var=val_var, chosen=chosen,
               specs=specs, labels=labels, vol_tab=vol_tab, var_tab=var_tab, es_tab=es_tab, sub_tab=sub_tab,
               refit_info=refit_info, out=out, every=every, selection=selection, elapsed=time.time() - t0)
    write_report(ctx, REPORTS / "d04_risque.md")
    log(f"Rapport écrit : reports/d04_risque.md ({time.time() - t0:.0f} s)")
    return ctx


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_arch_effect(train: pd.Series, path: Path) -> Path:
    from statsmodels.tsa.stattools import acf
    plt = _plt()
    nl = 20
    band = 1.96 / np.sqrt(len(train))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(train.index, train, lw=0.5, color="#1f5f8b")
    axes[0].set_title("Rendement log r_t de XLV en % (train 2018-2022)\nles fortes variations arrivent en grappes")
    axes[0].set_ylabel("r_t (%)")
    lags = np.arange(1, nl + 1)
    axes[1].bar(lags - 0.2, acf(train, nlags=nl)[1:], width=0.4, color="#1f5f8b", label="r_t")
    axes[1].bar(lags + 0.2, acf(train ** 2, nlags=nl)[1:], width=0.4, color="#c0392b", label="r_t²")
    axes[1].axhspan(-band, band, color="grey", alpha=0.2, label="±1,96/√n")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_title("ACF : r_t² est bien plus autocorrélé que r_t\n(autocorrélations positives et durables = effet ARCH)")
    axes[1].set_xlabel("retard (séances)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_garch_train(res, spec: G.GarchSpec, train: pd.Series, path: Path) -> Path:
    from statsmodels.tsa.stattools import acf
    plt = _plt()
    z = G.std_residuals(res)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(train.index, np.abs(train), lw=0.4, color="grey", label="|r_t| réalisé")
    axes[0].plot(train.index, res.conditional_volatility, lw=1.2, color="#1f5f8b", label="σ_t estimé")
    axes[0].set_title(f"{spec.name} sur le train : volatilité conditionnelle (%)")
    axes[0].set_ylabel("% par séance")
    axes[0].legend(fontsize=8)
    if spec.dist == "t":
        nu = float(res.params["nu"])
        dist, args, lab = stats.t, (nu,), f"Student-t({nu:.1f})"
        zz = z / np.sqrt((nu - 2) / nu)
    else:
        dist, args, lab, zz = stats.norm, (), "N(0,1)", z
    stats.probplot(zz, dist=dist, sparams=args, plot=axes[1])
    axes[1].set_title(f"QQ-plot des résidus standardisés z_t contre {lab}")
    axes[1].set_xlabel(f"quantiles théoriques {lab}")
    axes[1].set_ylabel("quantiles observés")
    nl = 20
    band = 1.96 / np.sqrt(len(z))
    axes[2].bar(range(1, nl + 1), acf(z ** 2, nlags=nl)[1:], color="#1f5f8b", width=0.6)
    axes[2].axhspan(-band, band, color="grey", alpha=0.2)
    axes[2].axhline(0, color="black", lw=0.6)
    axes[2].set_title("ACF de z_t² : l'effet ARCH a disparu")
    axes[2].set_xlabel("retard (séances)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_volatility(test: pd.Series, fc: dict, labels: dict, chosen: G.GarchSpec, path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(test.index, np.abs(test), width=1.0, color="lightgrey", label="|r_t| réalisé (%)")
    keys = [_slug(chosen.name), "garch_n", "ewma", "const"]
    for k in dict.fromkeys(keys):
        ax.plot(test.index, fc[k]["sigma"], lw=1.5 if k == _slug(chosen.name) else 1.0, color=COLORS[k],
                label=f"σ prévu — {labels[k]}" + (" (retenu)" if k == _slug(chosen.name) else ""))
    ax.set_title("Test 2025-2026 : volatilité prévue la veille (σ_t, % par séance) et ampleur réalisée |r_t|")
    ax.set_ylabel("% par séance")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_var(test: pd.Series, fc: dict, labels: dict, chosen: G.GarchSpec, path: Path) -> Path:
    plt = _plt()
    k = _slug(chosen.name)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, lv in zip(axes, (95, 99)):
        ax.bar(test.index, test, width=1.0, color="lightgrey", label="r_t réalisé (%)")
        for m, ls in ((k, "-"), ("hs", "--")):
            ax.plot(test.index, -fc[m][f"VaR{lv}"], color=COLORS[m], lw=1.2, ls=ls,
                    label=f"−VaR{lv} {labels[m]}")
        h = R.hits(test, fc[k][f"VaR{lv}"]).astype(bool)
        ax.scatter(test.index[h], test[h], color="black", zorder=5, s=18,
                   label=f"exceptions {labels[k]} : {h.sum()} (attendues {len(test) * (100 - lv) / 100:.1f})")
        ax.set_title(f"VaR {lv} % à une séance, position longue : une exception = perte réalisée > VaR prévue")
        ax.set_ylabel("rendement et −VaR (%)")
        lo = min(float(test.min()), float(-fc[k][f"VaR{lv}"].max())) - 0.5
        ax.set_ylim(lo, float(test.max()) + 3.0)          # place libre en haut pour la légende
        ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_stability(sub: pd.DataFrame, labels: dict, path: Path) -> Path:
    plt = _plt()
    order = [v for k, v in labels.items() if k != "ewma"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    d = sub["ecart_QLIKE_vs_EWMA"].unstack("periode").loc[order]
    n_m, n_p = d.shape
    w = 0.8 / n_m
    inv = {v: k for k, v in labels.items()}
    for i, (m, row) in enumerate(d.iterrows()):
        axes[0].bar(np.arange(n_p) + i * w - 0.4 + w / 2, row.values, width=w, color=COLORS[inv[m]], label=m)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xticks(range(n_p))
    axes[0].set_xticklabels(d.columns)
    axes[0].set_title("Écart de QLIKE avec EWMA par sous-période\n< 0 : prévision de volatilité meilleure que EWMA")
    axes[0].set_ylabel("QLIKE(modèle) − QLIKE(EWMA)")
    axes[0].legend(fontsize=7)
    e = sub["exceptions_99"].unstack("periode")
    exp = sub["attendues_99"].unstack("periode").iloc[0]
    full = [v for v in labels.values()]
    e = e.loc[full]
    w = 0.8 / len(full)
    for i, (m, row) in enumerate(e.iterrows()):
        axes[1].bar(np.arange(n_p) + i * w - 0.4 + w / 2, row.values, width=w, color=COLORS[inv[m]], label=m)
    axes[1].hlines(exp.values, np.arange(n_p) - 0.42, np.arange(n_p) + 0.42, colors="black", lw=2,
                   label="attendues (1 % des séances)")
    axes[1].set_xticks(range(n_p))
    axes[1].set_xticklabels(e.columns)
    axes[1].set_ylabel("nombre d'exceptions")
    axes[1].set_ylim(0, float(e.values.max()) + 3)
    axes[1].set_title("Exceptions de la VaR 99 % par sous-période")
    axes[1].legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Rapport
# --------------------------------------------------------------------------- #
def _usd(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def write_report(c: dict, path: Path) -> Path:
    cfg, chosen, labels = c["cfg"], c["chosen"], c["labels"]
    vol, var, es, sub = c["vol_tab"], c["var_tab"], c["es_tab"], c["sub_tab"]
    summ, lr = c["summary"], c["lr_tests"]
    garch_names = [s.name for s in c["specs"]]
    cap = cfg["risk_measures"]["capital_usd"]
    k_sel = _slug(chosen.name)
    out = c["out"][c["out"]["split"] == "test"]

    L = []
    a = L.append
    a("# D04 — Module de risque : volatilité conditionnelle, VaR et ES (XLV)\n")
    a("> Fichier généré par `python -m src.risk_management`. Ne pas modifier à la main : relancer la commande.\n")

    a("## 1. Cadre et unités\n")
    a(f"- Snapshot figé : `data/snapshots/{c['snap_path'].name}` (SHA-256 `{c['manifest']['sha256'][:16]}…`, "
      "vérifié au chargement). C'est le même que pour D02 et D03 ; il est fixé dans `configs/risk.yaml`, donc un "
      "snapshot plus récent ne change pas les résultats.")
    a("- Variable : rendement log de XLV **en %**, r_t = 100 · ln(P_t / P_{t-1}), P = prix ajusté.")
    a("- Prévision : à la clôture de t-1, on prévoit la **distribution** de r_t : moyenne μ_t, volatilité σ_t "
      "(en % par séance ; × √252 pour l'annualiser), puis la VaR et l'ES.")
    a("- **VaR** (position longue) : nombre positif, en % de la valeur de la position, tel que "
      "P(perte > VaR) = 1 − niveau, avec perte = −r_t. **ES** : perte moyenne au-delà de la VaR (ES ≥ VaR).")
    a(f"- Conversion en dollars sur le capital du cahier ({_usd(cap)} USD investis) : "
      "VaR_USD = C · (1 − exp(−VaR/100)).")
    a("- Une **exception** se produit à la séance t si la perte réalisée −r_t dépasse la VaR prévue la veille. "
      "Les prévisions sont donc toujours **comparées à des observations futures**.")
    a(f"- Échantillons : train {c['train'].index[0]:%Y-%m-%d} → {c['train'].index[-1]:%Y-%m-%d} "
      f"({len(c['train'])} séances) · validation ({len(c['val'])}) · test {c['test'].index[0]:%Y-%m-%d} → "
      f"{c['test'].index[-1]:%Y-%m-%d} ({len(c['test'])}). L'année 2017 amorce seulement EWMA et la simulation "
      "historique.")
    a(f"- Walk-forward : fenêtre croissante depuis 2018, paramètres ré-estimés toutes les {c['every']} séances, "
      "variance conditionnelle mise à jour chaque jour. Le test `test_garch_walk_forward_no_lookahead` vérifie "
      "qu'aucune donnée future n'est utilisée.\n")

    a("## 2. Effet ARCH (train 2018-2022)\n")
    a(_md(c["arch_tests"], index=False))
    n_rej = int((c["arch_tests"]["p_value"] < 0.05).sum())
    a(f"\n{n_rej} tests sur {len(c['arch_tests'])} rejettent l'absence d'effet ARCH, y compris sur les résidus de "
      f"l'{c['arima_spec'].label} retenu en D03. La variance de r_t n'est pas constante : les fortes variations "
      "arrivent en grappes. Modéliser la variance conditionnelle est donc justifié.\n")
    a("![Effet ARCH](figures/d04_effet_arch.png)\n")

    a("## 3. Modèles GARCH estimés sur le train\n")
    a("Modèle : r_t = μ + ε_t, ε_t = σ_t z_t, σ²_t = ω + α ε²_{t-1} + γ ε²_{t-1}·1{ε_{t-1}<0} + β σ²_{t-1}. "
      "La moyenne est constante, car D03 a montré que la dynamique ARMA n'améliore pas la prévision.\n")
    show = summ[["mu_%", "omega", "alpha", "gamma", "beta", "nu", "persistance", "demi_vie_seances",
                 "vol_incond_annuelle_%", "logvraisemblance", "AIC", "BIC", "ARCH_LM_z_p", "LB_z2_p",
                 "admissible", "motif"]]
    a(_md(show, floatfmt="{:.4f}"))
    a("\n- persistance = α + β + γ/2 ; demi-vie = nombre de séances pour qu'un choc de volatilité diminue de moitié.")
    a("- ARCH_LM_z_p et LB_z2_p : tests sur les résidus standardisés z_t. Une p-value > 0,05 signifie que le "
      "modèle a absorbé l'effet ARCH.")
    a(f"- Règle d'admissibilité (`configs/risk.yaml`) : convergence, paramètres de variance ≥ 0, persistance "
      f"< {cfg['admissibility']['max_persistence']}, et plus d'effet ARCH significatif dans z_t.\n")
    a("Tests du rapport de vraisemblance (H0 sur le bord de l'espace des paramètres : p-value = ½ P(χ²₁ > LR)) :\n")
    a(_md(lr, index=False))
    a("\nCoefficients et erreurs-types : `reports/d04_coefficients.csv`. Diagnostics complets : "
      "`reports/d04_diagnostics.csv`.\n")
    a(_md(c["diag_tab"], index=False))
    a("\n![GARCH sur le train](figures/d04_garch_train.png)\n")

    a("## 4. Sélection sur la validation 2023-2024 et gel\n")
    a("Critère : perte **QLIKE** = ln σ̂² + r²/σ̂², moyennée sur la validation. Elle est minimale en espérance pour "
      "la vraie variance et reste fiable avec un proxy bruité comme r² (Patton, 2011). Les références (EWMA, "
      "normale constante, simulation historique) utilisent des conventions de marché fixées à l'avance : elles "
      "n'ont aucun paramètre à choisir.\n")
    a(_md(c["val_vol"][["QLIKE", "ecart_QLIKE_vs_EWMA", "DM_vs_EWMA_p", "MZ_b", "sigma_moy_%"]]))
    a(f"\nModèle GARCH retenu : **{chosen.name}**. La sélection est gelée dans `reports/d04_selection.json` avant "
      "toute prévision du test. GARCH(1,1)-N, exigé par le cahier, reste rapporté dans tous les tableaux.\n")

    a("## 5. Test final 2025-01 → 2026-10 (une seule passe)\n")
    a("### 5.1 Volatilité prévue contre volatilité réalisée\n")
    a(_md(vol[["QLIKE", "ecart_QLIKE_vs_EWMA", "DM_vs_EWMA_p", "ecart_QLIKE_vs_constante", "DM_vs_constante_p",
               "QLIKE_Parkinson", "MZ_a", "MZ_b", "MZ_R2", "MZ_wald_p", "sigma_moy_%"]]))
    a("\n- QLIKE : plus faible = meilleure prévision de la variance. Écarts < 0 : meilleur que la référence ; "
      "DM_p : test de Diebold-Mariano sur la différence de pertes.")
    a("- QLIKE_Parkinson : même perte avec la variance de Parkinson (amplitude haut-bas de la séance) comme proxy "
      "réalisé ; elle sert à vérifier le classement.")
    a("- Mincer-Zarnowitz : r²_t = a + b σ̂²_t. Une prévision sans biais donne a ≈ 0 et b ≈ 1 (Wald p > 0,05). "
      "Pour la normale constante, σ̂ ne varie presque pas : a et b ne sont alors pas interprétables.\n")
    a("![Volatilité prévue](figures/d04_volatilite_prevue.png)\n")

    a("### 5.2 Backtest de la VaR (tableaux de vérification)\n")
    for lv in cfg["risk_measures"]["var_levels"]:
        t = var[var["niveau"] == lv].drop(columns="niveau")
        a(f"**VaR {int(lv * 100)} %** — {int(t['n'].iloc[0])} séances, {t['attendues'].iloc[0]:.1f} exceptions "
          "attendues.\n")
        a(_md(t, floatfmt="{:.3f}"))
        a("")
    a("- Kupiec : H0 = le taux d'exceptions est le bon. Indépendance (Christoffersen) : H0 = les exceptions ne "
      "se suivent pas. Couverture conditionnelle : les deux à la fois.")
    a("- Zone de Bâle généralisée : vert si le nombre d'exceptions est plausible au seuil de 95 %, jaune jusqu'à "
      "99,99 %, rouge au-delà.")
    a("- tick_loss : perte quantile, pour comparer la précision des VaR (plus faible = meilleure).\n")
    a("![Backtest VaR](figures/d04_var_backtest.png)\n")

    a("### 5.3 Backtest de l'Expected Shortfall\n")
    a(_md(es, floatfmt="{:.3f}"))
    a("\n- ratio_perte_ES : perte moyenne les jours d'exception / ES moyenne prévue ces jours-là (≈ 1 si l'ES est "
      "bien calibrée, > 1 si elle sous-estime les pertes extrêmes).")
    a("- MF_p : test unilatéral de McNeil-Frey (H0 : l'ES ne sous-estime pas les pertes). Peu puissant à 99 % : "
      "il y a très peu d'exceptions.")
    a("- FZ0 : perte conjointe (VaR, ES) de Fissler-Ziegel ; plus faible = meilleure paire.\n")

    a("### 5.4 Stabilité sur les sous-périodes (variante B05)\n")
    a(_md(sub.reset_index(), index=False, floatfmt="{:.3f}"))
    a("\n![Stabilité](figures/d04_stabilite.png)\n")

    a("### 5.5 Contrainte de position proposée pour D06\n")
    pl = cfg["position_limit"]
    expo = out["retenu_exposition_max"]
    a(f"Règle : on investit au plus la fraction w du capital telle que la VaR 99 % de la position, prévue par "
      f"{chosen.name}, reste ≤ {pl['max_var99_pct_capital']} % du capital : "
      f"w = min({pl['max_exposure']}, {pl['max_var99_pct_capital']} % / (1 − exp(−VaR99/100))). "
      f"Sur le test, w vaut en moyenne {expo.mean():.2f} (minimum {expo.min():.2f}) et est inférieur à 1 pendant "
      f"{int((expo < 1).sum())} séances sur {len(expo)}. VaR 99 % moyenne d'une position de {_usd(cap)} USD : "
      f"{_usd(out['retenu_VaR99_USD'].mean())} USD (maximum {_usd(out['retenu_VaR99_USD'].max())} USD). "
      "Cette règle n'est qu'une proposition : la stratégie sera définie et testée en D06.")
    a("")

    # ---------------- Conclusions générées à partir des chiffres ----------------
    a("## 6. Conclusions\n")
    best_ql = vol["QLIKE"].idxmin()
    garch_better_ewma = [g for g in garch_names if vol.loc[g, "ecart_QLIKE_vs_EWMA"] < 0]
    garch_signif_ewma = [g for g in garch_names if vol.loc[g, "ecart_QLIKE_vs_EWMA"] < 0
                         and vol.loc[g, "DM_vs_EWMA_p"] < 0.05]
    garch_signif_const = [g for g in garch_names if vol.loc[g, "ecart_QLIKE_vs_constante"] < 0
                          and vol.loc[g, "DM_vs_constante_p"] < 0.05]
    a(f"1. **L'effet ARCH est réel** sur le train ({n_rej}/{len(c['arch_tests'])} tests significatifs) et les GARCH "
      "l'absorbent : voir les p-values ARCH-LM des résidus standardisés (section 3).")
    a(f"2. **Prévision de la volatilité (test).** Meilleure QLIKE : {best_ql}. "
      + (f"Modèles qui battent significativement une volatilité constante (DM p < 0,05) : "
         f"{', '.join(garch_signif_const)}. "
         if garch_signif_const else "Aucun GARCH ne bat significativement la volatilité constante. ")
      + (f"Contre EWMA, QLIKE plus faible pour {', '.join(garch_better_ewma)}"
         + (f", significativement pour {', '.join(garch_signif_ewma)}." if garch_signif_ewma
            else ", mais l'écart n'est significatif pour aucun.")
         if garch_better_ewma else "Aucun GARCH ne fait mieux qu'EWMA en QLIKE.")
      + (f" Attention : sur la validation, la volatilité constante était la pire (QLIKE "
         f"{c['val_vol'].loc[BASELINES['const'], 'QLIKE']:.3f}). Elle inclut la crise de 2020, donc elle "
         f"surestime le risque en période calme (validation : écart-type réalisé {c['val'].std():.2f} %) et "
         f"devient acceptable quand le marché s'agite (test : {c['test'].std():.2f} %). Son bon score sur le "
         "test tient à la période, pas à une vraie capacité de prévision."
         if c["val_vol"]["QLIKE"].idxmax() == BASELINES["const"] else ""))
    vt = var[var["niveau"] == 0.99]
    green99 = [m for m in vt.index if vt.loc[m, "Kupiec_p"] > 0.05 and vt.loc[m, "independance_p"] > 0.05]
    v95 = var[var["niveau"] == 0.95]
    green95 = [m for m in v95.index if v95.loc[m, "Kupiec_p"] > 0.05 and v95.loc[m, "independance_p"] > 0.05]
    a(f"3. **VaR.** À 95 %, couverture et indépendance non rejetées pour : {', '.join(green95) or 'aucun modèle'}. "
      f"À 99 % : {', '.join(green99) or 'aucun modèle'}. "
      f"Exceptions à 99 % du modèle retenu : {int(vt.loc[chosen.name, 'exceptions'])} pour "
      f"{vt.loc[chosen.name, 'attendues']:.1f} attendues ; GARCH(1,1)-N : "
      f"{int(vt.loc['GARCH(1,1)-N', 'exceptions'])}.")
    hs99 = vt.loc[BASELINES["hs"]]
    not_green = [m for m in vt.index if vt.loc[m, "zone_Bale"] != "vert"]
    if hs99["Kupiec_p"] < 0.05:
        a(f"   La simulation historique est rejetée à 99 % ({int(hs99['exceptions'])} exceptions, Kupiec "
          f"p = {hs99['Kupiec_p']:.3f}, zone {hs99['zone_Bale']}) : sa fenêtre de 250 séances réagit trop "
          "lentement quand la volatilité monte d'un coup. Modèles hors zone verte à 99 % : "
          f"{', '.join(not_green) or 'aucun'}.")
    gjr, gt = "GJR-GARCH(1,1)-t", "GARCH(1,1)-t"
    gjr_val = c["val_vol"].loc[gjr, "QLIKE"] - c["val_vol"].loc[gt, "QLIKE"]
    gjr_test = vol.loc[gjr, "QLIKE"] - vol.loc[gt, "QLIKE"]
    a(f"4. **Asymétrie et queues épaisses.** Sur le train, la loi t (LR p = {lr['p_value'].iloc[1]:.2g}) et "
      f"l'asymétrie GJR (LR p = {lr['p_value'].iloc[0]:.2g}) sont toutes deux significatives. "
      + ("Pourtant, hors échantillon, le GJR prévoit moins bien la volatilité que le GARCH-t symétrique "
         f"(écart de QLIKE {gjr_val:+.3f} en validation, {gjr_test:+.3f} en test) : un terme significatif sur "
         "le train n'est pas forcément utile pour prévoir. "
         if gjr_val > 0 and gjr_test > 0 else
         f"Hors échantillon, écart de QLIKE GJR − GARCH-t : {gjr_val:+.3f} en validation, {gjr_test:+.3f} en test. ")
      + "Les queues épaisses comptent surtout pour la VaR 99 % (tableau 5.2).")
    d = sub["ecart_QLIKE_vs_EWMA"].unstack("periode")
    n_per = d.shape[1]
    wins_ewma = int((d.loc[chosen.name] < 0).sum())
    dc = (sub["QLIKE"].unstack("periode").loc[chosen.name]
          - sub["QLIKE"].unstack("periode").loc[BASELINES["const"]])
    wins_const = int((dc < 0).sum())
    a(f"5. **Réponse à la question B05 (parcimonie), côté risque.** La structure GARCH est clairement présente "
      f"dans les données (sections 2 et 3), mais son avantage hors échantillon est faible et instable : "
      f"{chosen.name} bat EWMA dans {wins_ewma} sous-période(s) sur {n_per} et la volatilité constante dans "
      f"{wins_const} sur {n_per} (tableau 5.4). Son plus grand gain sur EWMA est en {d.loc[chosen.name].idxmin()} "
      f"(écart de QLIKE {d.loc[chosen.name].min():.3f}). En D03, la complexité n'apportait rien pour la "
      "**moyenne** ; pour le **risque**, le GARCH apporte un gain modeste et non significatif, "
      "concentré sur les périodes où la volatilité change vite. À 99 %, la loi de Student donne une VaR plus "
      f"prudente que la loi normale ({int(vt.loc['GARCH(1,1)-t', 'exceptions'])} exceptions contre "
      f"{int(vt.loc['GARCH(1,1)-N', 'exceptions'])} pour {vt.loc['GARCH(1,1)-N', 'attendues']:.1f} attendues)."
      if wins_ewma < n_per else
      f"5. **Réponse à la question B05 (parcimonie), côté risque.** {chosen.name} bat EWMA dans toutes les "
      f"sous-périodes ({n_per}/{n_per}) et la volatilité constante dans {wins_const}/{n_per}.")
    a("6. Les prévisions datées de σ, VaR et ES (validation et test) sont dans `reports/d04_risk_forecasts.csv`. "
      "Elles serviront à la règle de taille de position de D06 et à l'espace Risk Management de l'application (D07).\n")

    a("## 7. Limites\n")
    a("- Une seule série (XLV) et une prévision à 1 séance ; pas de VaR multi-jours ni de portefeuille.")
    a("- r² est un proxy très bruité de la variance réalisée ; le proxy de Parkinson ignore l'écart de la nuit. "
      "Sans données intrajournalières, la précision des comparaisons de volatilité reste limitée.")
    a("- À 99 %, 459 séances donnent environ 4,6 exceptions attendues : les tests de la VaR 99 % et de l'ES ont "
      "peu de puissance.")
    a("- La moyenne μ est constante ; les seuils (admissibilité, λ = 0,94, fenêtre de 250 séances, budget de VaR) "
      "sont des conventions fixées à l'avance dans `configs/risk.yaml`.\n")
    a(f"_Logiciels : arch, statsmodels, scipy. Durée d'exécution : {c['elapsed']:.0f} s._\n")
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Module de risque — livrable D04")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)
    run(verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
