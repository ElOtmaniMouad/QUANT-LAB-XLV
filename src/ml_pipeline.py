"""Pipeline du livrable D05 — Machine Learning et Deep Learning (XLV).

Chaîne :  snapshot figé -> variables (clôture t-1) -> grille d'hyperparamètres évaluée sur la
VALIDATION -> choix GELÉ dans reports/d05_selection.json -> prévisions du TEST (walk-forward)
-> métriques, tests, stabilité T1–T4, parcimonie -> rapport reports/d05_ml_dl.md.

Lancement :  ``python -m src.ml_pipeline``

Les modèles sont comparés aux références et aux modèles de D03 sur la MÊME cible (rendement log à
une séance), le MÊME instant d'information, les MÊMES fenêtres et les MÊMES données de test.
"""
from __future__ import annotations

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

from src import evaluation as E
from src.data import ROOT, load_config, load_snapshot, read_manifest
from src.econometrics import _between, _md
from src.features import build_dataset, load_model_config, modelling_sample
from src.ml_features import build_features, make_xy
from src.models import baseline as B
from src.models import machine_learning as M
from src.models.arima import ArimaSpec, walk_forward as arima_walk_forward

DEFAULT_ML_CONFIG = ROOT / "configs" / "ml.yaml"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ML = list(M.FAMILIES)
COLORS = {"zero": "#7f7f7f", "hist_mean": "#a6a6a6", "ma60": "#c9c9c9", "ARIMA": "#1f6f8b",
          "SARIMAX": "#5aa5c0", "Ridge": "#e39e21", "RandomForest": "#4c9a4c",
          "GradientBoosting": "#b5483f", "MLP": "#7b52a6"}


def load_ml_config(path: Path | str = DEFAULT_ML_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Statistiques
# --------------------------------------------------------------------------- #
def holm(p: pd.Series) -> pd.Series:
    """Correction de Holm (contrôle du risque global) ; les NaN sont ignorés."""
    q = p.dropna().sort_values()
    m = len(q)
    adj, running = {}, 0.0
    for i, (k, v) in enumerate(q.items()):
        running = max(running, min(1.0, (m - i) * v))
        adj[k] = running
    return pd.Series(adj).reindex(p.index)


def information_coefficient(actual: pd.Series, pred: pd.Series) -> tuple[float, float]:
    """Corrélation de Pearson prévision/réalisation et sa p-value (NaN si prévision constante)."""
    df = pd.concat({"a": actual, "p": pred}, axis=1).dropna()
    if df["p"].std() < 1e-12:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(df["p"], df["a"])
    return float(r), float(p)


def mincer_zarnowitz_slope(actual: pd.Series, pred: pd.Series) -> float:
    """Pente de la régression r_t = a + b·r̂_t : 1 = prévision calibrée, 0 = sans information."""
    df = pd.concat({"a": actual, "p": pred}, axis=1).dropna()
    if df["p"].std() < 1e-12:
        return float("nan")
    return float(np.polyfit(df["p"], df["a"], 1)[0])


def pesaran_timmermann(actual: pd.Series, pred: pd.Series) -> tuple[float, float]:
    """Test de Pesaran-Timmermann (1992) : l'exactitude directionnelle diffère-t-elle du hasard ?

    Compare la fréquence de bonnes directions à celle qu'on obtiendrait avec une prévision
    INDÉPENDANTE de la réalité, mais de même fréquence de hausses prévues. Même convention de
    signe que ``evaluate`` (r ≥ 0 = « hausse »). Renvoie (statistique z, p-value bilatérale) ;
    NaN si la prévision est de signe constant (le test n'a alors pas de sens).
    """
    df = pd.concat({"a": actual, "p": pred}, axis=1).dropna()
    n = len(df)
    ya, xp = (df["a"] >= 0).to_numpy(), (df["p"] >= 0).to_numpy()
    py, px = ya.mean(), xp.mean()
    if n < 10 or px in (0.0, 1.0) or py in (0.0, 1.0):
        return float("nan"), float("nan")
    hit = float((ya == xp).mean())
    p_star = py * px + (1 - py) * (1 - px)
    var_hit = p_star * (1 - p_star) / n
    var_star = (((2 * py - 1) ** 2 * px * (1 - px) + (2 * px - 1) ** 2 * py * (1 - py)) / n
                + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    if var_hit - var_star <= 0:
        return float("nan"), float("nan")
    z = (hit - p_star) / np.sqrt(var_hit - var_star)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def full_metrics(data: pd.DataFrame, preds: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Métriques de D03 (``evaluate``) + DM contre la moyenne historique, Holm, IC, pente MZ."""
    tab = E.evaluate(data, preds)
    a = data["y"]
    dm_h, ic, ic_p, mz = {}, {}, {}, {}
    for c in preds.columns:
        dm_h[c] = E.diebold_mariano(a, preds[c], preds["hist_mean"]) if c != "hist_mean" else (np.nan, np.nan)
        ic[c], ic_p[c] = information_coefficient(a, preds[c])
        mz[c] = mincer_zarnowitz_slope(a, preds[c])
    tab["DM_vs_moyenne"] = pd.Series({c: v[0] for c, v in dm_h.items()})
    tab["DM_moyenne_p"] = pd.Series({c: v[1] for c, v in dm_h.items()})
    tab["p_Holm"] = holm(tab["DM_p"])
    tab["IC"] = pd.Series(ic)
    tab["IC_p"] = pd.Series(ic_p)
    tab["pente_MZ"] = pd.Series(mz)
    tab["DA_PT_p"] = pd.Series({c: pesaran_timmermann(a, preds[c])[1] for c in preds.columns})
    tab["DA_PT_Holm"] = holm(tab["DA_PT_p"])
    tab["signif_5%"] = (tab["p_Holm"] < alpha) & (tab["DM_vs_zero"] < 0)
    return tab


def _edge_axes(spec: M.ModelSpec, family_cfg: dict) -> list[str]:
    """Hyperparamètres (grille d'au moins 3 valeurs) dont la valeur retenue est en BORD de grille."""
    return [k for k in spec.varied if len(family_cfg[k]) >= 3 and spec.p[k] in (min(family_cfg[k]), max(family_cfg[k]))]


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(verbose: bool = True, config_path: Path | str = DEFAULT_ML_CONFIG) -> dict:
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    data_cfg, mcfg, cfg = load_config(), load_model_config(), load_ml_config(config_path)
    seed, every, alpha = int(cfg["seed"]), int(cfg["walk_forward"]["refit_every"]), float(cfg["alpha"])
    snap = data_cfg.snapshot_dir / cfg["snapshot"]
    df = load_snapshot(snap)                                    # vérifie le SHA-256 du snapshot
    manifest = read_manifest(snap)
    ds = modelling_sample(build_dataset(df, data_cfg, mcfg))
    X = build_features(df, cfg, mcfg["target"]["ticker"], mcfg["exog"]["ticker"])
    Xs, ys = make_xy(ds, X)
    if len(Xs) != len(ds):
        raise RuntimeError("Des séances du jeu de modélisation n'ont pas toutes leurs variables.")
    sp = data_cfg.splits
    train, val, test = _between(ds, sp["train"]), _between(ds, sp["validation"]), _between(ds, sp["test"])
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    log(f"Snapshot {snap.name} · {Xs.shape[1]} variables · train {len(train)} · validation {len(val)} · test {len(test)}")

    # ---------------- 1. Références communes (baselines + modèles de D03) ----------------
    y = ds["y"]
    windows = [int(n[2:]) for n in cfg["references"]["baselines"] if n.startswith("ma")]
    base = pd.concat([B.zero(y), B.hist_mean(y)] + [B.moving_avg(y, w) for w in windows], axis=1)
    d03_val, d03_test, d03_specs = {}, {}, {}
    for name, d in cfg["references"]["d03_models"].items():
        spec = ArimaSpec(name, order=tuple(d["order"]), exog=bool(d["exog"]))
        d03_specs[name] = spec
        d03_val[name] = arima_walk_forward(spec, ds, val.index[0], val.index[-1], every).predictions
        d03_test[name] = arima_walk_forward(spec, ds, test.index[0], test.index[-1], every).predictions
    log("Références recalculées (baselines, ARIMA, SARIMAX de D03)")

    # ---------------- 2. Hyperparamètres : VALIDATION uniquement ----------------
    grids, tables, ml_val, selected = {}, [], {}, {}
    for fam in ML:
        specs = M.expand_grid(fam, cfg["models"][fam])
        grids[fam] = specs
        table, preds = M.validate_grid(specs, Xs, ys, val.index[0], val.index[-1], every, seed)
        best = M.pick_best(table, specs)
        table["retenu"] = table["candidat"] == best.label
        table["bord_de_grille"] = [", ".join(_edge_axes(s, cfg["models"][fam])) if s.label == best.label else ""
                                   for s in specs]
        tables.append(table)
        selected[fam] = best
        ml_val[fam] = preds[best.label]
        log(f"  {fam:17s} retenu : {best.label}  (RMSE validation {table.loc[table['retenu'], 'RMSE_val_%'].iloc[0]:.4f} %)")
    grid_table = pd.concat(tables, ignore_index=True)
    grid_table.to_csv(REPORTS / "d05_validation_grille.csv", index=False, float_format="%.6f")

    val_preds = pd.concat([base.loc[val.index], pd.DataFrame(d03_val), pd.DataFrame(ml_val)], axis=1).loc[val.index]
    val_metrics = full_metrics(val, val_preds, alpha)
    val_metrics.to_csv(REPORTS / "d05_validation_metrics.csv", float_format="%.6f")

    # ---------------- 3. Gel de la sélection AVANT la première prévision du test ----------------
    frozen = {
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "snapshot": snap.name,
        "snapshot_sha256": manifest["sha256"],
        "selected_on": "RMSE walk-forward minimale sur la validation 2023-2024, par famille de modèles ; "
                       "aucune donnée du test n'a été utilisée avant cette écriture",
        "seed": seed,
        "refit_every": every,
        "features": list(Xs.columns),
        "models": {fam: {**M.spec_to_dict(spec),
                         "RMSE_val_pct": float(grid_table[(grid_table["famille"] == fam) & grid_table["retenu"]]["RMSE_val_%"].iloc[0]),
                         "bord_de_grille": [a for a in _edge_axes(spec, cfg["models"][fam])]}
                   for fam, spec in selected.items()},
    }
    (REPORTS / "d05_selection.json").write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log("Sélection gelée -> reports/d05_selection.json")

    # ---------------- 4. Test final ----------------
    ml_test, cost = {}, {}
    for fam, spec in selected.items():
        res = M.walk_forward(lambda sp=spec: M.make_estimator(sp, seed), Xs, ys, test.index[0], test.index[-1], every)
        ml_test[fam] = res.predictions
        cost[fam] = {"candidat": spec.label, "parametres": M.n_parameters(res.last_estimator),
                     "secondes_test": res.fit_seconds, "refits": res.n_refits,
                     "ms_par_refit": 1000 * res.fit_seconds / res.n_refits}
    for name, p in d03_test.items():
        cost[name] = {"candidat": d03_specs[name].label, "parametres": _arima_params(d03_specs[name]),
                      "secondes_test": float("nan"), "refits": int(np.ceil(len(test) / every)), "ms_par_refit": float("nan")}
    test_preds = pd.concat([base.loc[test.index], pd.DataFrame(d03_test), pd.DataFrame(ml_test)], axis=1).loc[test.index]
    test_metrics = full_metrics(test, test_preds, alpha)
    test_metrics.to_csv(REPORTS / "d05_test_metrics.csv", float_format="%.6f")
    log("Test évalué : " + ", ".join(f"{m} {test_metrics.loc[m, 'RMSE_%']:.4f}" for m in ["zero"] + ML))

    sub = E.evaluate_by_period(test, test_preds, mcfg.subperiods)
    sub.to_csv(REPORTS / "d05_test_subperiods.csv", float_format="%.6f")
    ratio = sub["ratio_RMSE_zero"].unstack("periode")[list(mcfg.subperiods)]
    cost_table = pd.DataFrame(cost).T
    cost_table.index.name = "modele"
    cost_table.to_csv(REPORTS / "d05_parcimonie.csv", float_format="%.4f")

    # Importance des variables : modèles ré-estimés sur train + validation (jamais sur le test)
    imp = {}
    upto = Xs.index <= val.index[-1]
    for fam in ["Ridge", "RandomForest", "GradientBoosting"]:
        est = M.make_estimator(selected[fam], seed)
        est.fit(Xs[upto].to_numpy(float), ys[upto].to_numpy(float) * M.SCALE)
        imp[fam] = M.feature_importance(est, list(Xs.columns))
    importance = pd.DataFrame(imp)
    importance.index.name = "variable"
    importance.to_csv(REPORTS / "d05_importances.csv", float_format="%.6f")

    # Prévisions datées (entrée de D06 / D07)
    out = pd.concat([pd.concat([val[["y"]], val_preds], axis=1).assign(split="validation"),
                     pd.concat([test[["y"]], test_preds], axis=1).assign(split="test")])
    out.index.name = "date"
    out.to_csv(REPORTS / "d05_predictions.csv", float_format="%.8f")

    c = {"cfg": cfg, "manifest": manifest, "frozen": frozen, "features": list(Xs.columns), "selected": selected,
         "grid_table": grid_table, "val_metrics": val_metrics, "test_metrics": test_metrics, "ratio": ratio,
         "sub": sub, "cost": cost_table, "importance": importance, "val_preds": val_preds, "test_preds": test_preds,
         "test": test, "val": val, "train": train, "subperiods": mcfg.subperiods, "elapsed": time.time() - t0,
         "n_features": Xs.shape[1], "sp": sp, "every": every, "alpha": alpha}
    figs = {
        "grille": plot_validation_grid(c, FIGURES / "d05_validation_grille.png"),
        "test": plot_test_ratio(c, FIGURES / "d05_test_rmse.png"),
        "calibration": plot_calibration(c, FIGURES / "d05_prevision_vs_reel.png"),
        "stabilite": plot_stability(c, FIGURES / "d05_stabilite.png"),
        "importances": plot_importances(c, FIGURES / "d05_importances.png"),
    }
    c["figures"] = figs
    c["elapsed"] = time.time() - t0
    write_report(c, REPORTS / "d05_ml_dl.md")
    log(f"Rapport écrit : reports/d05_ml_dl.md ({c['elapsed']:.0f} s)")
    return c


def _arima_params(spec: ArimaSpec) -> int:
    p, _, q = spec.order
    return int(p + q + 1 + 1 + int(spec.exog))      # AR + MA + constante + variance + coefficient SPY


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "font.size": 10})
    return plt


def plot_validation_grid(c: dict, path: Path) -> Path:
    plt = _plt()
    g, vm = c["grid_table"], c["val_metrics"]
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(g) + 1.8))
    ypos = np.arange(len(g))[::-1]
    for (_, r), yv in zip(g.iterrows(), ypos):
        ax.barh(yv, r["RMSE_val_%"], color=COLORS[r["famille"]], alpha=1.0 if r["retenu"] else 0.45,
                edgecolor="black" if r["retenu"] else "none", height=0.7)
        ax.text(r["RMSE_val_%"] + 0.0005, yv, f"{r['RMSE_val_%']:.4f}" + ("  ◄ retenu" if r["retenu"] else ""),
                va="center", fontsize=8)
    ax.axvline(vm.loc["zero", "RMSE_%"], color="black", ls="--", lw=1, label="marche aléatoire (r̂ = 0)")
    ax.axvline(vm.loc["hist_mean", "RMSE_%"], color="gray", ls=":", lw=1.2, label="moyenne historique")
    lo = min(g["RMSE_val_%"].min(), vm.loc["zero", "RMSE_%"]) - 0.01
    ax.set_xlim(lo, g["RMSE_val_%"].max() + 0.02)
    ax.set_yticks(ypos)
    ax.set_yticklabels(g["candidat"], fontsize=8)
    ax.set_xlabel("RMSE de validation 2023-2024 (% de rendement journalier) — plus petit = meilleur")
    ax.set_title("D05 — Hyperparamètres évalués sur la validation (walk-forward, 1 séance)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_test_ratio(c: dict, path: Path) -> Path:
    plt = _plt()
    tm = c["test_metrics"].drop(index="zero")
    order = tm["ratio_RMSE_zero"].sort_values().index
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, m in enumerate(order):
        r = tm.loc[m, "ratio_RMSE_zero"]
        sig = bool(tm.loc[m, "signif_5%"])
        ax.scatter(r, i, s=90, color=COLORS[m], edgecolor="black", zorder=3, marker="o" if sig else "D")
        ax.annotate(f"{r:.4f}", (r, i), xytext=(9, 0), textcoords="offset points", va="center", fontsize=8)
    ax.axvline(1.0, color="black", ls="--", lw=1)
    ax.margins(x=0.12)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("RMSE du modèle / RMSE de la marche aléatoire — < 1 : le modèle fait mieux")
    ax.set_title("D05 — Test 2025-2026 : erreur de prévision relative à la marche aléatoire\n"
                 "◆ = écart non significatif après correction de Holm (● = significatif)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_calibration(c: dict, path: Path) -> Path:
    plt = _plt()
    tp, y, tm = c["test_preds"], c["test"]["y"] * 100, c["test_metrics"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, m in zip(axes, ML):
        p = tp[m] * 100
        ax.scatter(p, y, s=7, alpha=0.45, color=COLORS[m])
        lim = float(np.ceil(max(abs(p).max(), 0.5) * 10) / 10)
        xs = np.array([-lim, lim])
        ax.plot(xs, xs, color="black", lw=1, ls="--", label="prévision parfaite (pente 1)")
        b = tm.loc[m, "pente_MZ"]
        if np.isfinite(b):
            coef = np.polyfit(p, y, 1)
            ax.plot(xs, np.polyval(coef, xs), color="red", lw=1.5, label=f"régression (pente {b:.2f})")
        ax.set_xlim(-lim, lim)
        ax.set_title(f"{m}\nIC = {tm.loc[m, 'IC']:.3f}, RMSE/zéro = {tm.loc[m, 'ratio_RMSE_zero']:.3f}", fontsize=9)
        ax.set_xlabel("Rendement prévu (%)")
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    axes[0].set_ylabel("Rendement réalisé (%)")
    fig.suptitle("D05 — Prévisions confrontées aux réalisations (test, 1 séance) : un nuage plat = aucune information", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_stability(c: dict, path: Path) -> Path:
    plt = _plt()
    ratio = c["ratio"].drop(index="zero")
    ratio = ratio.assign(Test=c["test_metrics"]["ratio_RMSE_zero"].drop(index="zero"))
    models = [m for m in ["hist_mean", "ARIMA", "SARIMAX"] + ML if m in ratio.index]
    cols = list(ratio.columns)
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, m in enumerate(models):
        ax.bar(np.arange(len(cols)) + i * width, ratio.loc[m, cols] - 1.0, width, bottom=1.0, color=COLORS[m], label=m)
    ax.axhline(1.0, color="black", lw=1, ls="--")
    ax.set_xticks(np.arange(len(cols)) + 0.4 - width / 2)
    ax.set_xticklabels([f"{k}\n{c['subperiods'][k][0]:%m/%Y}–{c['subperiods'][k][1]:%m/%Y}" if k in c["subperiods"] else "Test complet"
                        for k in cols])
    ax.set_ylabel("RMSE / RMSE marche aléatoire (< 1 : meilleur que zéro)")
    ax.set_title("D05 — Stabilité de l'avantage prévisionnel dans les sous-périodes du test (variante B05)")
    ax.legend(ncol=len(models), frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_importances(c: dict, path: Path) -> Path:
    plt = _plt()
    imp = c["importance"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    titles = {"Ridge": "Ridge : coefficient standardisé\n(% de rendement par écart-type de la variable)",
              "RandomForest": "Forêt aléatoire : importance par impureté\n(somme = 1)",
              "GradientBoosting": "Gradient boosting : importance par impureté\n(somme = 1)"}
    order = imp["RandomForest"].sort_values().index
    for ax, fam in zip(axes, titles):
        ax.barh(order, imp.loc[order, fam], color=COLORS[fam])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(titles[fam], fontsize=9)
    fig.suptitle("D05 — Variables utilisées par les modèles (ré-estimés sur train + validation)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Rapport
# --------------------------------------------------------------------------- #
FEATURE_DOC = [
    ("xlv_r_lag1 … lag5", "rendements log de XLV des 5 séances précédentes (r_{t-1} … r_{t-5})"),
    ("xlv_moy5 / 20 / 60", "moyenne des rendements de XLV sur 5, 20 et 60 séances (jusqu'à t-1)"),
    ("xlv_vol5 / 20 / 60", "écart-type des rendements de XLV sur 5, 20 et 60 séances (jusqu'à t-1)"),
    ("spy_r_lag1 … lag3", "rendements log de SPY de t-1, t-2, t-3"),
    ("spy_vol20", "écart-type des rendements de SPY sur 20 séances (jusqu'à t-1)"),
    ("force_rel5 / 20", "somme de (r^XLV − r^SPY) sur 5 et 20 séances (jusqu'à t-1)"),
    ("tendance_ma50 / 200", "ln(prix ajusté de t-1 / moyenne mobile 50 ou 200 séances)"),
    ("volume_rel20", "ln(volume de t-1 / volume moyen sur 20 séances)"),
]


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def _names(items: list) -> str:
    return ", ".join(items) if items else "aucun"


def conclusions(c: dict) -> list[str]:
    tm, vm, ratio, cost, alpha = c["test_metrics"], c["val_metrics"], c["ratio"], c["cost"], c["alpha"]
    simple = ["zero", "hist_mean", "ma60"]
    best_simple = tm.loc[simple, "RMSE_%"].idxmin()
    best_ml = tm.loc[ML, "RMSE_%"].idxmin()
    beat0 = [m for m in ML if tm.loc[m, "ratio_RMSE_zero"] < 1]
    sig = [m for m in ML if bool(tm.loc[m, "signif_5%"])]
    d03_best = tm.loc[["ARIMA", "SARIMAX"], "RMSE_%"].idxmin()
    out = []

    if sig:
        out.append(f"**Prévision du rendement (test), au sens de l'erreur quadratique.** Après correction de Holm, {_names(sig)} bat(tent) significativement "
                   f"la marche aléatoire au seuil de {_pct(alpha)}. Meilleur modèle ML/DL : **{best_ml}** "
                   f"(RMSE {tm.loc[best_ml, 'RMSE_%']:.4f} %, soit {tm.loc[best_ml, 'ratio_RMSE_zero']:.3f} fois celui de la marche aléatoire).")
    else:
        out.append(f"**Prévision du rendement (test), au sens de l'erreur quadratique.** Aucun des 4 modèles ML/DL ne bat significativement la marche aléatoire "
                   f"(p ajustée de Holm ≥ {_pct(alpha)} pour tous). Meilleur modèle ML/DL : **{best_ml}** "
                   f"(RMSE {tm.loc[best_ml, 'RMSE_%']:.4f} %, soit {tm.loc[best_ml, 'ratio_RMSE_zero']:.3f} fois celui de la marche aléatoire ; "
                   f"p ajustée {tm.loc[best_ml, 'p_Holm']:.2f}). "
                   f"{len(beat0)} modèle(s) sur 4 ont un RMSE inférieur à 1 fois celui de la marche aléatoire ({_names(beat0)}), mais l'écart n'est pas "
                   f"distinguable du hasard d'échantillonnage.")

    gap_bp = 100 * (tm.loc[best_simple, "RMSE_%"] - tm.loc[best_ml, "RMSE_%"])
    out.append(f"**Face aux références simples et à D03.** Meilleure référence simple : {best_simple} (RMSE {tm.loc[best_simple, 'RMSE_%']:.4f} %). "
               f"Meilleur modèle de D03 : {d03_best} ({tm.loc[d03_best, 'RMSE_%']:.4f} %). Écart entre le meilleur ML/DL et la meilleure référence simple : "
               f"{gap_bp:+.1f} point(s) de base par séance sur le RMSE (un point de base = 0,01 % de rendement), "
               f"{'en faveur du ML/DL' if gap_bp > 0 else 'en faveur de la référence simple'}. "
               f"La **significativité économique** est donc {'faible' if abs(gap_bp) < 1.0 else 'à examiner'} : "
               f"elle ne sera établie ou infirmée que par le backtest net de coûts (D06).")

    da = tm.loc[ML, "DA"]
    ic = tm.loc[ML, "IC"]
    da_raw = [m for m in ML if tm.loc[m, "DA_PT_p"] < alpha]
    da_holm = [m for m in ML if tm.loc[m, "DA_PT_Holm"] < alpha]
    txt = (f"**Direction.** L'exactitude directionnelle des modèles ML/DL va de {_pct(da.min())} à {_pct(da.max())} "
           f"(la référence « toujours hausse » obtient {_pct(tm.loc['zero', 'DA'])}). Les corrélations prévision/réalisation (IC) vont de "
           f"{ic.min():+.3f} à {ic.max():+.3f} ("
           + ("aucune significative à 5 %" if (tm.loc[ML, "IC_p"] >= 0.05).all()
              else f"significatives à 5 %, p brute : {_names([m for m in ML if tm.loc[m, 'IC_p'] < 0.05])}") + "). ")
    if da_raw:
        det = "; ".join(f"{m} : exactitude {_pct(tm.loc[m, 'DA'])} en test contre {_pct(vm.loc[m, 'DA'])} en validation, "
                        f"p brute {tm.loc[m, 'DA_PT_p']:.3f}, p ajustée de Holm {tm.loc[m, 'DA_PT_Holm']:.3f}" for m in da_raw)
        txt += (f"Test de Pesaran-Timmermann (l'exactitude diffère-t-elle du hasard ?) : **{_names(da_raw)} dépasse(nt) le hasard à 5 % en p brute** ({det}). "
                + ("L'écart reste significatif après correction de Holm sur les modèles de la famille ML/DL. " if da_holm
                   else "Il ne l'est plus après correction de Holm sur l'ensemble des modèles du tableau. ")
                + "C'est le **seul indice favorable** de ce livrable : il n'est accompagné d'aucun gain de RMSE, il est apparu après avoir examiné "
                  "plusieurs modèles et plusieurs métriques (un p voisin de 0,01 peut survenir par hasard), et il n'existait pas en validation. "
                  "Il doit être traité comme une hypothèse à confirmer (backtest net de coûts en D06, nouvelles séances en D08), pas comme une preuve.")
    else:
        txt += "Aucun modèle n'a une exactitude directionnelle significativement supérieure au hasard (test de Pesaran-Timmermann, p brute ≥ 5 %)."
    out.append(txt)

    opt = [m for m in ML if vm.loc[m, "ratio_RMSE_zero"] < 1 and tm.loc[m, "ratio_RMSE_zero"] >= 1]
    keep = [m for m in ML if vm.loc[m, "ratio_RMSE_zero"] < 1 and tm.loc[m, "ratio_RMSE_zero"] < 1]
    if opt:
        out.append(f"**Optimisme de la validation.** {_names(opt)} faisai(en)t mieux que la marche aléatoire sur la validation (hyperparamètres choisis "
                   f"pour cela) mais pas sur le test : un avantage trouvé en validation ne s'est pas confirmé hors échantillon."
                   + (f" {_names(keep)} garde(nt) un RMSE inférieur sur les deux périodes." if keep else ""))
    elif keep:
        out.append(f"**Cohérence validation/test.** {_names(keep)} bat(tent) la marche aléatoire en RMSE sur la validation et sur le test.")
    else:
        out.append("**Validation et test.** Aucun modèle ML/DL ne bat la marche aléatoire en RMSE sur la validation ; le test confirme l'absence d'avantage.")

    stab = {m: int((ratio.loc[m] < 1).sum()) for m in ML}
    out.append(f"**Stabilité (variante B05).** Nombre de sous-périodes T1–T4 où le RMSE est inférieur à celui de la marche aléatoire : "
               + ", ".join(f"{m} {stab[m]}/{ratio.shape[1]}" for m in ML) + ". "
               + ("Un avantage présent dans toutes les sous-périodes serait un indice de robustesse ; ici "
                  + ("aucun modèle ne l'obtient." if max(stab.values()) < ratio.shape[1] else f"{_names([m for m in ML if stab[m] == ratio.shape[1]])} l'obtient.")))

    cplx = cost.loc[ML]
    out.append(f"**Parcimonie.** Les modèles ML/DL comptent de {int(cplx['parametres'].min())} à {int(cplx['parametres'].max())} paramètres "
               f"(coefficients, nœuds d'arbres ou poids) contre {int(cost.loc['ARIMA', 'parametres'])} pour l'ARIMA de D03 et 0 pour la marche aléatoire, "
               f"et exigent de {cplx['secondes_test'].min():.1f} à {cplx['secondes_test'].max():.0f} s de calcul pour le seul test "
               f"(la marche aléatoire n'en exige aucun). "
               + ("Comme aucun gain significatif n'est observé, cette complexité n'est **pas justifiée** sur cet échantillon."
                  if not sig else "Un gain significatif existe : il faut le mettre en balance avec ce coût et le confirmer en D06."))

    if sig:
        verdict = (f"**Réponse à la question B05.** Sur ce test, {_names(sig)} apporte(nt) un gain statistiquement significatif sur la marche aléatoire, "
                   f"à confirmer par la stabilité et par les coûts de transaction (D06).")
    else:
        verdict = ("**Réponse à la question B05.** Sur ce secteur défensif, les modèles complexes (ML et réseau de neurones) n'apportent pas de gain "
                   "d'erreur de prévision (RMSE) significatif face à des références simples"
                   + (", avec une réserve sur la direction du rendement (conclusion 3)" if da_raw else "") + ". Ce résultat est cohérent avec D03 (ARIMA) : le rendement quotidien de XLV est "
                   "très proche d'un bruit à moyenne quasi constante, et l'information supplémentaire contenue dans 20 variables ne suffit pas à "
                   "le prévoir hors échantillon. Les modèles de risque de D04, eux, ont un rôle plus net.")
    out.append(verdict)
    out.append("Les prévisions datées de chaque modèle (validation et test) sont dans `reports/d05_predictions.csv`. Elles alimenteront la règle de "
               "décision de D06 et l'espace Model Laboratory de l'application (D07).")
    return out


def write_report(c: dict, path: Path) -> Path:
    cfg, m, tm, vm = c["cfg"], c["manifest"], c["test_metrics"], c["val_metrics"]
    sp, sel, frozen = c["sp"], c["selected"], c["frozen"]
    L = []
    L += ["# D05 — Machine Learning et Deep Learning : prévision du rendement de XLV", "",
          "> Fichier généré par `python -m src.ml_pipeline`. Ne pas modifier à la main : relancer la commande.", ""]

    L += ["## 1. Cadre et protocole", "",
          f"- Snapshot figé : `data/snapshots/{cfg['snapshot']}` (SHA-256 `{m['sha256'][:16]}…`, vérifié au chargement). C'est le même que D02, D03 et D04 ; il est fixé dans `configs/ml.yaml`.",
          "- **Cible** (identique à D03) : r_t = ln(P_t / P_{t-1}) sur prix ajusté, prévu à la clôture de t-1 (horizon : une séance). Les métriques sont exprimées en % de rendement journalier.",
          f"- **Information** : les {c['n_features']} variables de la ligne t sont calculées avec les données jusqu'à la clôture de t-1 ; la normalisation (`StandardScaler`) est ajustée sur les seules données d'entraînement de chaque ré-estimation.",
          f"- **Échantillons** : train {sp['train']['start']} → {sp['train']['end']} ({len(c['train'])} séances) · validation {sp['validation']['start']} → {sp['validation']['end']} ({len(c['val'])}) · test {sp['test']['start']} → {sp['test']['end']} ({len(c['test'])}). L'année 2017 amorce seulement les moyennes glissantes ; elle n'est jamais utilisée pour estimer.",
          f"- **Walk-forward** : fenêtre croissante depuis le début du train, modèles ré-estimés toutes les {c['every']} séances avec les seules lignes antérieures au bloc (même protocole que D03). Le test `test_walk_forward_no_lookahead` vérifie qu'aucune cible future n'influence une prévision passée.",
          "- **Hyperparamètres** : grille limitée (`configs/ml.yaml`), évaluée **uniquement sur la validation** ; le choix est gelé dans `reports/d05_selection.json` **avant** la première prévision du test.",
          f"- **Comparaison** : mêmes données de test et mêmes coûts pour tous les modèles ; références = marche aléatoire (r̂ = 0), moyenne historique, moyenne mobile 60, et ARIMA(3,0,2) / SARIMAX de D03 ré-estimés ici sur le même snapshot. Graine globale : {cfg['seed']}.",
          f"- Tests de Diebold-Mariano (correction de Harvey-Leybourne-Newbold) contre la marche aléatoire ; correction de **Holm** sur l'ensemble des {int(tm['DM_p'].notna().sum())} comparaisons du tableau 5 pour ne pas surestimer les gains en multipliant les modèles.", ""]

    L += ["## 2. Variables explicatives", "",
          f"{c['n_features']} variables, toutes connues à la clôture de t-1 (aucune variable de la séance t) :", "",
          _md(pd.DataFrame(FEATURE_DOC, columns=["variable", "définition"]), index=False), "",
          "Le niveau de `adj_close` contient les dividendes futurs, mais tout rapport entre deux prix ajustés de dates passées (rendement, écart à une moyenne mobile) ne dépend que des dividendes situés entre ces dates : aucune fuite (voir `data/README.md`).", ""]

    L += ["## 3. Hyperparamètres : sélection sur la validation", "",
          "RMSE de validation (walk-forward, 2023-2024) de chaque candidat ; le candidat retenu par famille est celui de RMSE minimal.", "",
          _md(c["grid_table"].rename(columns={"RMSE_val_%": "RMSE_val_%", "secondes": "secondes_calcul"})[
              ["famille", "candidat", "RMSE_val_%", "secondes_calcul", "retenu", "bord_de_grille"]], "{:.4f}", index=False), "",
          f"![Hyperparamètres](figures/{c['figures']['grille'].name})", ""]
    edge = {f: v["bord_de_grille"] for f, v in frozen["models"].items() if v["bord_de_grille"]}
    L += [("- Valeur retenue en **bord de grille** : " + "; ".join(f"{f} ({', '.join(a)})" for f, a in edge.items())
           + ". La grille n'a pas été élargie après coup : le faire reviendrait à chercher l'optimum sur la validation en réutilisant ses résultats. Un optimum en bord de grille signale surtout que le modèle préfère être le plus contraint possible, c'est-à-dire proche d'une constante.")
          if edge else "- Aucun hyperparamètre retenu en bord de grille.", ""]
    L += ["Modèles retenus :", "", _md(pd.DataFrame({
        "famille": ML, "retenu": [sel[f].label for f in ML],
        "RMSE_val_%": [frozen["models"][f]["RMSE_val_pct"] for f in ML]}), "{:.4f}", index=False), ""]

    cols = ["n", "MAE_%", "RMSE_%", "DA", "R2_oos", "ratio_RMSE_zero", "DM_p"]
    L += ["## 4. Résultats sur la validation (2023-2024)", "",
          "Ces chiffres ont servi à choisir les hyperparamètres : ils sont **optimistes** pour les modèles ML/DL et ne prouvent rien. Seul le test (section 5) compte.", "",
          _md(vm[cols], "{:.4f}"), ""]

    L += ["## 5. Résultats sur le test (2025-2026)", "",
          "DA = exactitude directionnelle ; R2_oos = 1 − SSE(modèle)/SSE(moyenne historique) ; ratio_RMSE_zero < 1 : le modèle fait mieux que la marche aléatoire ; "
          "DM_p = p-value brute du test de Diebold-Mariano contre la marche aléatoire ; p_Holm = p-value ajustée ; IC = corrélation prévision/réalisation ; "
          "pente_MZ = pente de la régression du réalisé sur le prévu (1 = prévision calibrée, 0 = sans information, < 0 = à contre-sens ; non interprétable quand la prévision est quasi constante, comme pour la moyenne historique) ; "
          "DA_PT_p = p-value brute du test de Pesaran-Timmermann sur l'exactitude directionnelle, DA_PT_Holm = p-value ajustée (— si la prévision est toujours de même signe).", "",
          _md(tm[["n", "MAE_%", "RMSE_%", "MAE_prix_USD", "DA", "R2_oos", "ratio_RMSE_zero", "DM_p", "p_Holm", "IC", "pente_MZ", "DA_PT_p", "DA_PT_Holm"]], "{:.4f}"), "",
          "Comparaison à la moyenne historique (Diebold-Mariano, p-value brute) :", "",
          _md(tm.loc[[i for i in tm.index if i != "hist_mean"], ["DM_vs_moyenne", "DM_moyenne_p"]], "{:.4f}"), "",
          f"![Test](figures/{c['figures']['test'].name})", "",
          "### Prévisions confrontées aux réalisations", "",
          f"![Calibration](figures/{c['figures']['calibration'].name})", ""]

    L += ["## 6. Stabilité dans les sous-périodes du test (variante B05)", "",
          "Ratio RMSE(modèle) / RMSE(marche aléatoire) dans chaque sous-période du test, comme en D03. Une valeur < 1 indique un avantage dans la période.", "",
          _md(c["ratio"].drop(index="zero"), "{:.3f}"), "",
          f"![Stabilité](figures/{c['figures']['stabilite'].name})", ""]

    L += ["## 7. Parcimonie et coût de calcul", "",
          "Paramètres : coefficients (Ridge), nœuds d'arbres (forêt, boosting), poids d'un réseau (MLP, dont le modèle final moyenne 3 réseaux), AR + MA + constante + variance (+ coefficient SPY) pour D03. "
          "Secondes : temps cumulé d'estimation pendant le walk-forward du test (CPU). La marche aléatoire n'a aucun paramètre.", "",
          _md(c["cost"][["candidat", "parametres", "refits", "secondes_test", "ms_par_refit"]], "{:.1f}"), "",
          "### Variables utilisées", "", f"![Importances](figures/{c['figures']['importances'].name})", "",
          "Ridge : coefficients standardisés ; arbres : importance par impureté (biaisée en faveur des variables continues très variables, à lire comme une indication). Modèles ré-estimés sur train + validation, sans le test.", ""]

    L += ["## 8. Conclusions", ""] + [f"{i}. {t}" for i, t in enumerate(conclusions(c), 1)] + [""]

    L += ["## 9. Limites", "",
          "- Une seule série (XLV), une seule fréquence (journalière) et un horizon d'une séance ; 459 séances de test seulement : seuls des écarts de RMSE assez grands sont détectables.",
          "- Les gains de RMSE sur un rendement quotidien sont minuscules par nature (le rendement est dominé par le bruit) ; l'absence de significativité n'est pas la preuve qu'aucun signal n'existe, seulement qu'on n'en détecte pas ici.",
          "- Grilles d'hyperparamètres volontairement petites (cahier : recherche limitée à la validation) ; un réseau plus grand, un LSTM ou un TCN (facultatifs) n'ont pas été explorés.",
          "- Le RMSE mesure la qualité de la prévision, pas la rentabilité : la valeur économique d'une prévision légèrement meilleure se juge en D06, avec frais, glissement et turnover.",
          "- Les temps de calcul dépendent de la machine ; seuls leurs ordres de grandeur sont comparables.", "",
          f"_Logiciels : scikit-learn, PyTorch, statsmodels, scipy. Durée d'exécution : {c['elapsed']:.0f} s._", ""]
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
