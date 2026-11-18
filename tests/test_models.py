"""Tests du livrable D03 — références, ARIMA/SARIMA/SARIMAX, métriques.

Les points critiques testés sont l'absence de fuite d'information (une prévision
pour la séance t ne dépend jamais de y_t ni des séances suivantes) et la
justesse des métriques. Aucun test n'utilise le réseau.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import evaluation as E
from src.data import load_config, load_snapshot
from src.features import build_dataset, load_model_config, modelling_sample
from src.models import baseline as B
from src.models.arima import (
    ArimaSpec, admissibility, fit, residual_diagnostics, robust_ljung_box, root_checks, walk_forward,
)
from statsmodels.stats.diagnostic import acorr_ljungbox


# --------------------------------------------------------------------------- #
# Données synthétiques
# --------------------------------------------------------------------------- #
def synthetic_sample(n: int = 400, phi: float = 0.0, beta: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """AR(1) + effet d'une variable externe connue la veille, échelle « rendements »."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.01, n)
    e = rng.normal(0, 0.01, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + beta * x[t] + e[t]
    idx = pd.bdate_range("2018-01-02", periods=n)
    price = 100 * np.exp(np.cumsum(y))
    return pd.DataFrame({"y": y, "x_spy_lag1": x, "price": price,
                         "price_prev": np.r_[100.0, price[:-1]]}, index=idx)


# --------------------------------------------------------------------------- #
# Jeu de modélisation
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dataset():
    return build_dataset(load_snapshot())


def test_dataset_alignment(dataset):
    ds = dataset
    snap = load_snapshot()
    px = snap[snap.ticker == "XLV"].set_index("date")["adj_close"]
    spy = snap[snap.ticker == "SPY"].set_index("date")["adj_close"]
    d = ds.index[100]
    prev = px.index[px.index.get_loc(d) - 1]
    prev2 = px.index[px.index.get_loc(d) - 2]
    assert ds.loc[d, "y"] == pytest.approx(np.log(px[d] / px[prev]))
    # la variable externe de la ligne t est le rendement SPY de t-1 (connu en t-1)
    assert ds.loc[d, "x_spy_lag1"] == pytest.approx(np.log(spy[prev] / spy[prev2]))
    assert ds.loc[d, "price_prev"] == pytest.approx(px[prev])


def test_modelling_sample_excludes_warmup(dataset):
    ms = modelling_sample(dataset)
    assert ms.index.min() >= pd.Timestamp("2018-01-01")
    assert "warmup" not in set(ms["split"])
    assert not ms[["y", "x_spy_lag1"]].isna().any().any()


def test_test_subperiods_inside_test_split():
    sp = load_config().splits["test"]
    for a, b in load_model_config().subperiods.values():
        assert pd.Timestamp(sp["start"]) <= a <= b <= pd.Timestamp(sp["end"])


# --------------------------------------------------------------------------- #
# Références
# --------------------------------------------------------------------------- #
def test_baselines_values():
    y = pd.Series([0.01, -0.02, 0.03, 0.00, 0.02])
    assert np.isnan(B.zero(y).iloc[0]) and (B.zero(y).iloc[1:] == 0).all()
    assert B.naive(y).iloc[3] == pytest.approx(0.03)
    assert B.hist_mean(y).iloc[3] == pytest.approx(np.mean([0.01, -0.02, 0.03]))
    ma = B.moving_avg(y, 2)
    assert np.isnan(ma.iloc[1]) and ma.iloc[4] == pytest.approx(0.015)


@pytest.mark.parametrize("make", [B.naive, B.hist_mean, lambda y: B.moving_avg(y, 3)])
def test_baselines_no_lookahead(make):
    y = synthetic_sample(60)["y"]
    base = make(y)
    y2 = y.copy()
    y2.iloc[30:] += 5.0          # on modifie le présent et le futur
    pd.testing.assert_series_equal(make(y2).iloc[:31], base.iloc[:31])


# --------------------------------------------------------------------------- #
# ARIMA / SARIMAX
# --------------------------------------------------------------------------- #
def test_spec_labels():
    assert ArimaSpec("a", (1, 0, 1)).family == "ARIMA"
    assert ArimaSpec("s", (1, 0, 1), (0, 0, 1, 5)).label == "SARIMA(1,0,1)(0,0,1)_5"
    assert ArimaSpec("x", (0, 0, 0), exog=True).family == "SARIMAX"


def test_fit_recovers_ar_coefficient():
    df = synthetic_sample(1500, phi=0.5, seed=1)
    res = fit(ArimaSpec("a", (1, 0, 0)), df["y"])
    assert res.params[1] == pytest.approx(0.5, abs=0.06)


def test_fit_recovers_exog_coefficient():
    df = synthetic_sample(1500, beta=0.6, seed=2)
    res = fit(ArimaSpec("x", (0, 0, 0), exog=True), df["y"], df[["x_spy_lag1"]])
    assert res.params[1] == pytest.approx(0.6, abs=0.08)


@pytest.mark.parametrize("spec", [
    ArimaSpec("a", (1, 0, 1)),
    ArimaSpec("x", (1, 0, 0), exog=True),
])
def test_walk_forward_no_lookahead(spec):
    """Modifier y à partir de la séance k ne change aucune prévision ≤ k."""
    df = synthetic_sample(300, phi=0.3, beta=0.4, seed=3)
    start, end = df.index[200], df.index[-1]
    base = walk_forward(spec, df, start, end, refit_every=25).predictions
    k = 240
    df2 = df.copy()
    df2.iloc[k:, df2.columns.get_loc("y")] += 0.05
    alt = walk_forward(spec, df2, start, end, refit_every=25).predictions
    upto = df.index[k]
    np.testing.assert_allclose(alt[:upto].to_numpy(), base[:upto].to_numpy(), rtol=1e-6, atol=1e-10)
    assert not np.allclose(alt[upto:].iloc[1:].to_numpy(), base[upto:].iloc[1:].to_numpy())


def test_walk_forward_covers_window_and_beats_zero_on_ar_data():
    df = synthetic_sample(700, phi=0.5, seed=4)
    start, end = df.index[500], df.index[-1]
    wf = walk_forward(ArimaSpec("a", (1, 0, 0)), df, start, end, refit_every=50)
    assert wf.predictions.index[0] == start and wf.predictions.index[-1] == end
    assert wf.predictions.notna().all()
    assert wf.n_refits == int(np.ceil(200 / 50))
    y = df.loc[start:end, "y"]
    assert E.rmse(y, wf.predictions) < E.rmse(y, pd.Series(0.0, index=y.index))


def test_walk_forward_rejects_short_history():
    df = synthetic_sample(100)
    with pytest.raises(ValueError):
        walk_forward(ArimaSpec("a", (1, 0, 0)), df, df.index[10], df.index[-1])


def test_residual_diagnostics_white_noise():
    df = synthetic_sample(1000, seed=5)
    d = residual_diagnostics(fit(ArimaSpec("w", (0, 0, 0)), df["y"]))
    assert set(d["verdict"]) <= {"H0 rejetée", "H0 non rejetée", "non applicable"}
    assert d["test"].str.contains("robuste").sum() == 3
    jb = d[d["test"] == "Jarque-Bera"].iloc[0]
    assert jb["p_value"] > 0.01          # données gaussiennes simulées


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #
def test_metrics_values():
    a = pd.Series([0.01, -0.02, 0.03, -0.01])
    p = pd.Series([0.00, -0.01, 0.01, 0.01])
    assert E.mae(a, p) == pytest.approx(np.mean([0.01, 0.01, 0.02, 0.02]))
    assert E.rmse(a, p) == pytest.approx(np.sqrt(np.mean([1e-4, 1e-4, 4e-4, 4e-4])))
    # convention r̂ ≥ 0 ⇒ hausse : 0.00 prévoit une hausse alors que 0.01 monte -> correct
    assert E.directional_accuracy(a, p) == pytest.approx(0.75)
    assert E.r2_oos(a, p, p) == pytest.approx(0.0)


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(6)
    a = pd.Series(rng.normal(0, 1, 500))
    good = a + rng.normal(0, 0.3, 500)
    bad = pd.Series(np.zeros(500))
    stat, p = E.diebold_mariano(a, good, bad)
    assert stat < 0 and p < 0.01
    stat2, p2 = E.diebold_mariano(a, bad, bad)
    assert np.isnan(stat2) and np.isnan(p2)


def test_evaluate_table():
    df = synthetic_sample(200, seed=7)
    preds = B.all_baselines(df["y"], [5]).dropna()
    tab = E.evaluate(df.loc[preds.index], preds)
    assert tab.loc["zero", "ratio_RMSE_zero"] == pytest.approx(1.0)
    assert tab.loc["hist_mean", "R2_oos"] == pytest.approx(0.0)
    assert (tab["n"] == len(preds)).all()


# --------------------------------------------------------------------------- #
# Diagnostics utilisés pour la sélection (revue du binôme)
# --------------------------------------------------------------------------- #
def arch_noise(n: int, seed: int) -> np.ndarray:
    """Bruit blanc NON autocorrélé mais à variance conditionnelle forte (ARCH(1))."""
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = np.sqrt(0.2 + 0.7 * e[t - 1] ** 2) * rng.standard_normal()
    return e


def test_robust_ljung_box_controls_size_under_arch():
    """Sur un bruit ARCH, le Ljung-Box classique rejette trop souvent, pas la version robuste."""
    rej_classic = rej_robust = 0
    n_sim = 60
    for seed in range(n_sim):
        e = arch_noise(1500, seed)
        rej_classic += acorr_ljungbox(e, lags=[10])["lb_pvalue"].iloc[0] < 0.05
        rej_robust += robust_ljung_box(e, 10)[1] < 0.05
    assert rej_robust / n_sim <= 0.15
    assert rej_classic > rej_robust


def test_robust_ljung_box_detects_autocorrelation():
    y = synthetic_sample(1500, phi=0.3, seed=8)["y"].to_numpy()
    assert robust_ljung_box(y, 10)[1] < 0.01


def test_root_checks_detect_cancellation():
    # ARMA(1,1) estimé sur un bruit blanc : la racine AR et la racine MA tendent à se compenser
    df = synthetic_sample(1000, seed=9)
    r = fit(ArimaSpec("c", (1, 0, 1)), df["y"])
    rc = root_checks(r)
    assert set(rc) == {"ar_min_modulus", "ma_min_modulus", "root_cancellation"}
    assert rc["root_cancellation"] >= 0
    # Modèle sans MA : aucune compensation possible
    assert root_checks(fit(ArimaSpec("a", (1, 0, 0)), df["y"]))["root_cancellation"] == float("inf")


def test_admissibility_rules():
    rules = load_model_config()["selection"]
    ok = admissibility(fit(ArimaSpec("a", (1, 0, 0)), synthetic_sample(1500, phi=0.4, seed=10)["y"]), rules)
    assert ok["admissible"] and ok["motif"] == "—"
    # bruit blanc fortement autocorrélé, ajusté sans dynamique : résidus autocorrélés -> écarté
    bad = admissibility(fit(ArimaSpec("w", (0, 0, 0)), synthetic_sample(1500, phi=0.4, seed=10)["y"]), rules)
    assert not bad["admissible"] and "autocorrélés" in bad["motif"]
