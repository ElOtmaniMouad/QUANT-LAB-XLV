"""Tests du livrable D05 — variables, walk-forward, modèles ML/DL, tests statistiques.

Points critiques : (1) aucune fuite d'information (variables, normalisation, ré-estimation,
choix des hyperparamètres) ; (2) les modèles retrouvent un signal connu sur données
synthétiques et n'en inventent pas sur du bruit pur ; (3) justesse des tests statistiques.
Aucun test n'utilise le réseau ; les tests sur données réelles lisent le snapshot gelé du dépôt.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import ml_pipeline as P
from src.data import ROOT, load_config, load_snapshot
from src.features import build_dataset, load_model_config, modelling_sample
from src.ml_features import build_features, make_xy
from src.models import machine_learning as M
from src.models.deep_learning import TorchMLPRegressor

CFG = P.load_ml_config()


# --------------------------------------------------------------------------- #
# Données synthétiques
# --------------------------------------------------------------------------- #
def synth_xy(n: int = 700, signal: float = 0.0, k: int = 6, seed: int = 0):
    """X gaussien ; y = 0,01·(signal·x0 + bruit). signal = 0 : bruit pur."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    X = pd.DataFrame(rng.normal(size=(n, k)), index=idx, columns=[f"x{i}" for i in range(k)])
    y = pd.Series(0.01 * (signal * X["x0"] + rng.normal(size=n)), index=idx, name="y")
    return X, y


def synth_long(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Table longue au schéma de D02 pour XLV et SPY."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-03", periods=n)
    frames = []
    for t in ("XLV", "SPY"):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": t, "open": close, "high": close * 1.004, "low": close * 0.996,
            "close": close, "adj_close": close, "volume": rng.integers(5_000_000, 9_000_000, n),
            "dividends": 0.0, "stock_splits": 0.0}))
    return pd.concat(frames, ignore_index=True)


def spec(family: str, **params) -> M.ModelSpec:
    return M.ModelSpec(family, tuple(sorted(params.items())))


SPECS = {
    "Ridge": spec("Ridge", alpha=10.0),
    "RandomForest": spec("RandomForest", n_estimators=60, max_depth=3, min_samples_leaf=10, max_features=0.5),
    "GradientBoosting": spec("GradientBoosting", n_estimators=60, learning_rate=0.05, max_depth=2,
                             subsample=0.7, min_samples_leaf=10),
    "MLP": spec("MLP", hidden=8, weight_decay=0.01, dropout=0.1, epochs=40, batch_size=64, lr=0.01, n_seeds=1),
}


def oos(X, y, family, seed=0, every=21):
    start, end = X.index[400], X.index[-1]
    return M.walk_forward(lambda: M.make_estimator(SPECS[family], seed), X, y, start, end, every)


def rmse_ratio_vs_zero(y: pd.Series, pred: pd.Series) -> float:
    a = y.loc[pred.index]
    return float(np.sqrt(np.mean((a - pred) ** 2)) / np.sqrt(np.mean(a ** 2)))


# --------------------------------------------------------------------------- #
# Variables : information connue à la clôture de t-1
# --------------------------------------------------------------------------- #
def test_features_use_only_information_up_to_previous_close():
    df = synth_long()
    base = build_features(df, CFG)
    cut = pd.Timestamp(base.index[300])
    # Perturber les prix ET les volumes de XLV et SPY à la date `cut` ne change aucune variable des lignes ≤ cut.
    pert = df.copy()
    m = pert["date"] == cut
    pert.loc[m, ["adj_close", "close", "open", "high", "low"]] *= 1.5
    pert.loc[m, "volume"] *= 7
    new = build_features(pert, CFG)
    pd.testing.assert_frame_equal(base.loc[:cut], new.loc[:cut])
    assert not base.iloc[301].equals(new.iloc[301])      # la ligne suivante, elle, utilise la séance perturbée


def test_features_alignment_row_t_contains_return_of_t_minus_1():
    df = synth_long()
    X = build_features(df, CFG)
    p = df[df["ticker"] == "XLV"].set_index("date")["adj_close"]
    t = X.index[250]
    expected = np.log(p.loc[X.index[249]] / p.loc[X.index[248]])      # r_{t-1}
    assert X.loc[t, "xlv_r_lag1"] == pytest.approx(expected)
    expected2 = np.log(p.loc[X.index[248]] / p.loc[X.index[247]])     # r_{t-2}
    assert X.loc[t, "xlv_r_lag2"] == pytest.approx(expected2)


def test_features_truncation_invariance():
    """Calculer sur une histoire tronquée donne les mêmes variables que sur l'histoire complète."""
    df = synth_long()
    full = build_features(df, CFG)
    cut = full.index[320]
    part = build_features(df[df["date"] <= cut], CFG)
    pd.testing.assert_frame_equal(full.loc[:cut], part.loc[:cut])


def test_feature_names_and_count():
    X = build_features(synth_long(), CFG)
    assert X.shape[1] == 20
    assert X.columns.is_unique


# --------------------------------------------------------------------------- #
# Walk-forward : pas de fuite, normalisation sur l'entraînement seul
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", ["Ridge", "RandomForest", "GradientBoosting", "MLP"])
def test_walk_forward_no_lookahead(family):
    X, y = synth_xy(signal=0.3)
    base = oos(X, y, family)
    # d = DÉBUT d'un bloc : c'est la séance dont la prévision serait faussée si le modèle voyait sa propre cible.
    d = base.train_sizes[3]
    y2 = y.copy()
    y2.iloc[d] += 0.5                                     # énorme choc sur la cible du jour d
    alt = oos(X, y2, family)
    nxt = next(b for b in base.train_sizes if b > d)      # première ré-estimation qui peut voir y_d
    before = base.predictions.index[base.predictions.index < X.index[nxt]]
    assert X.index[d] in before                           # la séance perturbée est bien comparée
    pd.testing.assert_series_equal(base.predictions.loc[before], alt.predictions.loc[before])
    assert not np.allclose(base.predictions.iloc[-20:], alt.predictions.iloc[-20:])   # le futur, lui, change


class _Recorder:
    """Estimateur factice : enregistre ce qu'il reçoit à l'apprentissage et prédit 0."""
    fits: list = []

    def fit(self, X, y):
        _Recorder.fits.append((len(X), float(np.asarray(y)[-1])))
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_each_fit_receives_exactly_the_rows_before_its_block():
    X, y = synth_xy()
    _Recorder.fits = []
    res = M.walk_forward(_Recorder, X, y, X.index[400], X.index[-1], 21)
    n_rows = [n for n, _ in _Recorder.fits]
    assert n_rows == res.train_sizes == list(range(400, len(X), 21))       # b lignes, jamais b+1
    # la dernière cible vue à l'apprentissage est celle de la séance b-1 (×100), pas celle de la séance b
    for (n, last_y), b in zip(_Recorder.fits, res.train_sizes):
        assert last_y == pytest.approx(100 * y.iloc[b - 1])


def test_walk_forward_schedule_and_coverage():
    X, y = synth_xy()
    res = oos(X, y, "Ridge")
    first = int(np.flatnonzero(X.index >= X.index[400])[0])
    assert res.train_sizes == list(range(first, len(X), 21))      # fenêtre croissante, ré-estimation tous les 21
    assert res.n_refits == len(res.train_sizes)
    assert res.predictions.index[0] == X.index[400] and res.predictions.index[-1] == X.index[-1]
    assert res.predictions.notna().all()


def test_scaler_is_fitted_on_training_rows_only():
    X, y = synth_xy()
    res = oos(X, y, "Ridge")
    b = res.train_sizes[-1]
    scaler = res.last_estimator.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, X.iloc[:b].mean().to_numpy())
    np.testing.assert_allclose(scaler.scale_, X.iloc[:b].std(ddof=0).to_numpy())


def test_walk_forward_rejects_short_history():
    X, y = synth_xy()
    with pytest.raises(ValueError):
        M.walk_forward(lambda: M.make_estimator(SPECS["Ridge"]), X, y, X.index[20], X.index[-1])


@pytest.mark.parametrize("family", ["Ridge", "RandomForest", "GradientBoosting", "MLP"])
def test_predictions_are_deterministic_and_in_return_units(family):
    X, y = synth_xy(signal=0.3)
    a, b = oos(X, y, family, seed=7).predictions, oos(X, y, family, seed=7).predictions
    pd.testing.assert_series_equal(a, b)
    assert a.abs().max() < 0.05                 # ≈ rendements (0,01), pas des « % » (×100)
    assert a.std() > 0                           # le modèle n'est pas figé sur une constante


# --------------------------------------------------------------------------- #
# Capacité de détection : signal connu retrouvé, bruit pur non « appris »
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", ["Ridge", "RandomForest", "GradientBoosting", "MLP"])
def test_models_recover_a_known_signal(family):
    X, y = synth_xy(signal=0.5, seed=3)
    assert rmse_ratio_vs_zero(y, oos(X, y, family).predictions) < 0.97


@pytest.mark.parametrize("family", ["Ridge", "RandomForest", "GradientBoosting", "MLP"])
def test_models_do_not_invent_signal_on_pure_noise(family):
    X, y = synth_xy(signal=0.0, seed=4)
    assert rmse_ratio_vs_zero(y, oos(X, y, family).predictions) > 0.985


# --------------------------------------------------------------------------- #
# Hyperparamètres : sélection sur la validation uniquement
# --------------------------------------------------------------------------- #
def test_hyperparameter_selection_ignores_data_after_validation():
    X, y = synth_xy(signal=0.3)
    specs = M.expand_grid("Ridge", {"alpha": [1, 100, 10000]})
    v0, v1 = X.index[400], X.index[549]
    t1, _ = M.validate_grid(specs, X, y, v0, v1, 21, 0)
    y2 = y.copy()
    y2.loc[X.index > v1] = 5.0                       # le « test » est ravagé : la sélection ne doit pas bouger
    t2, _ = M.validate_grid(specs, X, y2, v0, v1, 21, 0)
    pd.testing.assert_series_equal(t1["RMSE_val_%"], t2["RMSE_val_%"])
    assert M.pick_best(t1, specs) == M.pick_best(t2, specs)


def test_config_grids():
    sizes = {f: len(M.expand_grid(f, CFG["models"][f])) for f in M.FAMILIES}
    assert sizes == {"Ridge": 5, "RandomForest": 4, "GradientBoosting": 4, "MLP": 4}
    for fam in M.FAMILIES:
        specs = M.expand_grid(fam, CFG["models"][fam])
        assert len({s.label for s in specs}) == len(specs)                  # étiquettes uniques
        assert all(M.spec_from_dict(M.spec_to_dict(s)) == s for s in specs)  # aller-retour JSON
    with pytest.raises(ValueError):
        M.expand_grid("SVM", {})


def test_edge_axes_flags_only_real_grids():
    s = M.expand_grid("Ridge", CFG["models"]["Ridge"])
    assert P._edge_axes(s[-1], CFG["models"]["Ridge"]) == ["alpha"]
    assert P._edge_axes(s[2], CFG["models"]["Ridge"]) == []
    two = M.expand_grid("MLP", CFG["models"]["MLP"])[0]
    assert P._edge_axes(two, CFG["models"]["MLP"]) == []        # grilles à 2 valeurs : jamais signalées


# --------------------------------------------------------------------------- #
# Modèles : taille, MLP
# --------------------------------------------------------------------------- #
def test_parameter_counts():
    X, y = synth_xy(k=20)
    ridge = M.make_estimator(SPECS["Ridge"]).fit(X.to_numpy(), y.to_numpy())
    assert M.n_parameters(ridge) == 21
    mlp = M.make_estimator(SPECS["MLP"]).fit(X.to_numpy(), y.to_numpy() * 100)
    assert M.n_parameters(mlp) == 20 * 8 + 8 + 8 + 1
    rf = M.make_estimator(SPECS["RandomForest"]).fit(X.to_numpy(), y.to_numpy())
    assert M.n_parameters(rf) > 60
    assert M.feature_importance(mlp, list(X.columns)) is None
    assert M.feature_importance(rf, list(X.columns)).sum() == pytest.approx(1.0)


def test_mlp_is_a_scikit_learn_compatible_estimator():
    est = TorchMLPRegressor(hidden=4, epochs=5, n_seeds=2, seed=1)
    assert est.get_params()["hidden"] == 4
    X, y = synth_xy(n=200)
    est.fit(X.to_numpy(), y.to_numpy())
    assert est.predict(X.to_numpy()).shape == (200,)
    assert len(est.nets_) == 2


# --------------------------------------------------------------------------- #
# Tests statistiques
# --------------------------------------------------------------------------- #
def test_holm_correction():
    adj = P.holm(pd.Series({"a": 0.01, "b": 0.04, "c": 0.03, "d": np.nan}))
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert np.isnan(adj["d"])
    assert (adj.dropna() >= pd.Series({"a": 0.01, "b": 0.04, "c": 0.03})).all()


def test_pesaran_timmermann():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=400))
    good = y + rng.normal(0, 0.3, 400)
    assert P.pesaran_timmermann(y, good)[1] < 1e-6                      # direction très bien prévue
    assert P.pesaran_timmermann(y, pd.Series(rng.normal(size=400)))[1] > 0.01   # prévision indépendante
    z, p = P.pesaran_timmermann(y, pd.Series(0.1, index=y.index))       # signe constant : test sans objet
    assert np.isnan(z) and np.isnan(p)


def test_ic_and_mincer_zarnowitz():
    rng = np.random.default_rng(1)
    y = pd.Series(rng.normal(size=300))
    r, p = P.information_coefficient(y, y)
    assert r == pytest.approx(1.0) and p < 1e-6
    assert P.mincer_zarnowitz_slope(y, y) == pytest.approx(1.0)
    assert P.mincer_zarnowitz_slope(y, 2 * y) == pytest.approx(0.5)
    assert np.isnan(P.mincer_zarnowitz_slope(y, pd.Series(0.0, index=y.index)))


# --------------------------------------------------------------------------- #
# Données réelles et livrables générés (snapshot gelé du dépôt)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real():
    path = ROOT / "data" / "snapshots" / CFG["snapshot"]
    if not path.exists():
        pytest.skip("snapshot gelé absent")
    df = load_snapshot(path)                          # vérifie aussi le SHA-256
    dc, mc = load_config(), load_model_config()
    ds = modelling_sample(build_dataset(df, dc, mc))
    return df, dc, ds


def test_real_features_cover_every_modelling_row(real):
    df, _, ds = real
    X = build_features(df, CFG)
    Xs, ys = make_xy(ds, X)
    assert len(Xs) == len(ds) and Xs.notna().all().all()
    assert Xs.shape[1] == 20 and ds.index.min() == pd.Timestamp("2018-01-02")


def test_real_features_do_not_depend_on_the_future(real):
    df, _, _ = real
    full = build_features(df, CFG)
    cut = pd.Timestamp("2024-06-28")
    part = build_features(df[df["date"] <= cut], CFG)
    pd.testing.assert_frame_equal(full.loc[:cut], part.loc[:cut])


def test_frozen_selection_is_consistent(real):
    path = ROOT / "reports" / "d05_selection.json"
    if not path.exists():
        pytest.skip("pipeline D05 non lancé")
    sel = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "snapshots" / CFG["snapshot"].replace(".csv", ".json")).read_text(encoding="utf-8"))
    assert sel["snapshot"] == CFG["snapshot"] and sel["snapshot_sha256"] == manifest["sha256"]
    assert set(sel["models"]) == set(M.FAMILIES) and len(sel["features"]) == 20
    for fam, d in sel["models"].items():
        grid = M.expand_grid(fam, CFG["models"][fam])
        assert M.spec_from_dict(d) in grid                                  # le choix appartient à la grille gelée


def test_generated_predictions_cover_validation_and_test(real):
    path = ROOT / "reports" / "d05_predictions.csv"
    if not path.exists():
        pytest.skip("pipeline D05 non lancé")
    _, dc, ds = real
    out = pd.read_csv(path, index_col=0, parse_dates=True)
    sp = dc.splits
    val = out[out["split"] == "validation"]
    test = out[out["split"] == "test"]
    assert val.index.min() >= pd.Timestamp(sp["validation"]["start"]) and val.index.max() <= pd.Timestamp(sp["validation"]["end"])
    assert test.index.min() >= pd.Timestamp(sp["test"]["start"]) and test.index.max() <= pd.Timestamp(sp["test"]["end"])
    assert len(test) == 459 and test[list(M.FAMILIES)].notna().all().all()
    np.testing.assert_allclose(out["y"], ds["y"].reindex(out.index), atol=1e-7)   # cible = celle de D03
