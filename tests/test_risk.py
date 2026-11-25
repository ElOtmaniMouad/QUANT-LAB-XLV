"""Tests du livrable D04 — GARCH, VaR/ES, références de risque et backtests.

Points critiques : absence de fuite d'information (la prévision de la séance t n'utilise que
r_{≤t-1}), justesse des formules de VaR/ES et des tests de backtest, cohérence des unités.
Aucun test n'utilise le réseau.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from src import risk as R
from src.data import ROOT, load_config, load_snapshot
from src.models import garch as G


# --------------------------------------------------------------------------- #
# Données synthétiques
# --------------------------------------------------------------------------- #
def simulate_garch(n: int = 1500, mu=0.05, omega=0.05, alpha=0.10, beta=0.85, seed: int = 0) -> pd.Series:
    """Rendements en % suivant un GARCH(1,1) gaussien connu."""
    rng = np.random.default_rng(seed)
    r, s2 = np.zeros(n), np.zeros(n)
    s2[0] = omega / (1 - alpha - beta)
    for t in range(n):
        if t > 0:
            s2[t] = omega + alpha * (r[t - 1] - mu) ** 2 + beta * s2[t - 1]
        r[t] = mu + np.sqrt(s2[t]) * rng.standard_normal()
    return pd.Series(r, index=pd.bdate_range("2015-01-02", periods=n), name="r")


# --------------------------------------------------------------------------- #
# Formules de VaR et d'ES
# --------------------------------------------------------------------------- #
def test_normal_var_es_known_values():
    var, es = R.parametric_var_es(0.0, 1.0, 0.99)
    assert float(var) == pytest.approx(2.326348, abs=1e-5)
    assert float(es) == pytest.approx(2.665214, abs=1e-5)
    var95, es95 = R.parametric_var_es(0.0, 2.0, 0.95)
    assert float(var95) == pytest.approx(2 * 1.644854, abs=1e-5)
    assert float(es95) == pytest.approx(2 * 2.062713, abs=1e-5)


def test_mean_shifts_var():
    v0, _ = R.parametric_var_es(0.0, 1.0, 0.99)
    v1, _ = R.parametric_var_es(0.1, 1.0, 0.99)
    assert float(v0 - v1) == pytest.approx(0.1)          # un rendement attendu positif réduit la perte


@pytest.mark.parametrize("nu", [4.5, 6.4, 12.0])
def test_standardized_t_has_unit_variance_and_correct_es(nu):
    s = R.std_t_scale(nu)
    assert stats.t.var(nu) * s ** 2 == pytest.approx(1.0)
    p = 0.01
    q = R.std_quantile(p, "t", nu)
    assert stats.t.cdf(q / s, nu) == pytest.approx(p)
    # E[Z | Z ≤ q] par intégration numérique
    num, _ = integrate.quad(lambda z: z * stats.t.pdf(z / s, nu) / s, -np.inf, q)
    assert R.std_tail_mean(p, "t", nu) == pytest.approx(num / p, rel=1e-6)


def test_es_exceeds_var_and_t_tails_are_heavier():
    for dist, nu in (("normal", None), ("t", 5.0)):
        v, e = R.parametric_var_es(0.0, 1.0, 0.99, dist, nu)
        assert float(e) > float(v) > 0
    vn, _ = R.parametric_var_es(0.0, 1.0, 0.99)
    vt, _ = R.parametric_var_es(0.0, 1.0, 0.99, "t", 5.0)
    assert float(vt) > float(vn)                          # même variance, queue plus épaisse


def test_var_usd_units():
    assert R.var_to_usd(2.0, 100_000) == pytest.approx(100_000 * (1 - np.exp(-0.02)))
    assert R.var_to_usd(0.0, 100_000) == 0.0


def test_exposure_limit_bounds():
    w = R.exposure_limit(np.array([0.5, 2.0, 4.0, 10.0]), max_var_pct_capital=2.0)
    assert np.all((w >= 0) & (w <= 1))
    assert w[0] == 1.0                                     # risque faible : position pleine, sans levier
    # la VaR99 de la position réduite respecte le budget de 2 % du capital
    assert w[2] * (1 - np.exp(-0.04)) == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# Références simples : valeurs et absence de fuite
# --------------------------------------------------------------------------- #
def test_ewma_recursion_and_no_lookahead():
    r = simulate_garch(300)
    s = R.ewma_sigma(r, 0.94, init_obs=20)
    t = 100
    expected = np.sqrt(0.94 * s.iloc[t - 1] ** 2 + 0.06 * r.iloc[t - 1] ** 2)
    assert s.iloc[t] == pytest.approx(expected)
    r2 = r.copy()
    r2.iloc[t] += 50.0
    s2 = R.ewma_sigma(r2, 0.94, init_obs=20)
    assert np.allclose(s.iloc[: t + 1], s2.iloc[: t + 1], equal_nan=True)   # σ_t ne lit pas r_t
    assert s2.iloc[t + 1] > s.iloc[t + 1]


def test_historical_var_es_values_and_no_lookahead():
    r = simulate_garch(400)
    var, es = R.historical_var_es(r, 0.95, window=250)
    t = 300
    w = r.iloc[t - 250:t].to_numpy()
    q = np.quantile(w, 0.05)
    assert var.iloc[t] == pytest.approx(-q)
    assert es.iloc[t] == pytest.approx(-w[w <= q].mean())
    assert es.iloc[t] >= var.iloc[t]
    r2 = r.copy()
    r2.iloc[t] = -99.0
    v2, _ = R.historical_var_es(r2, 0.95, window=250)
    assert v2.iloc[t] == var.iloc[t]


def test_constant_moments_are_lagged():
    r = simulate_garch(200)
    mu, sd = R.constant_moments(r, min_obs=60)
    assert mu.iloc[100] == pytest.approx(r.iloc[:100].mean())
    assert sd.iloc[100] == pytest.approx(r.iloc[:100].std())


# --------------------------------------------------------------------------- #
# Pertes et backtests
# --------------------------------------------------------------------------- #
def test_qlike_minimised_at_true_variance():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 1.5, 20_000)
    grid = np.linspace(0.8, 2.5, 69)
    losses = [R.qlike(r, np.full_like(r, s)).mean() for s in grid]
    assert grid[int(np.argmin(losses))] == pytest.approx(1.5, abs=0.05)


def test_kupiec_and_christoffersen():
    n = 1000
    h = np.zeros(n, dtype=int)
    h[np.linspace(5, 995, 10).astype(int)] = 1             # 1 % d'exceptions, bien espacées
    lr, p = R.kupiec_pof(h, 0.99)
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    assert R.christoffersen(h, 0.99)["ind_p"] > 0.05
    many = np.zeros(n, dtype=int)
    many[:40] = 1                                          # 4 % et toutes groupées
    assert R.kupiec_pof(many, 0.99)[1] < 0.001
    assert R.christoffersen(many, 0.99)["ind_p"] < 0.001


def test_traffic_light_matches_basel_table():
    zones = [R.traffic_light(x, 250, 0.99) for x in range(12)]
    assert zones[:5] == ["vert"] * 5
    assert zones[5:10] == ["jaune"] * 5
    assert zones[10:] == ["rouge"] * 2


def test_var_backtest_on_correct_model_is_not_rejected():
    rng = np.random.default_rng(3)
    sigma = np.exp(rng.normal(0, 0.3, 5000))
    r = sigma * rng.standard_normal(5000)
    var, es = R.parametric_var_es(0.0, sigma, 0.99)
    tab = R.var_backtest_table(pd.Series(r), {"vrai": pd.Series(var)}, 0.99)
    assert tab.loc["vrai", "Kupiec_p"] > 0.01
    bt = R.es_backtest(r, var, es, sigma)
    assert bt["ratio_perte_ES"] == pytest.approx(1.0, abs=0.1)
    assert bt["MF_p"] > 0.01


def test_hits_sign_convention():
    # perte de 3 % (r = −3) avec une VaR de 2 % : exception ; gain de 3 % : pas d'exception
    assert list(R.hits([-3.0, 3.0, -1.0], [2.0, 2.0, 2.0])) == [1, 0, 0]


def test_fz0_prefers_true_risk_measures():
    rng = np.random.default_rng(4)
    r = rng.standard_normal(50_000)
    v, e = R.parametric_var_es(0.0, 1.0, 0.975)
    vb, eb = R.parametric_var_es(0.0, 0.7, 0.975)        # risque sous-estimé
    good = R.fz0_loss(r, np.full_like(r, v), np.full_like(r, e), 0.975).mean()
    bad = R.fz0_loss(r, np.full_like(r, vb), np.full_like(r, eb), 0.975).mean()
    assert good < bad


# --------------------------------------------------------------------------- #
# GARCH
# --------------------------------------------------------------------------- #
def test_garch_recovers_parameters():
    r = simulate_garch(4000, seed=7)
    res = G.fit(G.GarchSpec("g"), r)
    p = res.params
    assert G.persistence(p) == pytest.approx(0.95, abs=0.03)
    assert p["alpha[1]"] == pytest.approx(0.10, abs=0.04)
    row = G.summary_row(G.GarchSpec("g"), res)
    assert row["demi_vie_seances"] == pytest.approx(np.log(0.5) / np.log(row["persistance"]))


def test_garch_absorbs_arch_effect():
    # Sous le bon modèle, un test à 5 % rejette quand même 1 fois sur 20 par hasard : graine fixée
    # (sur les graines 0 à 9, seule la graine 2 rejette, avec p = 0,049).
    r = simulate_garch(2000, seed=1)
    from statsmodels.stats.diagnostic import het_arch
    assert het_arch(r - r.mean(), nlags=5)[1] < 0.01        # effet ARCH présent dans les rendements
    spec = G.GarchSpec("g")
    adm = G.admissibility(spec, G.fit(spec, r), {"max_persistence": 0.999, "arch_lm_lags": 5,
                                                 "robust_lb_lags": 10, "alpha": 0.05})
    assert adm["admissible"], adm["motif"]


def test_garch_walk_forward_no_lookahead():
    """Modifier r_t ne doit changer aucune prévision des séances ≤ t."""
    r = simulate_garch(700, seed=5)
    spec = G.GarchSpec("g", dist="t")
    start, end = r.index[400], r.index[-1]
    base = G.walk_forward(spec, r, start, end, refit_every=50).frame
    t = r.index[480]
    r2 = r.copy()
    r2[t] += 15.0
    pert = G.walk_forward(spec, r2, start, end, refit_every=50).frame
    upto = base.index <= t
    assert np.allclose(base.loc[upto, "sigma"], pert.loc[upto, "sigma"])
    assert np.allclose(base.loc[upto, "mu"], pert.loc[upto, "mu"])
    nxt = base.index[base.index > t][0]
    assert pert.loc[nxt, "sigma"] > base.loc[nxt, "sigma"]   # le choc entre dès la séance suivante


def test_garch_walk_forward_matches_recursion():
    """Entre deux ré-estimations, σ²_t = ω + α (r_{t-1} − μ)² + β σ²_{t-1} avec les paramètres du bloc."""
    r = simulate_garch(700, seed=6)
    spec = G.GarchSpec("g")
    wf = G.walk_forward(spec, r, r.index[400], r.index[-1], refit_every=50)
    p = wf.params[0][1]
    f = wf.frame
    for i in range(1, 10):
        t, prev = f.index[i], f.index[i - 1]
        s2 = p["omega"] + p["alpha[1]"] * (r[prev] - p["mu"]) ** 2 + p["beta[1]"] * f.loc[prev, "sigma"] ** 2
        assert f.loc[t, "sigma"] ** 2 == pytest.approx(s2, rel=1e-6)


# --------------------------------------------------------------------------- #
# Projet réel
# --------------------------------------------------------------------------- #
def test_risk_config_pins_existing_snapshot():
    from src.risk_management import load_risk_config
    cfg = load_risk_config()
    path = load_config().snapshot_dir / cfg["snapshot"]
    assert path.exists()
    df = load_snapshot(path)                                 # l'empreinte SHA-256 est vérifiée
    assert set(df["ticker"]) == {"XLV", "SPY"}
    assert cfg["garch_candidates"][0]["name"] == "GARCH(1,1)-N"   # le modèle exigé est toujours présent


def test_parkinson_proxy_units():
    from src.risk_management import parkinson_proxy
    df = load_snapshot(load_config().snapshot_dir / "XLV_SPY_1d_20261116.csv")
    pk = parkinson_proxy(df)
    assert (pk >= 0).all()
    # ordre de grandeur : variance journalière de XLV de l'ordre de 1 %²
    assert 0.1 < pk.mean() < 3.0
