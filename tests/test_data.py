"""Tests du livrable D02 — collecte, snapshot et contrôles qualité.

Aucun test n'utilise le réseau : les contrôles sont éprouvés sur des données
synthétiques dans lesquelles on injecte une anomalie à la fois, puis sur le
snapshot gelé du dépôt.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src import data as D
from src import quality as Q


# --------------------------------------------------------------------------- #
# Jeu synthétique propre
# --------------------------------------------------------------------------- #
def make_clean(ticker: str = "XLV", start="2024-01-02", end="2024-06-28", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = Q.nyse_sessions(start, end)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, len(dates))))
    open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    return pd.DataFrame({
        "date": dates, "ticker": ticker, "open": open_, "high": high, "low": low,
        "close": close, "adj_close": close.copy(),
        "volume": pd.array(rng.integers(5e6, 1e7, len(dates)), dtype="Int64"),
        "dividends": 0.0, "stock_splits": 0.0,
    })[D.COLUMNS]


@pytest.fixture
def config(tmp_path) -> D.DataConfig:
    cfg = D.load_config()
    splits = {"train": {"start": "2024-01-01", "end": "2024-06-30"}}
    return replace(cfg, snapshot_dir=tmp_path / "snapshots", splits=splits)


@pytest.fixture
def clean(config) -> pd.DataFrame:
    return pd.concat([make_clean(config.asset, seed=1), make_clean(config.benchmark, seed=2)],
                     ignore_index=True)


def status(checks: pd.DataFrame, name: str, ticker: str) -> str:
    row = checks[(checks["controle"] == name) & (checks["ticker"] == ticker)]
    assert len(row) == 1, f"contrôle {name}/{ticker} introuvable"
    return row["statut"].iloc[0]


# --------------------------------------------------------------------------- #
# Calendrier NYSE
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("closed", [
    "2018-12-05",  # deuil national G. H. W. Bush
    "2019-04-19",  # Vendredi saint
    "2020-07-03",  # 4 juillet tombant un samedi -> observé le vendredi
    "2022-06-20",  # Juneteenth tombant un dimanche -> observé le lundi
    "2023-01-02",  # 1er janvier tombant un dimanche -> observé le lundi
    "2024-11-28",  # Thanksgiving
    "2025-01-09",  # deuil national J. Carter
    "2026-01-19",  # Martin Luther King Jr.
])
def test_nyse_closed_days(closed):
    assert pd.Timestamp(closed) not in Q.nyse_sessions("2017-01-01", "2026-12-31")


@pytest.mark.parametrize("opened", [
    "2021-06-18",  # Juneteenth pas encore férié au NYSE en 2021
    "2021-12-31",  # 1er janvier 2022 un samedi : pas de report au vendredi
    "2024-11-29",  # lendemain de Thanksgiving (séance écourtée mais ouverte)
])
def test_nyse_open_days(opened):
    assert pd.Timestamp(opened) in Q.nyse_sessions("2017-01-01", "2026-12-31")


@pytest.mark.parametrize("year,n", [(2018, 251), (2019, 252), (2020, 253), (2021, 252),
                                    (2022, 251), (2023, 250), (2024, 252), (2025, 250)])
def test_nyse_sessions_per_year(year, n):
    assert len(Q.nyse_sessions(f"{year}-01-01", f"{year}-12-31")) == n


# --------------------------------------------------------------------------- #
# Collecte : normalisation et barre du jour
# --------------------------------------------------------------------------- #
def test_normalize_history_handles_timezone_and_schema():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"]).tz_localize("America/New_York")
    raw = pd.DataFrame({"Open": [1.0, 2.0], "High": [1.5, 2.5], "Low": [0.5, 1.5],
                        "Close": [1.2, 2.2], "Adj Close": [1.1, 2.1], "Volume": [10, 20],
                        "Dividends": [0.0, 0.1], "Stock Splits": [0.0, 0.0],
                        "Capital Gains": [0.0, 0.0]}, index=idx)
    out = D.normalize_history(raw, "XLV")
    assert list(out.columns) == D.COLUMNS
    assert out["date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert out["date"].dt.tz is None


def test_normalize_history_rejects_empty():
    with pytest.raises(ValueError):
        D.normalize_history(pd.DataFrame(), "XLV")


def test_partial_bar_of_current_session_is_excluded():
    # 15 h 00 à New York un jour de séance : la barre du jour n'est pas définitive.
    during = datetime(2026, 11, 16, 20, 0, tzinfo=timezone.utc)
    assert D.last_closed_session_date(during) == pd.Timestamp("2026-11-15")
    # 19 h 00 à New York : la séance est clôturée.
    after = datetime(2026, 11, 17, 0, 0, tzinfo=timezone.utc)
    assert D.last_closed_session_date(after) == pd.Timestamp("2026-11-16")


# --------------------------------------------------------------------------- #
# Snapshot : aller-retour, empreinte, non-écrasement
# --------------------------------------------------------------------------- #
def test_snapshot_roundtrip_and_manifest(clean, config):
    csv_path, json_path = D.save_snapshot(clean, config)
    assert csv_path.exists() and json_path.exists()
    manifest = D.read_manifest(csv_path)
    assert manifest["sha256"] == D.sha256_file(csv_path)
    assert manifest["rows"] == len(clean)
    assert set(manifest["tickers"]) == {config.asset, config.benchmark}

    loaded = D.load_snapshot(config=config)  # dernier snapshot, sans réseau
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True), clean.reset_index(drop=True),
        check_exact=False, atol=1e-6, check_dtype=False,
    )


def test_snapshot_is_never_overwritten(clean, config):
    D.save_snapshot(clean, config)
    with pytest.raises(FileExistsError):
        D.save_snapshot(clean, config)


def test_tampered_snapshot_is_rejected(clean, config):
    csv_path, _ = D.save_snapshot(clean, config)
    csv_path.write_text(csv_path.read_text().replace("XLV", "XLW", 1))
    with pytest.raises(ValueError, match="SHA-256"):
        D.load_snapshot(csv_path)


def test_load_without_snapshot_gives_actionable_error(config):
    with pytest.raises(FileNotFoundError, match="collect"):
        D.load_snapshot(config=config)


# --------------------------------------------------------------------------- #
# Rendements et périodes : pas d'information future
# --------------------------------------------------------------------------- #
def test_returns_definition_and_no_lookahead():
    p = pd.Series([100.0, 110.0, 99.0], index=pd.date_range("2024-01-02", periods=3))
    r = D.log_returns(p)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(1.1))
    assert D.simple_returns(p).iloc[2] == pytest.approx(-0.1)
    # Modifier le futur ne change aucun rendement passé.
    p2 = p.copy()
    p2.iloc[2] = 500.0
    pd.testing.assert_series_equal(D.log_returns(p2).iloc[:2], r.iloc[:2])


def test_splits_are_disjoint_ordered_and_cover_real_config():
    cfg = D.load_config()
    dates = Q.nyse_sessions("2017-01-03", "2026-11-13")
    labels = D.assign_split(dates, cfg.splits)
    assert "hors_periode" not in set(labels)
    order = ["warmup", "train", "validation", "test", "paper"]
    last = [labels[labels == name].index.max() for name in order]
    first = [labels[labels == name].index.min() for name in order]
    assert all(last[i] < first[i + 1] for i in range(len(order) - 1))
    assert first[3] == pd.Timestamp("2025-01-02") and last[3] == pd.Timestamp("2026-10-30")


# --------------------------------------------------------------------------- #
# Contrôles qualité : chaque anomalie injectée doit être détectée
# --------------------------------------------------------------------------- #
def test_clean_data_has_no_failure(clean, config):
    checks = Q.run_quality_checks(clean, config)
    assert not (checks["statut"] == Q.FAIL).any(), checks[checks["statut"] == Q.FAIL]


def _xlv_rows(df):
    return df.index[df["ticker"] == "XLV"]


def test_detects_duplicate_date(clean, config):
    bad = pd.concat([clean, clean.iloc[[5]]], ignore_index=True)
    assert status(Q.run_quality_checks(bad, config), "dates_dupliquees", "XLV") == Q.FAIL


def test_detects_missing_value(clean, config):
    bad = clean.copy()
    bad.loc[_xlv_rows(bad)[10], "close"] = np.nan
    assert status(Q.run_quality_checks(bad, config), "valeurs_manquantes", "XLV") == Q.FAIL


def test_detects_non_positive_price(clean, config):
    bad = clean.copy()
    bad.loc[_xlv_rows(bad)[10], "low"] = 0.0
    assert status(Q.run_quality_checks(bad, config), "prix_positifs", "XLV") == Q.FAIL


def test_detects_ohlc_inconsistency(clean, config):
    bad = clean.copy()
    i = _xlv_rows(bad)[10]
    bad.loc[i, "high"] = bad.loc[i, "close"] * 0.9  # high < close
    assert status(Q.run_quality_checks(bad, config), "coherence_ohlc", "XLV") == Q.FAIL


def test_detects_zero_and_negative_volume(clean, config):
    bad = clean.copy()
    bad.loc[_xlv_rows(bad)[10], "volume"] = 0
    assert status(Q.run_quality_checks(bad, config), "volume", "XLV") == Q.WARN
    bad.loc[_xlv_rows(bad)[11], "volume"] = -5
    assert status(Q.run_quality_checks(bad, config), "volume", "XLV") == Q.FAIL


def test_detects_missing_session_and_desynchronisation(clean, config):
    bad = clean.drop(index=_xlv_rows(clean)[20]).reset_index(drop=True)
    checks = Q.run_quality_checks(bad, config)
    assert status(checks, "seances_manquantes", "XLV") == Q.FAIL
    assert status(checks, "seances_manquantes", "SPY") == Q.OK
    assert status(checks, "synchronisation", "XLV/SPY") == Q.FAIL
    assert status(checks, "couverture_train", "XLV") == Q.FAIL


def test_detects_weekend_row(clean, config):
    extra = clean.iloc[[3]].copy()
    extra["date"] = pd.Timestamp("2024-03-09")  # samedi
    bad = pd.concat([clean, extra]).sort_values(["ticker", "date"]).reset_index(drop=True)
    assert status(Q.run_quality_checks(bad, config), "seances_inattendues", "XLV") == Q.FAIL


def test_detects_stale_close(clean, config):
    bad = clean.copy()
    rows = _xlv_rows(bad)[30:34]
    for col in ["open", "high", "low", "close", "adj_close"]:
        bad.loc[rows, col] = bad.loc[rows[0], col]
    assert status(Q.run_quality_checks(bad, config), "clotures_figees", "XLV") == Q.WARN


def test_flags_extreme_return_as_info(clean, config):
    bad = clean.copy()
    rows = _xlv_rows(bad)[40:]
    for col in ["open", "high", "low", "close", "adj_close"]:
        bad.loc[rows, col] *= 0.80  # saut de -20 % en une séance
    checks = Q.run_quality_checks(bad, config)
    row = checks[(checks["controle"] == "rendements_extremes") & (checks["ticker"] == "XLV")].iloc[0]
    assert row["statut"] == Q.INFO and row["n"] == 1


def _with_dividend(df: pd.DataFrame, pos: int = 60, amount: float = 0.5) -> pd.DataFrame:
    """Applique le rétro-ajustement Yahoo d'un dividende à XLV."""
    out = df.copy()
    rows = _xlv_rows(out)
    prev_close = out.loc[rows[pos - 1], "close"]
    out.loc[rows[:pos], "adj_close"] *= 1 - amount / prev_close
    out.loc[rows[pos], "dividends"] = amount
    return out


def test_consistent_dividend_passes(clean, config):
    checks = Q.run_quality_checks(_with_dividend(clean), config)
    assert status(checks, "dividendes_coherents", "XLV") == Q.OK
    assert status(checks, "ajustement_hors_dividende", "XLV") == Q.OK


def test_detects_dividend_not_reflected_in_adj_close(clean, config):
    bad = clean.copy()
    bad.loc[_xlv_rows(bad)[60], "dividends"] = 0.5  # déclaré mais adj_close inchangé
    assert status(Q.run_quality_checks(bad, config), "dividendes_coherents", "XLV") == Q.WARN


def test_detects_adjustment_jump_without_dividend(clean, config):
    bad = _with_dividend(clean)
    bad.loc[_xlv_rows(bad)[60], "dividends"] = 0.0  # ajustement présent, dividende absent
    assert status(Q.run_quality_checks(bad, config), "ajustement_hors_dividende", "XLV") == Q.FAIL


def test_detects_adj_close_above_close(clean, config):
    bad = clean.copy()
    bad.loc[_xlv_rows(bad)[10], "adj_close"] *= 1.05
    assert status(Q.run_quality_checks(bad, config), "ajustement_borne", "XLV") == Q.FAIL


def test_detects_missing_column(clean, config):
    checks = Q.run_quality_checks(clean.drop(columns="adj_close"), config)
    assert status(checks, "schema", "*") == Q.FAIL


# --------------------------------------------------------------------------- #
# Snapshot gelé du dépôt (mode DEMO)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def frozen():
    cfg = D.load_config()
    try:
        return cfg, D.load_snapshot(config=cfg)  # vérifie aussi le SHA-256
    except FileNotFoundError:
        pytest.skip("aucun snapshot gelé : lancer `python -m src.data collect`")


def test_frozen_snapshot_passes_audit(frozen):
    cfg, df = frozen
    checks = Q.run_quality_checks(df, cfg)
    assert not (checks["statut"] == Q.FAIL).any(), checks[checks["statut"] == Q.FAIL]


def test_frozen_snapshot_covers_frozen_periods(frozen):
    cfg, df = frozen
    xlv = D.get_ticker(df, cfg.asset)
    assert xlv.index.min() <= pd.Timestamp("2018-01-02")
    assert xlv.index.max() >= pd.Timestamp(cfg.splits["test"]["end"])
    assert set(df["ticker"]) == {cfg.asset, cfg.benchmark}
    assert list(df.columns) == D.COLUMNS


def test_figure_and_report_are_generated(frozen, tmp_path):
    cfg, df = frozen
    fig = Q.plot_coherence(df, cfg, tmp_path / "fig.png")
    assert fig.exists() and fig.stat().st_size > 20_000
    snapshot = D.latest_snapshot(cfg.snapshot_dir)
    report = Q.write_report(df, Q.run_quality_checks(df, cfg), cfg, snapshot, fig, tmp_path / "audit.md")
    text = report.read_text(encoding="utf-8")
    assert "Verdict" in text and D.read_manifest(snapshot)["sha256"] in text
