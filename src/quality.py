"""Audit qualité des données de marché (livrable D02).

Chaque contrôle renvoie une ligne {controle, ticker, statut, n, detail} :

  OK    le contrôle est satisfait ;
  INFO  fait notable mais légitime (ex. rendement extrême de mars 2020) ;
  WARN  anomalie mineure à documenter, n'empêche pas l'utilisation ;
  FAIL  anomalie bloquante : les données ne doivent pas être utilisées telles quelles.

Lancement :  ``python -m src.quality``  -> reports/d02_quality_checks.csv,
             reports/d02_data_audit.md, reports/figures/d02_coherence.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay

from src.data import (
    COLUMNS, PRICE_COLUMNS, ROOT, DataConfig, assign_split, get_ticker, latest_snapshot,
    load_config, load_snapshot, log_returns, read_manifest, wide,
)

OK, INFO, WARN, FAIL = "OK", "INFO", "WARN", "FAIL"
REPORTS = ROOT / "reports"


# --------------------------------------------------------------------------- #
# Calendrier NYSE (indépendant de la source de données)
# --------------------------------------------------------------------------- #
class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Jours fériés réguliers du NYSE (règles en vigueur sur 2017-2026)."""
    rules = [
        # 1er janvier : reporté au lundi s'il tombe un dimanche, non observé un samedi.
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


#: Fermetures exceptionnelles (deuils nationaux).
NYSE_SPECIAL_CLOSURES = pd.to_datetime([
    "2018-12-05",  # obsèques de George H. W. Bush
    "2025-01-09",  # obsèques de Jimmy Carter
])


def nyse_sessions(start, end) -> pd.DatetimeIndex:
    """Séances régulières attendues du NYSE entre ``start`` et ``end`` inclus."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    holidays = NYSEHolidayCalendar().holidays(start, end).union(NYSE_SPECIAL_CLOSURES)
    return pd.date_range(start, end, freq=CustomBusinessDay(holidays=holidays))


# --------------------------------------------------------------------------- #
# Contrôles élémentaires
# --------------------------------------------------------------------------- #
def _row(check: str, ticker: str, status: str, n: int = 0, detail: str = "") -> dict:
    return {"controle": check, "ticker": ticker, "statut": status, "n": int(n), "detail": detail}


def _dates(idx, limit: int = 8) -> str:
    idx = pd.DatetimeIndex(idx)
    txt = ", ".join(d.strftime("%Y-%m-%d") for d in idx[:limit])
    return txt + (f" … (+{len(idx) - limit})" if len(idx) > limit else "")


def robust_extremes(returns: pd.Series, k: float) -> pd.Series:
    """Rendements dont l'écart à la médiane dépasse ``k`` × MAD normalisé.

    Le MAD (× 1,4826) est un écart-type robuste : il n'est pas gonflé par les
    valeurs extrêmes que l'on cherche justement à repérer.
    """
    r = returns.dropna()
    med = r.median()
    mad = 1.4826 * (r - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return r.iloc[0:0]
    return r[(r - med).abs() > k * mad]


def implied_dividends(g: pd.DataFrame) -> pd.Series:
    """Dividende implicite déduit du facteur d'ajustement f_t = adj_close_t / close_t.

    Convention Yahoo : à la date ex-dividende t,  f_{t-1} / f_t = 1 - D_t / close_{t-1}
    d'où  D_t = close_{t-1} × (1 - f_{t-1} / f_t).  Hors date ex-dividende, D_t ≈ 0.
    """
    factor = g["adj_close"] / g["close"]
    return g["close"].shift(1) * (1.0 - factor.shift(1) / factor)


def check_ticker(df: pd.DataFrame, ticker: str, quality: dict) -> list[dict]:
    """Contrôles propres à un ticker (table longue en entrée)."""
    tol = float(quality.get("ohlc_tolerance", 1e-6))
    k = float(quality.get("extreme_return_mad_k", 8.0))
    max_stale = int(quality.get("max_stale_run", 2))
    div_tol = float(quality.get("dividend_rel_tolerance", 0.05))
    rows: list[dict] = []
    raw = df[df["ticker"] == ticker]

    # 1. Unicité et ordre des dates
    dup = raw[raw["date"].duplicated(keep=False)]
    rows.append(_row("dates_dupliquees", ticker, FAIL if len(dup) else OK, dup["date"].nunique(),
                     _dates(dup["date"].unique()) if len(dup) else "aucun doublon"))
    is_sorted = raw["date"].is_monotonic_increasing
    rows.append(_row("dates_triees", ticker, OK if is_sorted else FAIL, 0 if is_sorted else 1,
                     "ordre chronologique strict" if is_sorted else "dates non triées"))

    g = raw.drop_duplicates("date").set_index("date").sort_index()

    # 2. Valeurs manquantes
    na = g[PRICE_COLUMNS + ["volume"]].isna()
    n_na = int(na.any(axis=1).sum())
    rows.append(_row("valeurs_manquantes", ticker, FAIL if n_na else OK, n_na,
                     _dates(g.index[na.any(axis=1)]) if n_na else "aucune valeur manquante"))

    # 3. Prix strictement positifs
    bad = g[(g[PRICE_COLUMNS] <= 0).any(axis=1)]
    rows.append(_row("prix_positifs", ticker, FAIL if len(bad) else OK, len(bad),
                     _dates(bad.index) if len(bad) else "tous les prix > 0"))

    # 4. Cohérence interne OHLC : low <= min(open, close) <= max(open, close) <= high
    hi = g[["open", "close"]].max(axis=1)
    lo = g[["open", "close"]].min(axis=1)
    bad = g[(g["high"] + tol < hi) | (g["low"] - tol > lo) | (g["high"] + tol < g["low"])]
    rows.append(_row("coherence_ohlc", ticker, FAIL if len(bad) else OK, len(bad),
                     _dates(bad.index) if len(bad) else "low ≤ open, close ≤ high partout"))

    # 5. Volume
    neg = g[g["volume"].fillna(0) < 0]
    zero = g[g["volume"].fillna(0) == 0]
    status = FAIL if len(neg) else (WARN if len(zero) else OK)
    rows.append(_row("volume", ticker, status, len(neg) + len(zero),
                     f"volume négatif : {_dates(neg.index)}" if len(neg)
                     else f"volume nul : {_dates(zero.index)}" if len(zero)
                     else "volume > 0 à chaque séance"))

    # 6. Calendrier : séances manquantes / inattendues par rapport au NYSE
    if len(g):
        expected = nyse_sessions(g.index.min(), g.index.max())
        missing = expected.difference(g.index)
        extra = g.index.difference(expected)
        rows.append(_row("seances_manquantes", ticker, FAIL if len(missing) else OK, len(missing),
                         _dates(missing) if len(missing)
                         else f"{len(expected)} séances NYSE attendues, toutes présentes"))
        rows.append(_row("seances_inattendues", ticker, FAIL if len(extra) else OK, len(extra),
                         _dates(extra) if len(extra) else "aucune ligne un week-end ou jour férié"))

    # 7. Clôtures figées (prix répété plusieurs séances de suite)
    same = g["close"].diff().eq(0)
    run = same.groupby((~same).cumsum()).cumsum()
    stale = g.index[run >= max_stale]
    rows.append(_row("clotures_figees", ticker, WARN if len(stale) else OK, len(stale),
                     _dates(stale) if len(stale)
                     else f"aucune clôture répétée plus de {max_stale} fois de suite"))

    # 8. Rendements extrêmes (information, pas erreur : on les confronte aux événements connus)
    r = log_returns(g["adj_close"])
    ext = robust_extremes(r, k)
    detail = "; ".join(f"{d:%Y-%m-%d} ({v:+.2%})" for d, v in ext.items()) if len(ext) \
        else f"aucun rendement au-delà de {k:g} × MAD"
    rows.append(_row("rendements_extremes", ticker, INFO if len(ext) else OK, len(ext), detail))

    # 9. Fractionnements
    splits = g[g["stock_splits"] > 0]
    rows.append(_row("fractionnements", ticker, INFO if len(splits) else OK, len(splits),
                     "; ".join(f"{d:%Y-%m-%d} (×{v:g})" for d, v in splits["stock_splits"].items())
                     if len(splits) else "aucun fractionnement sur la période"))

    # 10. Convention d'ajustement : adj_close <= close, et facteur = 1 sur la période la plus récente
    factor = g["adj_close"] / g["close"]
    above = g.index[factor > 1 + 1e-6]
    rows.append(_row("ajustement_borne", ticker, FAIL if len(above) else OK, len(above),
                     _dates(above) if len(above)
                     else f"adj_close ≤ close partout ; facteur de {factor.iloc[0]:.4f} à {factor.iloc[-1]:.4f}"))

    # 11. Le facteur d'ajustement ne bouge qu'aux dates ex-dividende, du bon montant
    implied = implied_dividends(g)
    ex = g["dividends"] > 0
    ghost = g.index[(~ex) & (implied.abs() > 0.002 * g["close"].shift(1))].difference(g.index[:1])
    rows.append(_row("ajustement_hors_dividende", ticker, FAIL if len(ghost) else OK, len(ghost),
                     _dates(ghost) if len(ghost)
                     else "facteur d'ajustement constant hors dates ex-dividende"))
    ex_dates = g.index[ex].difference(g.index[:1])
    if len(ex_dates):
        rel = (implied[ex_dates] - g.loc[ex_dates, "dividends"]).abs() / g.loc[ex_dates, "dividends"]
        bad = rel[rel > div_tol]
        rows.append(_row("dividendes_coherents", ticker, WARN if len(bad) else OK, len(bad),
                         "; ".join(f"{d:%Y-%m-%d} (écart {v:.1%})" for d, v in bad.items()) if len(bad)
                         else f"{len(ex_dates)} dividendes, tous retrouvés dans adj_close "
                              f"(écart max {rel.max():.2%})"))
    else:
        rows.append(_row("dividendes_coherents", ticker, WARN, 0, "aucun dividende déclaré"))
    return rows


def check_schema(df: pd.DataFrame) -> list[dict]:
    missing = [c for c in COLUMNS if c not in df.columns]
    rows = [_row("schema", "*", FAIL if missing else OK, len(missing),
                 f"colonnes absentes : {missing}" if missing else f"{len(COLUMNS)} colonnes attendues présentes")]
    return rows


def check_pair(df: pd.DataFrame, config: DataConfig) -> list[dict]:
    """Contrôles croisés actif / benchmark et couverture des périodes figées."""
    rows: list[dict] = []
    # Les doublons sont déjà signalés par ticker ; on les neutralise ici pour que
    # les contrôles croisés restent calculables au lieu de planter.
    df = df.drop_duplicates(["ticker", "date"])
    a = get_ticker(df, config.asset).index
    b = get_ticker(df, config.benchmark).index
    only_a, only_b = a.difference(b), b.difference(a)
    n = len(only_a) + len(only_b)
    rows.append(_row("synchronisation", f"{config.asset}/{config.benchmark}", FAIL if n else OK, n,
                     f"dates propres à {config.asset} : {_dates(only_a)} ; à {config.benchmark} : {_dates(only_b)}"
                     if n else f"{len(a)} séances strictement communes"))

    # Couverture des périodes : première et dernière séance attendue de chaque période
    labels = assign_split(a, config.splits)
    for name, bounds in config.splits.items():
        if not bounds.get("end"):
            n_obs = int((labels == name).sum())
            rows.append(_row(f"couverture_{name}", config.asset, INFO, n_obs,
                             f"{n_obs} séances depuis le {bounds['start']} (période ouverte)"))
            continue
        expected = nyse_sessions(bounds["start"], bounds["end"])
        got = a[labels.values == name]
        lacking = expected.difference(got)
        rows.append(_row(f"couverture_{name}", config.asset, FAIL if len(lacking) else OK, len(lacking),
                         f"séances absentes : {_dates(lacking)}" if len(lacking)
                         else f"{len(got)} séances, du {got.min():%Y-%m-%d} au {got.max():%Y-%m-%d}"))

    # Vraisemblance économique : un ETF sectoriel du S&P 500 reste corrélé à SPY
    rets = log_returns(wide(df, "adj_close")).dropna()
    corr = float(rets[config.asset].corr(rets[config.benchmark]))
    rows.append(_row("correlation_benchmark", f"{config.asset}/{config.benchmark}",
                     OK if corr > 0.3 else WARN, 0,
                     f"corrélation des rendements journaliers = {corr:.3f}"))
    return rows


def run_quality_checks(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Exécute tous les contrôles et renvoie le tableau de résultats."""
    rows = check_schema(df)
    if rows[0]["statut"] == FAIL:
        return pd.DataFrame(rows)
    for ticker in config.tickers:
        rows += check_ticker(df, ticker, config.quality)
    rows += check_pair(df, config)
    return pd.DataFrame(rows, columns=["controle", "ticker", "statut", "n", "detail"])


# --------------------------------------------------------------------------- #
# Graphique de cohérence
# --------------------------------------------------------------------------- #
def plot_coherence(df: pd.DataFrame, config: DataConfig, path: Path) -> Path:
    """Planche de six graphiques permettant de juger d'un coup d'œil la cohérence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    asset, bench = config.asset, config.benchmark
    c_asset, c_bench, c_muted = "#1f6f8b", "#c0504d", "#7f7f7f"
    g = get_ticker(df, asset)
    adj = wide(df, "adj_close")
    rets = log_returns(adj)
    k = float(config.quality.get("extreme_return_mad_k", 8.0))

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    fig.suptitle(f"D02 — Cohérence des données {asset} (benchmark {bench}) · "
                 f"{g.index.min():%d/%m/%Y} → {g.index.max():%d/%m/%Y} · source Yahoo Finance",
                 fontsize=13, fontweight="bold")

    # (a) Rendement total base 100 + périodes figées
    ax = axes[0, 0]
    base = adj / adj.iloc[0] * 100
    ax.plot(base.index, base[asset], color=c_asset, lw=1.4, label=f"{asset} (adj_close)")
    ax.plot(base.index, base[bench], color=c_bench, lw=1.1, label=f"{bench} (adj_close)")
    shades = {"train": "#e8f1f5", "validation": "#fdf0d5", "test": "#f8dcd9", "paper": "#e3f0e0"}
    for name, bounds in config.splits.items():
        if name in shades:
            end = pd.Timestamp(bounds["end"]) if bounds.get("end") else g.index.max()
            ax.axvspan(pd.Timestamp(bounds["start"]), end, color=shades[name], zorder=0)
            ax.text(pd.Timestamp(bounds["start"]), 0.02, f" {name}", fontsize=8, color=c_muted,
                    transform=ax.get_xaxis_transform(), va="bottom")
    ax.set_title("(a) Rendement total, base 100 au premier jour — périodes figées en fond")
    ax.set_ylabel("Indice (base 100)")
    ax.legend(loc="upper left", frameon=False)

    # (b) Close vs adj_close : l'écart = dividendes réinvestis
    ax = axes[0, 1]
    ax.plot(g.index, g["close"], color=c_asset, lw=1.3, label="close (hors dividendes)")
    ax.plot(g.index, g["adj_close"], color="#e39e21", lw=1.1, label="adj_close (dividendes inclus)")
    ax.fill_between(g.index, g["low"], g["high"], color=c_asset, alpha=0.15, lw=0,
                    label="fourchette low–high")
    ax.set_title(f"(b) {asset} : clôture brute et clôture ajustée")
    ax.set_ylabel("Prix (USD)")
    ax.legend(loc="upper left", frameon=False)

    # (c) Rendements journaliers et valeurs extrêmes signalées
    ax = axes[1, 0]
    r = rets[asset] * 100
    ext = robust_extremes(rets[asset], k) * 100
    ax.plot(r.index, r, color=c_asset, lw=0.6)
    ax.scatter(ext.index, ext, color=c_bench, s=22, zorder=3,
               label=f"|écart à la médiane| > {k:g} × MAD ({len(ext)})")
    ax.axhline(0, color=c_muted, lw=0.5)
    ax.set_title(f"(c) {asset} : rendement logarithmique journalier")
    ax.set_ylabel("Rendement (%)")
    ax.legend(loc="upper right", frameon=False)

    # (d) Volume
    ax = axes[1, 1]
    vol = g["volume"].astype("float64") / 1e6
    ax.bar(vol.index, vol, width=1.0, color=c_asset, alpha=0.35, label="volume journalier")
    ax.plot(vol.index, vol.rolling(21).mean(), color=c_asset, lw=1.3, label="moyenne mobile 21 séances")
    ax.set_title(f"(d) {asset} : volume échangé")
    ax.set_ylabel("Millions de parts")
    ax.legend(loc="upper left", frameon=False)

    # (e) Facteur d'ajustement : marches d'escalier aux dates ex-dividende uniquement
    ax = axes[2, 0]
    factor = g["adj_close"] / g["close"]
    ax.step(factor.index, factor, where="post", color=c_asset, lw=1.3, label="adj_close / close")
    ex = g.index[g["dividends"] > 0]
    ax.scatter(ex, factor[ex], color="#e39e21", s=18, zorder=3,
               label=f"dates ex-dividende ({len(ex)})")
    ax.set_title(f"(e) {asset} : facteur d'ajustement (doit varier seulement aux dividendes)")
    ax.set_ylabel("Facteur (sans unité)")
    ax.legend(loc="upper left", frameon=False)

    # (f) Corrélation glissante avec le benchmark
    ax = axes[2, 1]
    corr = rets[asset].rolling(63).corr(rets[bench])
    ax.plot(corr.index, corr, color=c_asset, lw=1.2)
    ax.axhline(float(rets[asset].corr(rets[bench])), color=c_muted, lw=0.8, ls="--",
               label="corrélation sur toute la période")
    ax.set_ylim(-0.4, 1.05)
    ax.set_title(f"(f) Corrélation glissante {asset}/{bench} (63 séances)")
    ax.set_ylabel("Corrélation (sans unité)")
    ax.legend(loc="lower left", frameon=False)

    for ax in axes.ravel():
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes[2]:
        ax.set_xlabel("Date de séance (heure de New York)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Rapport Markdown
# --------------------------------------------------------------------------- #
def _md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |")
    return "\n".join(lines)


def descriptive_stats(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Statistiques des rendements log journaliers (adj_close), par ticker et période."""
    rets = log_returns(wide(df, "adj_close"))
    labels = assign_split(rets.index, config.splits)
    out = []
    for ticker in config.tickers:
        for name in list(config.splits) + ["ensemble"]:
            r = rets[ticker].dropna() if name == "ensemble" else rets.loc[labels.values == name, ticker].dropna()
            if r.empty:
                continue
            out.append({
                "ticker": ticker, "periode": name, "seances": len(r),
                "moyenne_%": f"{r.mean() * 100:.3f}", "ecart_type_%": f"{r.std() * 100:.3f}",
                "min_%": f"{r.min() * 100:.2f}", "max_%": f"{r.max() * 100:.2f}",
                "asymetrie": f"{r.skew():.2f}", "kurtosis_exces": f"{r.kurt():.2f}",
                "vol_annualisee_%": f"{r.std() * np.sqrt(252) * 100:.1f}",
            })
    return pd.DataFrame(out)


def write_report(df: pd.DataFrame, checks: pd.DataFrame, config: DataConfig,
                 snapshot: Path, figure: Path, path: Path) -> Path:
    manifest = read_manifest(snapshot)
    counts = checks["statut"].value_counts().reindex([OK, INFO, WARN, FAIL], fill_value=0)
    verdict = ("**Données utilisables** : aucun contrôle bloquant." if counts[FAIL] == 0
               else "**Données NON utilisables en l'état** : au moins un contrôle bloquant.")
    notable = checks[checks["statut"] != OK]
    lines = [
        f"# D02 — Rapport d'audit des données ({config.asset} / {config.benchmark})",
        "",
        "> Fichier généré par `python -m src.quality`. Ne pas modifier à la main : relancer la commande.",
        "",
        "## 1. Snapshot audité",
        "",
        f"- Fichier : `{snapshot.relative_to(ROOT)}`",
        f"- Empreinte SHA-256 : `{manifest['sha256']}`",
        f"- Date d'acquisition : {manifest['acquired_at_utc']} (UTC)",
        f"- Source : {manifest['source']} via {manifest['library']}",
        f"- Lignes : {manifest['rows']}",
        "",
        _md_table(pd.DataFrame([{"ticker": t, **v} for t, v in manifest["tickers"].items()])),
        "",
        "## 2. Verdict",
        "",
        f"{verdict}",
        "",
        f"{len(checks)} contrôles : {counts[OK]} OK · {counts[INFO]} INFO · "
        f"{counts[WARN]} WARN · {counts[FAIL]} FAIL.",
        "",
        "## 3. Anomalies et faits notables identifiés",
        "",
        _md_table(notable) if len(notable) else "Aucune anomalie ni fait notable.",
        "",
        "## 4. Détail de tous les contrôles",
        "",
        _md_table(checks),
        "",
        "## 5. Statistiques descriptives des rendements",
        "",
        "Rendement logarithmique journalier calculé sur `adj_close` (dividendes inclus).",
        "",
        _md_table(descriptive_stats(df, config)),
        "",
        "## 6. Graphique de cohérence",
        "",
        f"![Cohérence des données]({figure.relative_to(path.parent).as_posix()})",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit qualité des données — livrable D02")
    parser.add_argument("--config", default=None)
    parser.add_argument("--snapshot", default=None, help="CSV à auditer (défaut : le plus récent)")
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else load_config()
    snapshot = Path(args.snapshot) if args.snapshot else latest_snapshot(config.snapshot_dir)
    df = load_snapshot(snapshot)

    checks = run_quality_checks(df, config)
    REPORTS.mkdir(exist_ok=True)
    checks.to_csv(REPORTS / "d02_quality_checks.csv", index=False, lineterminator="\n")
    figure = plot_coherence(df, config, REPORTS / "figures" / "d02_coherence.png")
    report = write_report(df, checks, config, snapshot.resolve(), figure, REPORTS / "d02_data_audit.md")

    print(checks.to_string(index=False, max_colwidth=90))
    print(f"\nRapport   : {report.relative_to(ROOT)}")
    print(f"Graphique : {figure.relative_to(ROOT)}")
    n_fail = int((checks["statut"] == FAIL).sum())
    print("Verdict   : " + ("données utilisables." if n_fail == 0 else f"{n_fail} contrôle(s) bloquant(s)."))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
