"""Collecte, gel et chargement des données de marché (livrable D02).

Chaîne :  Yahoo Finance (yfinance)  ->  table longue normalisée  ->  snapshot CSV daté
          + manifeste JSON (provenance, conventions, empreinte SHA-256).

Deux modes d'utilisation :
  * collecte  : ``python -m src.data collect``  (réseau requis, relançable) ;
  * DEMO/import : ``load_snapshot()`` relit le dernier snapshot gelé, sans réseau,
    et vérifie son empreinte avant de le rendre.

Conventions (détaillées dans data/README.md) :
  * une ligne = une séance régulière du NYSE Arca pour un ticker ;
  * ``date`` = date de séance en heure de New York (America/New_York), sans heure ;
  * ``open/high/low/close`` ajustés des fractionnements, PAS des dividendes ;
  * ``adj_close`` ajusté des fractionnements ET des dividendes (rendement total).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "data.yaml"
MARKET_TZ = "America/New_York"

#: Schéma canonique du snapshot (ordre des colonnes du CSV).
COLUMNS = [
    "date", "ticker", "open", "high", "low", "close",
    "adj_close", "volume", "dividends", "stock_splits",
]
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]

_YF_RENAME = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Adj Close": "adj_close", "Volume": "volume",
    "Dividends": "dividends", "Stock Splits": "stock_splits",
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataConfig:
    asset: str
    benchmark: str
    source: str
    interval: str
    start: str
    end: Optional[str]
    snapshot_dir: Path
    splits: dict
    quality: dict

    @property
    def tickers(self) -> list[str]:
        return [self.asset, self.benchmark]


def load_config(path: Path | str = DEFAULT_CONFIG) -> DataConfig:
    """Lit configs/data.yaml (paramètres gelés)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    snap = Path(raw["snapshot_dir"])
    return DataConfig(
        asset=raw["asset"],
        benchmark=raw["benchmark"],
        source=raw["source"],
        interval=raw["interval"],
        start=str(raw["start"]),
        end=None if raw.get("end") is None else str(raw["end"]),
        snapshot_dir=snap if snap.is_absolute() else ROOT / snap,
        splits=raw["splits"],
        quality=raw.get("quality", {}),
    )


# --------------------------------------------------------------------------- #
# Collecte
# --------------------------------------------------------------------------- #
def last_closed_session_date(now_utc: Optional[datetime] = None) -> pd.Timestamp:
    """Dernière date de séance pouvant être considérée comme clôturée.

    Une barre journalière n'est définitive qu'après la clôture (16 h 00 NY). On
    attend 18 h 00 NY par prudence ; avant cette heure, la séance du jour est
    exclue pour ne jamais geler une barre intrajournalière partielle.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    now_ny = pd.Timestamp(now_utc).tz_convert(MARKET_TZ)
    day = now_ny.normalize().tz_localize(None)
    if now_ny.time() < time(18, 0):
        day -= pd.Timedelta(days=1)
    return day


def normalize_history(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Transforme la sortie de ``yfinance.Ticker.history`` au schéma canonique."""
    if raw is None or raw.empty:
        raise ValueError(f"Aucune donnée reçue pour {ticker}.")
    df = raw.rename(columns=_YF_RENAME).copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:  # horodatage 00:00 NY -> date de séance naïve
        idx = idx.tz_convert(MARKET_TZ).tz_localize(None)
    df["date"] = idx.normalize()
    df["ticker"] = ticker
    for col in ("dividends", "stock_splits"):
        if col not in df:
            df[col] = 0.0
    missing = [c for c in COLUMNS if c not in df]
    if missing:
        raise ValueError(f"Colonnes absentes pour {ticker} : {missing}")
    df = df[COLUMNS].reset_index(drop=True)
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].astype("float64")
    df[["dividends", "stock_splits"]] = df[["dividends", "stock_splits"]].fillna(0.0).astype("float64")
    df["volume"] = df["volume"].astype("Int64")
    return df


def download_ticker(ticker: str, start: str, end: Optional[str] = None,
                    interval: str = "1d") -> pd.DataFrame:
    """Télécharge l'OHLCV journalier d'un ticker depuis Yahoo Finance.

    ``auto_adjust=False`` conserve à la fois ``Close`` (ajusté des splits) et
    ``Adj Close`` (ajusté splits + dividendes) ; ``actions=True`` ajoute les
    dividendes et fractionnements, ce qui rend l'ajustement vérifiable.
    """
    import yfinance as yf  # import local : le mode DEMO ne dépend pas de yfinance

    end_excl = None
    if end is not None:  # borne ``end`` exclusive côté Yahoo
        end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.Ticker(ticker).history(
        start=start, end=end_excl, interval=interval,
        auto_adjust=False, actions=True, repair=False,
    )
    return normalize_history(raw, ticker)


def collect(config: DataConfig, now_utc: Optional[datetime] = None) -> pd.DataFrame:
    """Télécharge tous les tickers de la configuration (table longue triée)."""
    cutoff = last_closed_session_date(now_utc)
    if config.end is not None:
        cutoff = min(cutoff, pd.Timestamp(config.end))
    frames = []
    for ticker in config.tickers:
        df = download_ticker(ticker, config.start, config.end, config.interval)
        frames.append(df[df["date"] <= cutoff])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Snapshot gelé
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_snapshot(df: pd.DataFrame, config: DataConfig,
                  acquired_at_utc: Optional[datetime] = None) -> tuple[Path, Path]:
    """Écrit ``<ASSET>_<BENCH>_1d_<AAAAMMJJ>.csv`` et son manifeste ``.json``.

    La date du nom de fichier est la date d'acquisition (UTC). Un snapshot
    existant n'est jamais écrasé : une photographie gelée reste gelée.
    """
    acquired = acquired_at_utc or datetime.now(timezone.utc)
    config.snapshot_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config.asset}_{config.benchmark}_{config.interval}_{acquired:%Y%m%d}"
    csv_path = config.snapshot_dir / f"{stem}.csv"
    json_path = config.snapshot_dir / f"{stem}.json"
    if csv_path.exists():
        raise FileExistsError(
            f"{csv_path.name} existe déjà : un snapshot gelé n'est pas écrasé. "
            "Supprimez-le explicitement si vous voulez vraiment le remplacer."
        )

    out = df[COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(csv_path, index=False, float_format="%.6f", lineterminator="\n")

    try:
        import yfinance as yf
        lib_version = yf.__version__
    except Exception:  # pragma: no cover
        lib_version = "inconnue"

    per_ticker = {}
    for ticker, g in df.groupby("ticker"):
        per_ticker[ticker] = {
            "rows": int(len(g)),
            "first_date": pd.Timestamp(g["date"].min()).strftime("%Y-%m-%d"),
            "last_date": pd.Timestamp(g["date"].max()).strftime("%Y-%m-%d"),
            "n_dividends": int((g["dividends"] > 0).sum()),
            "n_splits": int((g["stock_splits"] > 0).sum()),
        }
    manifest = {
        "file": csv_path.name,
        "sha256": sha256_file(csv_path),
        "acquired_at_utc": acquired.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Yahoo Finance (API publique non officielle)",
        "library": f"yfinance {lib_version}",
        "call": "yfinance.Ticker(t).history(start, end, interval='1d', "
                "auto_adjust=False, actions=True, repair=False)",
        "instruments": {
            config.asset: "Health Care Select Sector SPDR Fund (NYSE Arca: XLV)"
            if config.asset == "XLV" else config.asset,
            config.benchmark: "SPDR S&P 500 ETF Trust (NYSE Arca: SPY)"
            if config.benchmark == "SPY" else config.benchmark,
        },
        "frequency": "journalière (1 séance régulière = 1 ligne)",
        "timezone": f"{MARKET_TZ} — 'date' = date de séance, clôture à 16:00 heure locale",
        "currency": "USD",
        "requested_start": config.start,
        "requested_end": config.end,
        "adjustment_convention": {
            "open/high/low/close": "ajustés des fractionnements, non ajustés des dividendes",
            "adj_close": "ajusté des fractionnements et des dividendes (rétro-ajustement Yahoo)",
            "volume": "nombre de parts, ajusté des fractionnements",
        },
        "columns": COLUMNS,
        "rows": int(len(df)),
        "tickers": per_ticker,
    }
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    return csv_path, json_path


def latest_snapshot(snapshot_dir: Path) -> Path:
    """Chemin du snapshot le plus récent (ordre lexicographique = chronologique)."""
    files = sorted(Path(snapshot_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"Aucun snapshot dans {snapshot_dir}. Lancez : python -m src.data collect"
        )
    return files[-1]


def read_manifest(csv_path: Path) -> dict:
    return json.loads(Path(csv_path).with_suffix(".json").read_text(encoding="utf-8"))


def load_snapshot(path: Optional[Path | str] = None, config: Optional[DataConfig] = None,
                  verify: bool = True) -> pd.DataFrame:
    """Charge un snapshot gelé (mode DEMO, sans réseau).

    Si ``verify`` est vrai, l'empreinte SHA-256 du CSV est comparée à celle du
    manifeste : toute altération du fichier lève une erreur.
    """
    if path is None:
        config = config or load_config()
        path = latest_snapshot(config.snapshot_dir)
    path = Path(path)
    if verify:
        expected = read_manifest(path)["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Empreinte SHA-256 invalide pour {path.name} : le snapshot a été modifié "
                f"(attendu {expected[:12]}…, obtenu {actual[:12]}…)."
            )
    df = pd.read_csv(path, parse_dates=["date"])
    df["volume"] = df["volume"].astype("Int64")
    return df[COLUMNS]


# --------------------------------------------------------------------------- #
# Vues utiles aux modules suivants
# --------------------------------------------------------------------------- #
def get_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Sous-table d'un ticker, indexée par date de séance."""
    out = df[df["ticker"] == ticker].drop(columns="ticker").set_index("date").sort_index()
    if out.empty:
        raise KeyError(f"Ticker absent du jeu de données : {ticker}")
    return out


def wide(df: pd.DataFrame, field: str = "adj_close") -> pd.DataFrame:
    """Table large date × ticker pour un champ (séances communes non forcées)."""
    return df.pivot(index="date", columns="ticker", values=field).sort_index()


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Rendement logarithmique d'une séance : r_t = ln(P_t / P_{t-1}).

    r_t n'est observable qu'APRÈS la clôture de t ; la première valeur est NaN.
    """
    return np.log(prices / prices.shift(1))


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Rendement simple d'une séance : R_t = P_t / P_{t-1} - 1."""
    return prices / prices.shift(1) - 1.0


def assign_split(dates: Iterable, splits: dict) -> pd.Series:
    """Étiquette chaque date par sa période figée (warmup/train/validation/test/paper)."""
    dates = pd.DatetimeIndex(dates)
    labels = pd.Series("hors_periode", index=dates, dtype="object")
    for name, bounds in splits.items():
        start = pd.Timestamp(bounds["start"])
        end = pd.Timestamp(bounds["end"]) if bounds.get("end") else pd.Timestamp.max
        labels[(dates >= start) & (dates <= end)] = name
    return labels


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Données de marché — livrable D02")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_collect = sub.add_parser("collect", help="télécharger et geler un nouveau snapshot daté")
    p_collect.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_show = sub.add_parser("show", help="recharger le dernier snapshot (mode DEMO) et le résumer")
    p_show.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.cmd == "collect":
        df = collect(config)
        csv_path, json_path = save_snapshot(df, config)
        print(f"Snapshot écrit : {csv_path.relative_to(ROOT)}  ({len(df)} lignes)")
        print(f"Manifeste      : {json_path.relative_to(ROOT)}")
    else:
        path = latest_snapshot(config.snapshot_dir)
        df = load_snapshot(path)
        print(f"Snapshot chargé et vérifié (SHA-256) : {path.relative_to(ROOT)}")
    summary = df.groupby("ticker")["date"].agg(["count", "min", "max"])
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
