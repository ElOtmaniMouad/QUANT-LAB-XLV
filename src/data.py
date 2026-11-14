"""Collecte des données journalières XLV et SPY depuis Yahoo Finance.

Usage:
    python -m src.data --start 2018-01-01 --end 2026-11-15

La date --end est exclusive chez yfinance. Le script enregistre les données
brutes, un dataset synchronisé et un manifeste daté pour la reproductibilité.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SNAPSHOT_DIR = ROOT / "data" / "snapshots"


def download_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Télécharge un ticker et retourne des colonnes OHLCV + Adj Close."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(
            f"Aucune donnée reçue pour {ticker}. Vérifiez Internet, les dates "
            "et la disponibilité de Yahoo Finance."
        )

    # Certaines versions de yfinance retournent des colonnes MultiIndex.
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Colonnes absentes pour {ticker}: {missing}. "
            f"Colonnes reçues: {list(df.columns)}"
        )

    df = df[required].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Collecte XLV et SPY")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=(pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    acquired_at = datetime.now(timezone.utc).isoformat()
    downloaded = {}

    for ticker in ("XLV", "SPY"):
        df = download_one(ticker, args.start, args.end)
        df.to_csv(RAW_DIR / f"{ticker.lower()}.csv", index=True)
        downloaded[ticker] = df
        print(f"{ticker}: {len(df)} lignes téléchargées ({df.index.min().date()} -> {df.index.max().date()})")

    # Synchronisation sur les dates communes afin de comparer les deux ETF.
    xlv = downloaded["XLV"].add_prefix("XLV_")
    spy = downloaded["SPY"].add_prefix("SPY_")
    combined = xlv.join(spy, how="inner").sort_index()
    combined.index.name = "Date"

    # Rendements simples calculés sur les prix ajustés.
    combined["XLV_Return"] = combined["XLV_Adj Close"].pct_change()
    combined["SPY_Return"] = combined["SPY_Adj Close"].pct_change()

    processed_path = PROCESSED_DIR / "xlv_spy_daily.csv"
    combined.to_csv(processed_path, index=True)

    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    snapshot_dir = SNAPSHOT_DIR / snapshot_date
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(snapshot_dir / "xlv_spy_daily.csv", index=True)

    manifest = {
        "project": "QUANT LAB - B05 XLV",
        "source": "Yahoo Finance via yfinance",
        "tickers": ["XLV", "SPY"],
        "start_requested": args.start,
        "end_requested_exclusive": args.end,
        "frequency": "daily",
        "timezone_index": "date de séance, sans heure",
        "price_adjustment": "auto_adjust=False; colonnes Close et Adj Close conservées",
        "downloaded_at_utc": acquired_at,
        "snapshot_date_local": snapshot_date,
        "rows_xlv": int(len(downloaded["XLV"])),
        "rows_spy": int(len(downloaded["SPY"])),
        "rows_common_dates": int(len(combined)),
        "first_common_date": str(combined.index.min().date()) if not combined.empty else None,
        "last_common_date": str(combined.index.max().date()) if not combined.empty else None,
        "files": [
            "data/raw/xlv.csv",
            "data/raw/spy.csv",
            "data/processed/xlv_spy_daily.csv",
            f"data/snapshots/{snapshot_date}/xlv_spy_daily.csv",
        ],
        "limitations": "yfinance/Yahoo Finance ne garantit pas un flux professionnel instantané; les dernières données disponibles peuvent être différées."
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDataset synchronisé : {processed_path}")
    print(f"Snapshot daté : {snapshot_dir}")
    print("Étape suivante : python -m src.audit")


if __name__ == "__main__":
    main()
