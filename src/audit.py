"""Contrôles qualité, rapport d'audit et graphique de cohérence XLV/SPY.

Usage:
    python -m src.audit
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "xlv_spy_daily.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def add_check(rows, name, status, details):
    rows.append({"controle": name, "statut": status, "details": str(details)})


def audit() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} introuvable. Lancez d'abord: python -m src.data"
        )

    df = pd.read_csv(INPUT, parse_dates=["Date"]).set_index("Date").sort_index()
    checks = []

    add_check(checks, "Fichier lisible", "OK", f"{len(df)} lignes chargées")
    add_check(checks, "Colonnes attendues", 
              "OK" if len(df.columns) > 0 else "ANOMALIE",
              ", ".join(df.columns))

    duplicated = int(df.index.duplicated().sum())
    add_check(checks, "Dates dupliquées", "OK" if duplicated == 0 else "ANOMALIE", duplicated)

    ordered = bool(df.index.is_monotonic_increasing)
    add_check(checks, "Dates triées", "OK" if ordered else "ANOMALIE", ordered)

    # La première ligne de rendement est normalement NaN: il n'existe pas
    # de séance précédente dans le fichier pour calculer le rendement.
    return_cols = [c for c in ("XLV_Return", "SPY_Return") if c in df.columns]
    expected_return_na = 0
    if len(df) > 0 and return_cols:
        expected_return_na = int(df.iloc[0][return_cols].isna().sum())

    missing_total = int(df.isna().sum().sum())
    missing_cols = df.isna().sum()
    unexpected_missing = df.copy()
    if len(unexpected_missing) > 0:
        for c in return_cols:
            unexpected_missing.loc[unexpected_missing.index[0], c] = 0
    unexpected_missing_total = int(unexpected_missing.isna().sum().sum())
    add_check(
        checks,
        "Valeurs manquantes (hors 1er rendement attendu)",
        "OK" if unexpected_missing_total == 0 else "À EXAMINER",
        f"Manquantes totales={missing_total}; manquantes attendues sur le 1er rendement={expected_return_na}; "
        f"manquantes inattendues={unexpected_missing_total}; par colonne={missing_cols[missing_cols > 0].to_dict()}"
    )

    for ticker in ("XLV", "SPY"):
        cols = [f"{ticker}_{c}" for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume")]
        absent = [c for c in cols if c not in df.columns]
        if absent:
            add_check(checks, f"{ticker} colonnes OHLCV", "ANOMALIE", f"Absentes: {absent}")
            continue

        price_cols = [f"{ticker}_{c}" for c in ("Open", "High", "Low", "Close", "Adj Close")]
        nonpositive = int((df[price_cols] <= 0).sum().sum())
        add_check(checks, f"{ticker} prix positifs", "OK" if nonpositive == 0 else "ANOMALIE", nonpositive)

        invalid_high_low = int((df[f"{ticker}_High"] < df[f"{ticker}_Low"]).sum())
        add_check(checks, f"{ticker} High >= Low", "OK" if invalid_high_low == 0 else "ANOMALIE", invalid_high_low)

        invalid_high = int((
            (df[f"{ticker}_High"] < df[f"{ticker}_Open"]) |
            (df[f"{ticker}_High"] < df[f"{ticker}_Close"])
        ).sum())
        add_check(checks, f"{ticker} High >= Open et Close",
                  "OK" if invalid_high == 0 else "ANOMALIE", invalid_high)

        invalid_low = int((
            (df[f"{ticker}_Low"] > df[f"{ticker}_Open"]) |
            (df[f"{ticker}_Low"] > df[f"{ticker}_Close"])
        ).sum())
        add_check(checks, f"{ticker} Low <= Open et Close",
                  "OK" if invalid_low == 0 else "ANOMALIE", invalid_low)

        negative_volume = int((df[f"{ticker}_Volume"] < 0).sum())
        add_check(checks, f"{ticker} volume non négatif",
                  "OK" if negative_volume == 0 else "ANOMALIE", negative_volume)

        returns_col = f"{ticker}_Return"
        if returns_col in df.columns:
            extreme = int((df[returns_col].abs() > 0.20).sum())
            add_check(checks, f"{ticker} rendements absolus > 20%",
                      "À EXAMINER" if extreme else "OK",
                      f"{extreme} observation(s); un mouvement extrême n'est pas automatiquement une erreur")

    # Vérification de la synchronisation: le dataset final ne contient que les dates communes.
    date_na = int(df.index.isna().sum())
    add_check(checks, "Dates valides", "OK" if date_na == 0 else "ANOMALIE", date_na)

    report = pd.DataFrame(checks)
    report.to_csv(REPORT_DIR / "quality_checks.csv", index=False)

    summary = {
        "input_file": str(INPUT.relative_to(ROOT)),
        "rows": int(len(df)),
        "first_date": str(df.index.min().date()) if len(df) else None,
        "last_date": str(df.index.max().date()) if len(df) else None,
        "columns": list(df.columns),
        "missing_values_total": missing_total,
        "duplicate_dates": duplicated,
        "anomalies": report[report["statut"] == "ANOMALIE"].to_dict(orient="records"),
        "items_to_review": report[report["statut"] == "À EXAMINER"].to_dict(orient="records"),
    }
    (REPORT_DIR / "quality_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Graphique de cohérence: évolution des prix ajustés, base 100 à la première date commune.
    plt.figure(figsize=(11, 6))
    for ticker in ("XLV", "SPY"):
        col = f"{ticker}_Adj Close"
        if col in df.columns and df[col].notna().any():
            series = df[col].dropna()
            normalized = 100 * series / series.iloc[0]
            plt.plot(normalized.index, normalized.values, label=ticker)
    plt.title("Cohérence des séries XLV et SPY — prix ajustés, base 100")
    plt.xlabel("Date")
    plt.ylabel("Indice (première observation = 100)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "coherence_xlv_spy.png", dpi=160)
    plt.close()

    print(report.to_string(index=False))
    print(f"\nRapport : {REPORT_DIR / 'quality_report.json'}")
    print(f"Graphique : {REPORT_DIR / 'coherence_xlv_spy.png'}")


if __name__ == "__main__":
    audit()
