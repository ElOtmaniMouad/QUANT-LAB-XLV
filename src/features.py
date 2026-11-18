"""Construction du jeu de modélisation (livrable D03).

Une ligne = une séance t. Toutes les colonnes d'une ligne sont connues à la
clôture de t, SAUF ``target`` qui est la valeur à prévoir pour cette même
ligne à partir de l'information disponible à la clôture de t-1 :

  * ``y``          rendement log de XLV de la séance t (cible, prévu en t-1) ;
  * ``x_spy_lag1`` rendement log de SPY de la séance t-1 (connu en t-1) ;
  * ``price``      prix ajusté de XLV à la clôture de t ;
  * ``price_prev`` prix ajusté de XLV à la clôture de t-1 ;
  * ``split``      période figée (warmup/train/validation/test/paper).

Convention : une prévision « pour la ligne t » utilise exclusivement les lignes
strictement antérieures (y_{≤t-1}, x_{≤t}) — x_t = r^SPY_{t-1} est donc licite.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.data import ROOT, DataConfig, assign_split, load_config, log_returns, wide

DEFAULT_MODEL_CONFIG = ROOT / "configs" / "models.yaml"


@dataclass(frozen=True)
class ModelConfig:
    raw: dict

    def __getitem__(self, key):
        return self.raw[key]

    @property
    def subperiods(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        return {k: (pd.Timestamp(v["start"]), pd.Timestamp(v["end"]))
                for k, v in self.raw["test_subperiods"].items()}


def load_model_config(path: Path | str = DEFAULT_MODEL_CONFIG) -> ModelConfig:
    return ModelConfig(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def build_dataset(df: pd.DataFrame, data_cfg: Optional[DataConfig] = None,
                  model_cfg: Optional[ModelConfig] = None) -> pd.DataFrame:
    """Table de modélisation indexée par date de séance (séances communes XLV/SPY)."""
    data_cfg = data_cfg or load_config()
    model_cfg = model_cfg or load_model_config()
    tgt, ex = model_cfg["target"], model_cfg["exog"]
    if tgt["kind"] != "log_return" or int(tgt["horizon"]) != 1:
        raise NotImplementedError("D03 ne traite que le rendement log à 1 séance.")

    prices = wide(df, tgt["field"]).dropna()          # séances communes aux deux tickers
    rets = log_returns(prices)
    out = pd.DataFrame({
        "y": rets[tgt["ticker"]],
        "x_spy_lag1": rets[ex["ticker"]].shift(int(ex["lag"])),
        "price": prices[tgt["ticker"]],
        "price_prev": prices[tgt["ticker"]].shift(1),
    })
    out["split"] = assign_split(out.index, data_cfg.splits).values
    out.index.name = "date"
    return out.dropna(subset=["y", "x_spy_lag1", "price_prev"])


def modelling_sample(ds: pd.DataFrame) -> pd.DataFrame:
    """Lignes utilisables pour l'estimation et l'évaluation (2017 = amorçage exclu)."""
    return ds[ds["split"].isin(["train", "validation", "test", "paper"])]
