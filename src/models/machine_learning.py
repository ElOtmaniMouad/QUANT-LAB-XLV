"""Modèles de Machine Learning et moteur d'évaluation walk-forward (livrable D05).

Quatre familles, toutes précédées du MÊME ``StandardScaler`` ajusté sur les seules
données d'entraînement de chaque ré-estimation :

  * ``Ridge``             régression linéaire régularisée (L2) ;
  * ``RandomForest``      forêt aléatoire, arbres peu profonds ;
  * ``GradientBoosting``  boosting d'arbres très peu profonds, pas d'apprentissage faible ;
  * ``MLP``               réseau de neurones PyTorch compact (``deep_learning.py``).

Protocole (identique à D03) : fenêtre croissante depuis le début du train ; les
paramètres sont ré-estimés toutes les ``refit_every`` séances avec les SEULES lignes
antérieures au bloc ; la prévision de la séance t n'utilise que des lignes < t pour
l'apprentissage et la ligne t (variables connues à la clôture de t-1) pour prédire.

Unités : la cible est multipliée par 100 (en %) pendant l'estimation, pour des
échelles raisonnables ; les prévisions renvoyées sont reconverties en rendement log.
"""
from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.deep_learning import TorchMLPRegressor

SCALE = 100.0
FAMILIES = ("Ridge", "RandomForest", "GradientBoosting", "MLP")
_SEED_OFFSET = {"Ridge": 0, "RandomForest": 1000, "GradientBoosting": 2000, "MLP": 3000}


# --------------------------------------------------------------------------- #
# Spécifications et grilles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """Une famille et un jeu complet d'hyperparamètres."""
    family: str
    params: tuple                       # ((nom, valeur), …) trié : hachable et reproductible
    varied: tuple = ()                  # noms des hyperparamètres explorés (pour l'étiquette)

    @property
    def p(self) -> dict:
        return dict(self.params)

    @property
    def label(self) -> str:
        if not self.varied:
            return self.family
        return f"{self.family}(" + ", ".join(f"{k}={self.p[k]}" for k in self.varied) + ")"


def expand_grid(family: str, cfg: dict) -> list[ModelSpec]:
    """Produit cartésien des hyperparamètres listés ; les scalaires restent fixes."""
    if family not in FAMILIES:
        raise ValueError(f"Famille inconnue : {family}")
    axes = {k: v for k, v in cfg.items() if isinstance(v, list)}
    fixed = {k: v for k, v in cfg.items() if not isinstance(v, list)}
    out = []
    for combo in itertools.product(*axes.values()):
        params = {**fixed, **dict(zip(axes.keys(), combo))}
        out.append(ModelSpec(family, tuple(sorted(params.items())), tuple(axes.keys())))
    return out


def spec_to_dict(spec: ModelSpec) -> dict:
    return {"family": spec.family, "label": spec.label, "params": spec.p, "varied": list(spec.varied)}


def spec_from_dict(d: dict) -> ModelSpec:
    return ModelSpec(d["family"], tuple(sorted(d["params"].items())), tuple(d["varied"]))


def make_estimator(spec: ModelSpec, seed: int = 0) -> Pipeline:
    """Estimateur scikit-learn (standardisation + modèle) pour une spécification."""
    p, s = spec.p, int(seed) + _SEED_OFFSET[spec.family]
    if spec.family == "Ridge":
        model = Ridge(alpha=float(p["alpha"]))
    elif spec.family == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]),
            min_samples_leaf=int(p["min_samples_leaf"]), max_features=float(p["max_features"]),
            random_state=s, n_jobs=-1)
    elif spec.family == "GradientBoosting":
        model = GradientBoostingRegressor(
            n_estimators=int(p["n_estimators"]), learning_rate=float(p["learning_rate"]),
            max_depth=int(p["max_depth"]), subsample=float(p["subsample"]),
            min_samples_leaf=int(p["min_samples_leaf"]), random_state=s)
    else:
        model = TorchMLPRegressor(
            hidden=int(p["hidden"]), weight_decay=float(p["weight_decay"]),
            dropout=float(p["dropout"]), epochs=int(p["epochs"]), batch_size=int(p["batch_size"]),
            lr=float(p["lr"]), n_seeds=int(p["n_seeds"]), seed=s)
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardResult:
    predictions: pd.Series
    n_refits: int
    fit_seconds: float
    last_estimator: Optional[Pipeline] = None
    train_sizes: list = field(default_factory=list)


def walk_forward(factory, X: pd.DataFrame, y: pd.Series, start, end,
                 refit_every: int = 21, min_history: int = 100) -> WalkForwardResult:
    """Prévisions à une séance de ``y`` (rendement log) pour les dates de [start, end].

    ``factory`` : fonction sans argument qui renvoie un estimateur neuf (fit / predict).
    ``X``, ``y`` : lignes alignées, commençant au début du train ; seules les lignes
    STRICTEMENT antérieures au début d'un bloc servent à l'estimer.
    """
    idx = X.index
    pos = np.flatnonzero((idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end)))
    if len(pos) == 0:
        raise ValueError("Aucune séance à prévoir dans l'intervalle demandé.")
    first, last = int(pos[0]), int(pos[-1])
    if first < min_history:
        raise ValueError("Historique d'estimation trop court avant la première prévision.")

    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float) * SCALE
    preds = np.full(len(yv), np.nan)
    elapsed, n_refits, sizes, est = 0.0, 0, [], None
    for b in range(first, last + 1, refit_every):
        stop = min(b + refit_every, last + 1)
        est = factory()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(Xv[:b], yv[:b])                 # lignes < b uniquement
        elapsed += time.perf_counter() - t0
        n_refits += 1
        sizes.append(b)
        preds[b:stop] = np.asarray(est.predict(Xv[b:stop])) / SCALE
    out = pd.Series(preds, index=idx).iloc[first:last + 1]
    return WalkForwardResult(out, n_refits, elapsed, est, sizes)


# --------------------------------------------------------------------------- #
# Sélection des hyperparamètres (VALIDATION uniquement)
# --------------------------------------------------------------------------- #
def validate_grid(specs: list[ModelSpec], X: pd.DataFrame, y: pd.Series, start, end,
                  refit_every: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """RMSE walk-forward de chaque spécification sur [start, end]. Renvoie (tableau, prévisions)."""
    rows, preds = [], {}
    for spec in specs:
        res = walk_forward(lambda sp=spec: make_estimator(sp, seed), X, y, start, end, refit_every)
        actual = y.loc[res.predictions.index]
        rmse = float(np.sqrt(np.mean((actual - res.predictions) ** 2)))
        rows.append({"famille": spec.family, "candidat": spec.label,
                     "RMSE_val_%": 100 * rmse, "secondes": res.fit_seconds, "refits": res.n_refits})
        preds[spec.label] = res.predictions
    return pd.DataFrame(rows), preds


def pick_best(table: pd.DataFrame, specs: list[ModelSpec]) -> ModelSpec:
    """Spécification de RMSE de validation minimale (égalité : première de la grille)."""
    best = table.loc[table["RMSE_val_%"].idxmin(), "candidat"]
    return next(s for s in specs if s.label == best)


# --------------------------------------------------------------------------- #
# Description des modèles ajustés (parcimonie, interprétation)
# --------------------------------------------------------------------------- #
def n_parameters(est: Pipeline) -> int:
    """Taille du modèle : coefficients (Ridge), nœuds d'arbres (forêt, boosting), poids (MLP)."""
    m = est.named_steps["model"]
    if isinstance(m, Ridge):
        return int(m.coef_.size + 1)
    if isinstance(m, RandomForestRegressor):
        return int(sum(t.tree_.node_count for t in m.estimators_))
    if isinstance(m, GradientBoostingRegressor):
        return int(sum(t.tree_.node_count for t in m.estimators_.ravel()))
    return int(m.n_params_)


def feature_importance(est: Pipeline, names: list[str]) -> Optional[pd.Series]:
    """Coefficient standardisé (Ridge) ou importance par impureté (arbres) ; None pour le MLP."""
    m = est.named_steps["model"]
    if isinstance(m, Ridge):
        return pd.Series(m.coef_, index=names)
    if isinstance(m, (RandomForestRegressor, GradientBoostingRegressor)):
        return pd.Series(m.feature_importances_, index=names)
    return None
