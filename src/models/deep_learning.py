"""MLP compact en PyTorch (livrable D05).

Réseau volontairement petit : une couche cachée (tanh), un dropout, une sortie
linéaire. Entraînement sous protocole temporel (jamais de mélange passé/futur :
le moteur walk-forward de ``machine_learning.py`` ne lui donne que des séances
antérieures à la date de prévision).

  * cible centrée-réduite avec la moyenne et l'écart-type du jeu d'entraînement ;
  * perte quadratique, AdamW (décroissance des poids découplée), nombre
    d'époques FIXÉ à l'avance (aucun arrêt précoce sur des données futures) ;
  * moyenne de ``n_seeds`` réseaux initialisés différemment : réduit la variance
    d'initialisation, qui dépasse sinon le signal sur des rendements quotidiens ;
  * CPU, un seul fil de calcul, graines fixées : mêmes données => mêmes prévisions.

L'interface est celle de scikit-learn (fit / predict) pour passer dans un ``Pipeline``
avec le même ``StandardScaler`` que les autres modèles.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden: int = 16, weight_decay: float = 1e-2, dropout: float = 0.2,
                 epochs: int = 60, batch_size: int = 64, lr: float = 3e-3,
                 n_seeds: int = 3, seed: int = 0):
        self.hidden = hidden
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_seeds = n_seeds
        self.seed = seed

    def _net(self, n_in: int) -> nn.Module:
        return nn.Sequential(nn.Linear(n_in, int(self.hidden)), nn.Tanh(),
                             nn.Dropout(float(self.dropout)), nn.Linear(int(self.hidden), 1))

    def fit(self, X, y):
        torch.set_num_threads(1)
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        y = np.asarray(y, dtype=np.float32)
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        yt = torch.as_tensor((y - self.y_mean_) / self.y_std_).unsqueeze(1)
        n = X.shape[0]
        self.nets_ = []
        for k in range(int(self.n_seeds)):
            seed = int(self.seed) + k
            torch.manual_seed(seed)                       # initialisation et dropout
            gen = torch.Generator().manual_seed(seed)     # ordre des mini-lots
            net = self._net(X.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=float(self.lr),
                                    weight_decay=float(self.weight_decay))
            loss_fn = nn.MSELoss()
            net.train()
            for _ in range(int(self.epochs)):
                order = torch.randperm(n, generator=gen)
                for i in range(0, n, int(self.batch_size)):
                    b = order[i:i + int(self.batch_size)]
                    opt.zero_grad()
                    loss_fn(net(X[b]), yt[b]).backward()
                    opt.step()
            net.eval()
            self.nets_.append(net)
        return self

    def predict(self, X) -> np.ndarray:
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            out = np.mean([net(Xt).squeeze(1).numpy() for net in self.nets_], axis=0)
        return out * self.y_std_ + self.y_mean_

    @property
    def n_params_(self) -> int:
        """Nombre de poids d'UN réseau (le modèle final en moyenne ``n_seeds``)."""
        return int(sum(p.numel() for p in self.nets_[0].parameters()))
