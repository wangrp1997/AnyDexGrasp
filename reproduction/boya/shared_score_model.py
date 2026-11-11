"""Small conditional Boya classifiers; no simulation or official model changes."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

CONFIG = {
    "epochs": 300,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "optimizer": "Adam",
    "batch": "full fit set, equal total weight per fit object",
    "hidden_dim": 64,
    "mlp_seeds": [20261111, 20261112, 20261113],
    "linear_seed": 20261111,
    "linear_epochs": 1000,
    "linear_learning_rate": 0.01,
    "checkpoint_selection": "fixed final epoch; no early stopping or seed selection",
}
FAMILIES = ("condition_linear", "shared_linear", "shared_mlp64")


def family_config(family):
    config = dict(CONFIG)
    if family != "shared_mlp64":
        config.update(epochs=CONFIG["linear_epochs"],
                      learning_rate=CONFIG["linear_learning_rate"])
    return config


class SharedScorer(nn.Module):
    def __init__(self, family):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError("Unknown fixed model family")
        self.family = family
        dim = 16 if family == "condition_linear" else 496
        if family == "shared_mlp64":
            self.layers = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1))
        else:
            self.layers = nn.Linear(dim, 1)
            # The linear objective is convex. A fixed zero initialization avoids
            # spending the seed budget on repeated identical linear fits.
            nn.init.zeros_(self.layers.weight)
            nn.init.zeros_(self.layers.bias)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != 496:
            raise ValueError("Expected canonical480 plus 16 pre-execution conditions")
        if self.family == "condition_linear":
            x = x[:, 480:]
        return self.layers(x).flatten()


def object_weights(object_ids):
    groups = np.asarray(object_ids)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("Nonempty fit groups required")
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return (weights / weights.sum()).astype(np.float32)


def fit_model(x, labels, object_ids, family, seed, config=None):
    """Only fit arrays enter this function; check labels cannot affect updates."""
    config = dict(family_config(family) if config is None else config)
    x = np.asarray(x, np.float32)
    labels = np.asarray(labels)
    if (x.shape != (len(labels), 496) or not len(labels)
            or len(object_ids) != len(labels) or not np.isfinite(x).all()
            or not np.isin(labels, [0, 1]).all()):
        raise ValueError("Invalid finite fit data")
    torch.manual_seed(seed)
    model = SharedScorer(family)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    tx = torch.from_numpy(x)
    ty = torch.tensor(labels, dtype=torch.float32)
    tw = torch.from_numpy(object_weights(object_ids))
    history = []
    model.train()
    for epoch in range(config["epochs"]):
        optimizer.zero_grad(set_to_none=True)
        loss = (nn.functional.binary_cross_entropy_with_logits(
            model(tx), ty, reduction="none") * tw).sum()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "weighted_bce": float(loss.detach()),
                        "learning_rate": optimizer.param_groups[0]["lr"]})
    model.eval()
    return model, history


def predict(model, x):
    x = np.asarray(x, np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite scoring input")
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(x))).numpy().astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite score")
    return scores
