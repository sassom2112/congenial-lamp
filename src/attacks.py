"""
Adversarial attack suite for network intrusion detection.

Implements FGSM and PGD with domain-aware constraint projection.
Attacks operate in the preprocessed (standardized + OHE) feature space.

Key design: constraint projection keeps adversarial flows physically plausible.
  - Non-negative features: lb = (0 - mean) / std in standardized space
  - TTL-bounded [0,255]: ub = (255 - mean) / std
  - Boolean [0,1]: lb/ub computed from scaler stats
  - Port [0,65535]: ub computed from scaler stats
  - OHE categorical features: clamped to [0,1]

Most published FGSM papers on IDS skip constraint projection entirely — flows
they produce would have negative packet counts, fractional TTLs, etc. Applying
domain constraints here is what makes the perturbations physically meaningful.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


# ── Feature index constants (position in NUM_FEATURES list) ──────────────────
# Matches order in feature_meta.json / fitted ColumnTransformer 'num' branch.
_FEAT_IDX = {
    "sport": 0, "dsport": 1, "dur": 2, "sbytes": 3, "dbytes": 4,
    "sttl": 5, "dttl": 6, "sloss": 7, "dloss": 8,
    "sload": 9, "dload": 10, "spkts": 11, "dpkts": 12,
    "swin": 13, "dwin": 14, "stcpb": 15, "dtcpb": 16,
    "smeansz": 17, "dmeansz": 18, "trans_depth": 19,
    "res_bdy_len": 20, "sjit": 21, "djit": 22,
    "sintpkt": 23, "dintpkt": 24, "tcprtt": 25,
    "synack": 26, "ackdat": 27, "is_sm_ips_ports": 28,
    "ct_state_ttl": 29, "ct_flw_http_mthd": 30, "is_ftp_login": 31,
    "ct_ftp_cmd": 32, "ct_srv_src": 33, "ct_srv_dst": 34,
    "ct_dst_ltm": 35, "ct_src_ltm": 36, "ct_src_dport_ltm": 37,
    "ct_dst_sport_ltm": 38, "ct_dst_src_ltm": 39,
}

# Features with a natural lower bound of 0 in the original domain
_NONNEG = {
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
    "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
    "sport", "dsport", "ct_state_ttl", "ct_flw_http_mthd",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
    "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_sm_ips_ports", "is_ftp_login",
}
_TTL = {"sttl": 255.0, "dttl": 255.0}
_BOOL = {"is_sm_ips_ports": 1.0, "is_ftp_login": 1.0}
_PORTS = {"sport": 65535.0, "dsport": 65535.0}


@dataclass
class ConstraintBounds:
    """Per-feature lower/upper bounds in the preprocessed feature space (n_features=200)."""
    lb: np.ndarray  # shape (200,)
    ub: np.ndarray  # shape (200,)
    n_num: int      # number of numeric (standardized) features = 40
    n_total: int    # total features = 200

    @classmethod
    def from_pipeline(cls, pipeline) -> "ConstraintBounds":
        """Extract bounds from the fitted sklearn pipeline."""
        pre = pipeline.named_steps["prep"]
        scaler = pre.named_transformers_["num"].named_steps["scaler"]
        ohe = pre.named_transformers_["cat"].named_steps["encoder"]

        n_num = scaler.n_features_in_
        n_ohe = sum(len(cats) for cats in ohe.categories_)
        n_total = n_num + n_ohe

        lb = np.full(n_total, -np.inf, dtype=np.float32)
        ub = np.full(n_total, np.inf, dtype=np.float32)

        mean_ = scaler.mean_
        std_ = scaler.scale_

        for feat, idx in _FEAT_IDX.items():
            if feat in _NONNEG:
                lb[idx] = (0.0 - mean_[idx]) / std_[idx]
            if feat in _TTL:
                ub[idx] = (_TTL[feat] - mean_[idx]) / std_[idx]
            if feat in _BOOL:
                ub[idx] = (_BOOL[feat] - mean_[idx]) / std_[idx]
            if feat in _PORTS:
                ub[idx] = (_PORTS[feat] - mean_[idx]) / std_[idx]

        # OHE features: continuous relaxation to [0, 1]
        lb[n_num:] = 0.0
        ub[n_num:] = 1.0

        return cls(lb=lb, ub=ub, n_num=n_num, n_total=n_total)


def _project(x: torch.Tensor, bounds: ConstraintBounds) -> torch.Tensor:
    lb = torch.tensor(bounds.lb, dtype=x.dtype, device=x.device)
    ub = torch.tensor(bounds.ub, dtype=x.dtype, device=x.device)
    return torch.clamp(x, lb, ub)


# ── Attack implementations ───────────────────────────────────────────────────

def fgsm(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    bounds: ConstraintBounds,
    device: torch.device = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """Single-step Fast Gradient Sign Method with domain constraint projection."""
    if device is None:
        device = next(model.parameters()).device

    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    results = []

    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, requires_grad=True).to(device)
        yb = torch.tensor(y[i:i+batch_size], dtype=torch.float32).to(device)

        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        with torch.no_grad():
            x_adv = xb + epsilon * xb.grad.sign()
            x_adv = _project(x_adv, bounds)
        results.append(x_adv.cpu().numpy())

    return np.concatenate(results, axis=0)


def pgd(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    alpha: float,
    n_steps: int,
    bounds: ConstraintBounds,
    device: torch.device = None,
    random_init: bool = True,
    batch_size: int = 2048,
) -> np.ndarray:
    """Multi-step PGD attack (Madry et al.) with domain constraint projection."""
    if device is None:
        device = next(model.parameters()).device

    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    results = []

    for i in range(0, len(X), batch_size):
        X_orig = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
        y_b = torch.tensor(y[i:i+batch_size], dtype=torch.float32).to(device)

        if random_init:
            delta = torch.zeros_like(X_orig).uniform_(-epsilon, epsilon)
            x_adv = _project(X_orig + delta, bounds)
        else:
            x_adv = X_orig.clone()

        for _ in range(n_steps):
            x_adv = x_adv.detach().requires_grad_(True)
            logits = model(x_adv)
            loss = criterion(logits, y_b)
            loss.backward()

            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                # Stay within epsilon-ball of the original point...
                delta = torch.clamp(x_adv - X_orig, -epsilon, epsilon)
                # ...and within domain constraints
                x_adv = _project(X_orig + delta, bounds)

        results.append(x_adv.detach().cpu().numpy())

    return np.concatenate(results, axis=0)


# ── Evaluation helpers ───────────────────────────────────────────────────────

def _get_preds(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            logits.append(model(xb).cpu().numpy())
    return (np.concatenate(logits) > 0).astype(int)


def evasion_rate(
    model: nn.Module,
    X_adv: np.ndarray,
    y: np.ndarray,
    device: torch.device = None,
) -> float:
    """Fraction of true-attack samples (y==1) that flip to predicted normal after perturbation."""
    if device is None:
        device = next(model.parameters()).device
    preds = _get_preds(model, X_adv, device)
    attack_mask = y == 1
    if attack_mask.sum() == 0:
        return 0.0
    return float((preds[attack_mask] == 0).mean())


def accuracy_under_attack(
    model: nn.Module,
    X_adv: np.ndarray,
    y: np.ndarray,
    device: torch.device = None,
) -> float:
    if device is None:
        device = next(model.parameters()).device
    preds = _get_preds(model, X_adv, device)
    return float((preds == y).mean())


def xgb_transfer_evasion(pipeline, X_adv_transformed: np.ndarray, y: np.ndarray) -> float:
    """
    Transfer attack: adversarial examples crafted against the MLP tested on XGBoost.

    X_adv_transformed is already in the preprocessed feature space (200-dim).
    We pass it directly to the XGBoost classifier, bypassing the sklearn preprocessor
    (which already ran when the data was transformed for the MLP).
    """
    xgb_clf = pipeline.named_steps["clf"]
    preds = xgb_clf.predict(X_adv_transformed)
    attack_mask = y == 1
    if attack_mask.sum() == 0:
        return 0.0
    return float((preds[attack_mask] == 0).mean())
