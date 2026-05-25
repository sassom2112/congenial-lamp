"""
PyTorch MLP surrogate detector for adversarial attack evaluation.

Trained on the same preprocessed UNSW-NB15 features as the XGBoost baseline,
giving a differentiable model for FGSM/PGD gradient computation.
Input dim: 200 (40 standardized numeric + 160 one-hot categorical).
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score


class IntrusionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 128, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: tuple = (256, 128, 64),
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 30,
    batch_size: int = 2048,
    patience: int = 5,
    device: torch.device = None,
) -> IntrusionMLP:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v_np = y_val

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, num_workers=0)

    model = IntrusionMLP(X_train.shape[1], hidden_dims, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-5
    )

    best_f1, best_state, no_improve = 0.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_v).cpu().numpy()
        val_preds = (val_logits > 0).astype(int)
        val_f1 = f1_score(y_v_np, val_preds, zero_division=0)
        scheduler.step(epoch_loss / len(X_train))

        print(f"  Epoch {epoch:3d} | loss={epoch_loss/len(X_train):.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch} (best val_f1={best_f1:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_model(
    model: IntrusionMLP,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device = None,
    batch_size: int = 4096,
) -> dict:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    logits_list = []
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
    )
    with torch.no_grad():
        for (xb,) in loader:
            logits_list.append(model(xb.to(device)).cpu())
    logits = torch.cat(logits_list).numpy()
    preds = (logits > 0).astype(int)
    proba = torch.sigmoid(torch.tensor(logits)).numpy()
    return {
        "f1": f1_score(y, preds, zero_division=0),
        "auc": roc_auc_score(y, proba),
        "accuracy": float((preds == y).mean()),
    }


def save_model(model: IntrusionMLP, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"input_dim": model.input_dim, "hidden_dims": model.hidden_dims},
        },
        path,
    )
    print(f"Model saved to {path}")


def load_model(path: str, device: torch.device = None) -> IntrusionMLP:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = IntrusionMLP(cfg["input_dim"], tuple(cfg["hidden_dims"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
