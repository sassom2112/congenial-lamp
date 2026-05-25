"""
Robustness evaluation: accuracy-vs-epsilon curves, per-category evasion,
and clean vs. hardened model comparison tables.
"""

from typing import Callable, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def _get_preds(
    model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 4096
) -> np.ndarray:
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            logits.append(model(xb).cpu().numpy())
    return (np.concatenate(logits) > 0).astype(int)


def robustness_curve(
    model: nn.Module,
    X_clean: np.ndarray,
    y: np.ndarray,
    epsilons: List[float],
    attack_fn: Callable,
    device: torch.device = None,
) -> pd.DataFrame:
    """
    Sweep epsilon values and record F1, accuracy, and evasion rate at each level.
    attack_fn signature: attack_fn(epsilon=float) -> np.ndarray (X_adv)
    """
    if device is None:
        device = next(model.parameters()).device

    clean_preds = _get_preds(model, X_clean, device)
    clean_f1 = f1_score(y, clean_preds, zero_division=0)
    clean_acc = float((clean_preds == y).mean())

    rows = []
    for eps in epsilons:
        X_adv = attack_fn(epsilon=eps)
        adv_preds = _get_preds(model, X_adv, device)
        attack_mask = y == 1
        evasion = float((adv_preds[attack_mask] == 0).mean()) if attack_mask.sum() > 0 else 0.0
        rows.append({
            "epsilon": eps,
            "clean_f1": clean_f1,
            "clean_acc": clean_acc,
            "adv_f1": f1_score(y, adv_preds, zero_division=0),
            "adv_acc": float((adv_preds == y).mean()),
            "evasion_rate": evasion,
            "f1_drop": clean_f1 - f1_score(y, adv_preds, zero_division=0),
        })

    return pd.DataFrame(rows)


def per_category_evasion(
    model: nn.Module,
    X_adv: np.ndarray,
    y: np.ndarray,
    attack_cat: np.ndarray,
    device: torch.device = None,
) -> pd.DataFrame:
    """
    Per MITRE attack category: sample count, evasion rate under perturbation.
    Only evaluates rows where y==1 (actual attacks).
    """
    if device is None:
        device = next(model.parameters()).device

    adv_preds = _get_preds(model, X_adv, device)
    rows = []
    for cat in sorted(np.unique(attack_cat[y == 1])):
        mask = (attack_cat == cat) & (y == 1)
        if mask.sum() == 0:
            continue
        evaded = float((adv_preds[mask] == 0).mean())
        rows.append({
            "attack_category": cat,
            "n_samples": int(mask.sum()),
            "evasion_rate": evaded,
        })

    return pd.DataFrame(rows).sort_values("evasion_rate", ascending=False).reset_index(drop=True)


def compare_clean_vs_hardened(
    standard_model: nn.Module,
    hardened_model: nn.Module,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y: np.ndarray,
    device: torch.device = None,
) -> pd.DataFrame:
    """Side-by-side metric comparison: standard vs. adversarially trained model."""
    if device is None:
        device = next(standard_model.parameters()).device

    rows = []
    for label, model in [("Standard MLP", standard_model), ("Adversarially Trained MLP", hardened_model)]:
        c_preds = _get_preds(model, X_clean, device)
        a_preds = _get_preds(model, X_adv, device)
        attack_mask = y == 1
        rows.append({
            "Model": label,
            "Clean F1": f1_score(y, c_preds, zero_division=0),
            "Adv F1": f1_score(y, a_preds, zero_division=0),
            "Clean Acc": float((c_preds == y).mean()),
            "Adv Acc": float((a_preds == y).mean()),
            "Evasion Rate": float((a_preds[attack_mask] == 0).mean()) if attack_mask.sum() > 0 else 0.0,
        })

    df = pd.DataFrame(rows).set_index("Model")
    df["F1 Retained (%)"] = (df["Adv F1"] / df["Clean F1"] * 100).round(1)
    return df


# ── Plotting helpers ─────────────────────────────────────────────────────────

def plot_robustness_curves(
    fgsm_df: pd.DataFrame,
    pgd_df: pd.DataFrame,
    save_path: str = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, col, ylabel, title in [
        (axes[0], "adv_f1",       "F1 Score",      "F1 Score vs Perturbation Budget"),
        (axes[1], "adv_acc",      "Accuracy",      "Accuracy vs Perturbation Budget"),
        (axes[2], "evasion_rate", "Evasion Rate",  "Evasion Rate vs Perturbation Budget"),
    ]:
        ax.plot(fgsm_df["epsilon"], fgsm_df[col], "o-", label="FGSM", color="#e74c3c")
        ax.plot(pgd_df["epsilon"],  pgd_df[col],  "s-", label="PGD",  color="#2980b9")
        if col != "evasion_rate":
            ax.axhline(fgsm_df["clean_f1"].iloc[0] if col == "adv_f1" else fgsm_df["clean_acc"].iloc[0],
                       ls="--", color="grey", alpha=0.6, label="Clean baseline")
        ax.set_xlabel("Epsilon (ε)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.grid(alpha=0.3)

    plt.suptitle("MLP Surrogate Robustness: FGSM vs PGD on UNSW-NB15", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_category_evasion(df: pd.DataFrame, title: str = "Evasion Rate by Attack Category", save_path: str = None):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(df)))
    bars = ax.barh(df["attack_category"], df["evasion_rate"], color=colors)
    ax.bar_label(bars, fmt="%.1%", padding=3, fontsize=9)
    ax.set_xlabel("Evasion Rate (fraction of attacks evaded)")
    ax.set_title(title)
    ax.set_xlim(0, min(1.05, df["evasion_rate"].max() * 1.2 + 0.05))
    ax.axvline(0.5, ls="--", color="red", alpha=0.4, label="50% threshold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_hardening_comparison(
    standard_curve: pd.DataFrame,
    hardened_curve: pd.DataFrame,
    save_path: str = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, ylabel in [
        (axes[0], "adv_f1",       "F1 Score"),
        (axes[1], "evasion_rate", "Evasion Rate"),
    ]:
        ax.plot(standard_curve["epsilon"], standard_curve[col], "o-",
                label="Standard MLP", color="#e74c3c")
        ax.plot(hardened_curve["epsilon"], hardened_curve[col], "s-",
                label="Adversarially Trained", color="#27ae60")
        ax.set_xlabel("Epsilon (ε)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Perturbation Budget")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Standard vs. Adversarially Trained MLP Robustness (PGD Attack)", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_perturbation_magnitude(
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    feature_names: List[str],
    top_n: int = 15,
    save_path: str = None,
):
    """Which features receive the largest mean absolute perturbation?"""
    delta = np.abs(X_adv - X_clean)
    mean_delta = delta.mean(axis=0)
    top_idx = np.argsort(mean_delta)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(top_n), mean_delta[top_idx], color="#8e44ad")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in top_idx], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean |Δ| (standardized units)")
    ax.set_title(f"Top {top_n} Most-Perturbed Features")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
