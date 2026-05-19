#!/usr/bin/env python3
"""Generate the three UNSW-NB15 portfolio notebooks."""
import json


def src(text):
    lines = text.strip("\n").split("\n")
    if not lines or lines == [""]:
        return [""]
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def C(cid, text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid,
        "metadata": {},
        "outputs": [],
        "source": src(text),
    }


def M(cid, text):
    return {
        "cell_type": "markdown",
        "id": cid,
        "metadata": {},
        "source": src(text),
    }


def NB(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }


# ===========================================================================
# NOTEBOOK 01 — EDA & PREPROCESSING
# ===========================================================================

NB01 = NB(
    [
        M(
            "a00",
            """\
# 01 — EDA & Preprocessing
### UNSW-NB15 Network Intrusion Detection Dataset

Loads, explores, and preprocesses ~2.54M labeled network flow records captured
across two days of synthetic network activity (UNSW Canberra Cyber Range, 2015).

Each record describes a single network connection with 47 features covering protocol
behaviour, packet statistics, and connection timing — labelled as **normal** or one of
**9 attack categories**.

**Outputs**
- `data/processed/traffic_cleaned.parquet` — merged, cleaned dataset for modeling
- PNG figures saved to `data/processed/`""",
        ),
        C(
            "a01",
            """\
import os
import warnings
import numpy as np
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 100})

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)""",
        ),
        M("a02", "## 1. Data Download"),
        C(
            "a03",
            """\
# kagglehub caches the dataset after the first download.
# Requires ~/.kaggle/kaggle.json — see README for setup.
dataset_path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
print(f"Dataset path: {dataset_path}")
print("Files:", sorted(os.listdir(dataset_path)))""",
        ),
        M(
            "a04",
            """\
## 2. Feature Reference

The four raw CSV files (`UNSW-NB15_1.csv` – `_4.csv`) have **no header row**.
Column names are applied manually from `NUSW-NB15_features.csv`, which documents all 49 columns.""",
        ),
        C(
            "a05",
            """\
features_df = pd.read_csv(
    f"{dataset_path}/NUSW-NB15_features.csv", encoding="ISO-8859-1"
)
display(features_df[["No.", "Name", "Type", "Description"]])""",
        ),
        C(
            "a06",
            """\
COLUMN_NAMES = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
    "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]
assert len(COLUMN_NAMES) == 49, "Expected 49 columns"
print(f"Defined {len(COLUMN_NAMES)} column names")""",
        ),
        M(
            "a07",
            """\
## 3. Attack Category Analysis

`UNSW-NB15_LIST_EVENTS.csv` summarises the threat landscape: attack categories,
subcategories, and event counts. This gives a high-level view before loading
the full traffic data.""",
        ),
        C(
            "a08",
            """\
list_events_df = pd.read_csv(f"{dataset_path}/UNSW-NB15_LIST_EVENTS.csv")
list_events_df["Attack category"] = (
    list_events_df["Attack category"].str.strip().str.lower()
)
list_events_df["Attack subcategory"] = (
    list_events_df["Attack subcategory"].str.strip().str.lower()
)
list_events_df.dropna(subset=["Attack category", "Attack subcategory"], inplace=True)

print("Categories:", sorted(list_events_df["Attack category"].unique()))
print("Subcategories:", list_events_df["Attack subcategory"].nunique())""",
        ),
        C(
            "a09",
            """\
attack_cat_sum = (
    list_events_df.groupby("Attack category")["Number of events"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
attack_sub_sum = (
    list_events_df.groupby("Attack subcategory")["Number of events"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(
    data=attack_cat_sum, y="Attack category", x="Number of events",
    ax=axes[0], palette="viridis",
)
axes[0].set_title("Total Events by Attack Category", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Number of Events")
axes[0].set_ylabel("")

sns.barplot(
    data=attack_sub_sum, y="Attack subcategory", x="Number of events",
    ax=axes[1], palette="magma",
)
axes[1].set_xscale("log")
axes[1].set_title("Top 20 Subcategories (log scale)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Number of Events (log)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig("data/processed/fig_attack_categories.png", dpi=150, bbox_inches="tight")
plt.show()""",
        ),
        M("a10", "## 4. Load Traffic Data"),
        C(
            "a11",
            """\
dfs = []
for i in range(1, 5):
    filepath = f"{dataset_path}/UNSW-NB15_{i}.csv"
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES, low_memory=False)
    print(f"  UNSW-NB15_{i}.csv  →  {df.shape[0]:>9,} rows")
    dfs.append(df)

traffic_df = pd.concat(dfs, ignore_index=True)
print(f"\\n  Combined: {traffic_df.shape[0]:,} rows × {traffic_df.shape[1]} columns")""",
        ),
        C(
            "a12",
            """\
traffic_df["label"] = (
    pd.to_numeric(traffic_df["label"], errors="coerce").fillna(0).astype(int)
)
traffic_df["attack_cat"] = (
    traffic_df["attack_cat"].fillna("normal").str.strip().str.lower()
)

counts = traffic_df["label"].value_counts().sort_index()
print(f"Normal  {counts[0]:>10,}  ({counts[0] / len(traffic_df) * 100:.1f}%)")
print(f"Attack  {counts[1]:>10,}  ({counts[1] / len(traffic_df) * 100:.1f}%)")""",
        ),
        M("a13", "## 5. Exploratory Data Analysis"),
        C(
            "a14",
            """\
# --- Class Balance ---
label_counts = (
    traffic_df["label"].value_counts().rename({0: "Normal", 1: "Attack"}).sort_index()
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].pie(
    label_counts, labels=label_counts.index, autopct="%1.1f%%",
    colors=["#2196F3", "#F44336"], startangle=90, textprops={"fontsize": 12},
)
axes[0].set_title("Class Balance", fontsize=13, fontweight="bold")

bars = axes[1].bar(label_counts.index, label_counts.values, color=["#2196F3", "#F44336"])
axes[1].set_title("Record Counts by Class", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Count")
for bar in bars:
    axes[1].text(
        bar.get_x() + bar.get_width() / 2, bar.get_height(),
        f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=11,
    )

plt.tight_layout()
plt.savefig("data/processed/fig_class_balance.png", dpi=150, bbox_inches="tight")
plt.show()""",
        ),
        C(
            "a15",
            """\
# --- Protocol & State Distributions ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

proto_counts = traffic_df["proto"].value_counts().head(10)
axes[0].barh(proto_counts.index, proto_counts.values,
             color=sns.color_palette("Blues_r", 10))
axes[0].set_title("Top 10 Protocols", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Count")

state_counts = traffic_df["state"].value_counts().head(8)
axes[1].barh(state_counts.index, state_counts.values,
             color=sns.color_palette("Greens_r", 8))
axes[1].set_title("Top 8 Connection States", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Count")

plt.tight_layout()
plt.savefig("data/processed/fig_protocol_state.png", dpi=150, bbox_inches="tight")
plt.show()""",
        ),
        C(
            "a16",
            """\
# --- Feature Distributions: Normal vs. Attack ---
KEY_FEATURES = [
    "dur", "sbytes", "dbytes", "sttl", "dttl",
    "sload", "dload", "spkts", "dpkts",
]
sample = traffic_df.sample(n=50_000, random_state=42)

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
for ax, feat in zip(axes.flatten(), KEY_FEATURES):
    for lbl, color, name in [(0, "#2196F3", "Normal"), (1, "#F44336", "Attack")]:
        data = (
            pd.to_numeric(sample.loc[sample["label"] == lbl, feat], errors="coerce")
            .dropna()
        )
        ax.hist(
            data.clip(upper=data.quantile(0.99)),
            bins=40, alpha=0.55, color=color, label=name, density=True,
        )
    ax.set_title(feat, fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylabel("Density")

plt.suptitle(
    "Feature Distributions: Normal vs. Attack  (50k sample)",
    fontsize=14, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig(
    "data/processed/fig_feature_distributions.png", dpi=150, bbox_inches="tight"
)
plt.show()""",
        ),
        C(
            "a17",
            """\
# --- Correlation Heatmap (100k sample for speed) ---
numeric_cols = (
    traffic_df.select_dtypes(include=[np.number])
    .columns.difference(["label"])
    .tolist()
)
corr = (
    traffic_df[numeric_cols]
    .apply(pd.to_numeric, errors="coerce")
    .sample(n=100_000, random_state=42)
    .corr()
)

plt.figure(figsize=(18, 15))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
    linewidths=0.3, annot=False,
)
plt.title(
    "Feature Correlation Matrix — lower triangle, 100k sample",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    "data/processed/fig_correlation_heatmap.png", dpi=150, bbox_inches="tight"
)
plt.show()""",
        ),
        M(
            "a18",
            """\
## 6. Preprocessing & Save

**Dropped columns**
| Column | Reason |
|---|---|
| `srcip`, `dstip` | Raw IP addresses — not generalizable ML features |
| `stime`, `ltime` | Absolute Unix timestamps — not meaningful across capture windows |

Mixed-type columns (`sport`, `dsport`, `ct_flw_http_mthd`, `is_ftp_login`, `ct_ftp_cmd`) are
coerced to numeric; non-parsable values become `NaN`.

Output: `data/processed/traffic_cleaned.parquet`""",
        ),
        C(
            "a19",
            """\
DROP_COLS = ["srcip", "dstip", "stime", "ltime"]
cleaned = traffic_df.drop(columns=DROP_COLS).copy()

for col in ["proto", "state", "service", "attack_cat"]:
    cleaned[col] = cleaned[col].astype(str).str.strip().str.lower()

mixed_cols = ["sport", "dsport", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"]
for col in mixed_cols:
    cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

print(f"Cleaned shape  : {cleaned.shape}")
print(f"Missing values : {cleaned.isnull().sum().sum():,}")

out = "data/processed/traffic_cleaned.parquet"
cleaned.to_parquet(out, index=False)
print(f"Saved → {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")""",
        ),
    ]
)


# ===========================================================================
# NOTEBOOK 02 — MODELING
# ===========================================================================

NB02 = NB(
    [
        M(
            "b00",
            """\
# 02 — Modeling Pipeline
### Binary Network Intrusion Detection: Normal vs. Attack

Three classifiers are trained and compared on the preprocessed UNSW-NB15 dataset.

| Model | Role |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Ensemble benchmark |
| XGBoost | Primary model |

**Sampling:** A stratified 500k-row sample is used by default to keep training time
reasonable on local hardware. Set `SAMPLE_SIZE = None` to train on the full 2.54M rows.

**Outputs**
- `models/xgb_model.joblib` — saved XGBoost pipeline
- `models/feature_meta.json` — feature name metadata for notebook 03
- `data/processed/X_test.parquet`, `data/processed/y_test.parquet`""",
        ),
        C(
            "b01",
            """\
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 100})

os.makedirs("models", exist_ok=True)""",
        ),
        M("b02", "## 1. Load Data"),
        C(
            "b03",
            """\
df = pd.read_parquet("data/processed/traffic_cleaned.parquet")
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
counts = df["label"].value_counts().sort_index()
print(f"\\nNormal  {counts[0]:>9,}  ({counts[0]/len(df)*100:.1f}%)")
print(f"Attack  {counts[1]:>9,}  ({counts[1]/len(df)*100:.1f}%)")""",
        ),
        M(
            "b04",
            """\
## 2. Feature Setup

- `attack_cat` is dropped — it directly encodes the label (data leakage).
- Categorical features are one-hot encoded; numerical features are standardized.
- `class_weight="balanced"` / `scale_pos_weight` compensate for the ~87/13 class split.""",
        ),
        C(
            "b05",
            """\
TARGET = "label"
DROP_FEATURES = ["attack_cat"]
CAT_FEATURES = ["proto", "state", "service"]
NUM_FEATURES = [
    c for c in df.columns
    if c not in CAT_FEATURES + DROP_FEATURES + [TARGET]
]

SAMPLE_SIZE = 500_000  # set to None to use full dataset

print(f"Categorical ({len(CAT_FEATURES)}): {CAT_FEATURES}")
print(f"Numerical   ({len(NUM_FEATURES)}): {NUM_FEATURES[:5]} ... ")""",
        ),
        C(
            "b06",
            """\
X = df.drop(columns=DROP_FEATURES + [TARGET])
y = df[TARGET]

if SAMPLE_SIZE:
    X, _, y, _ = train_test_split(
        X, y, train_size=SAMPLE_SIZE, stratify=y, random_state=42
    )
    print(f"Stratified sample: {X.shape[0]:,} rows")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"Attack rate — Train: {y_train.mean():.3f}  |  Test: {y_test.mean():.3f}")

X_test.to_parquet("data/processed/X_test.parquet", index=False)
y_test.to_frame().to_parquet("data/processed/y_test.parquet", index=False)""",
        ),
        M("b07", "## 3. Preprocessing Pipeline"),
        C(
            "b08",
            """\
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ],
    remainder="drop",
)

# Save feature metadata for notebook 03
meta = {
    "NUM_FEATURES": NUM_FEATURES,
    "CAT_FEATURES": CAT_FEATURES,
    "TARGET": TARGET,
    "DROP_FEATURES": DROP_FEATURES,
}
with open("models/feature_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Preprocessing pipeline defined")
print(f"  Numerical  (StandardScaler) : {len(NUM_FEATURES)} features")
print(f"  Categorical (OneHotEncoder) : {len(CAT_FEATURES)} features → expanded")
print("Saved → models/feature_meta.json")""",
        ),
        M("b09", "## 4. Model Training"),
        C(
            "b10",
            """\
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )),
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=42,
        )),
    ]),
    "XGBoost": Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )),
    ]),
}

results = {}
for name, model in models.items():
    print(f"Training {name}...", end=" ", flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    print(f"done  |  F1={results[name]['f1']:.4f}  AUC={results[name]['roc_auc']:.4f}")""",
        ),
        M("b11", "## 5. Evaluation"),
        C(
            "b12",
            """\
for name, res in results.items():
    print(f"\\n{'='*52}")
    print(f"  {name}")
    print("=" * 52)
    print(classification_report(y_test, res["y_pred"], target_names=["Normal", "Attack"]))""",
        ),
        C(
            "b13",
            """\
# Confusion matrices (row-normalised %)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues", ax=ax,
        xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"],
        cbar_kws={"label": "% of True Class"},
    )
    ax.set_title(
        f"{name}\\nF1={res['f1']:.4f}  AUC={res['roc_auc']:.4f}",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

plt.suptitle("Confusion Matrices (row-normalised %)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("data/processed/fig_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()""",
        ),
        C(
            "b14",
            """\
# ROC curves
plt.figure(figsize=(8, 6))
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, lw=2, color=color,
             label=f"{name}  (AUC = {res['roc_auc']:.4f})")

plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
plt.xlim([0, 1])
plt.ylim([0, 1.02])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves — Network Intrusion Detection", fontsize=13, fontweight="bold")
plt.legend(loc="lower right", fontsize=11)
plt.tight_layout()
plt.savefig("data/processed/fig_roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()""",
        ),
        C(
            "b15",
            """\
# Model comparison bar chart
metrics_df = pd.DataFrame(
    {name: {"F1-Score": res["f1"], "ROC-AUC": res["roc_auc"]} for name, res in results.items()}
).T.reset_index().rename(columns={"index": "Model"})

metrics_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(9, 5))
ax = sns.barplot(
    data=metrics_melted, x="Model", y="Score", hue="Metric",
    palette=["#4ECDC4", "#FF6B6B"],
)
plt.ylim(0.80, 1.01)
plt.title("Model Performance Comparison", fontsize=13, fontweight="bold")
plt.ylabel("Score")
plt.legend(title="Metric")
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.4f}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center", va="bottom", fontsize=9,
    )
plt.tight_layout()
plt.savefig("data/processed/fig_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print(metrics_df.to_string(index=False))""",
        ),
        M(
            "b16",
            "## 6. Save Best Model\n\nXGBoost is saved for SHAP explainability in notebook 03.",
        ),
        C(
            "b17",
            """\
joblib.dump(results["XGBoost"]["model"], "models/xgb_model.joblib")
print("Saved → models/xgb_model.joblib")
print(f"\\nFinal XGBoost results:")
print(f"  F1-Score : {results['XGBoost']['f1']:.4f}")
print(f"  ROC-AUC  : {results['XGBoost']['roc_auc']:.4f}")""",
        ),
    ]
)


# ===========================================================================
# NOTEBOOK 03 — EXPLAINABILITY
# ===========================================================================

NB03 = NB(
    [
        M(
            "c00",
            """\
# 03 — Model Explainability
### SHAP Analysis of XGBoost Network Intrusion Detector

SHAP (SHapley Additive exPlanations) provides theoretically grounded, additive feature
attributions for every prediction. This notebook interprets the XGBoost model trained
in notebook 02 at both the **global** (dataset-wide) and **local** (per-prediction) level.

| Plot | What it shows |
|---|---|
| Feature importance bar | Mean \\|SHAP\\| value — overall feature impact |
| Beeswarm summary | Distribution of SHAP values + direction of effect |
| Waterfall (normal) | Why a benign connection was classified as safe |
| Waterfall (attack) | Why a malicious connection was flagged |
| Dependence plot | How the top feature's impact varies with its value |""",
        ),
        C(
            "c01",
            """\
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

warnings.filterwarnings("ignore")
shap.initjs()
plt.rcParams.update({"figure.dpi": 100})""",
        ),
        M("c02", "## 1. Load Model & Test Data"),
        C(
            "c03",
            """\
model = joblib.load("models/xgb_model.joblib")

X_test = pd.read_parquet("data/processed/X_test.parquet")
y_test = pd.read_parquet("data/processed/y_test.parquet").squeeze()

with open("models/feature_meta.json") as f:
    meta = json.load(f)
NUM_FEATURES = meta["NUM_FEATURES"]
CAT_FEATURES = meta["CAT_FEATURES"]

print(f"Test set : {X_test.shape[0]:,} rows × {X_test.shape[1]} columns")
print(f"Attack rate : {y_test.mean():.3f}")""",
        ),
        M(
            "c04",
            """\
## 2. SHAP Explainer Setup

`TreeExplainer` computes exact SHAP values for tree-based models — no approximation needed.

We extract the XGBoost classifier from the sklearn Pipeline, transform the test data using
the fitted preprocessor, and reconstruct post-encoding feature names for interpretable plots.

A 5,000-row subsample is used for SHAP computation; the full test set can be used at the
cost of longer runtime.""",
        ),
        C(
            "c05",
            """\
preprocessor = model.named_steps["prep"]
xgb_clf = model.named_steps["clf"]

# Reconstruct feature names after one-hot encoding
cat_encoder = preprocessor.named_transformers_["cat"]
cat_feature_names = cat_encoder.get_feature_names_out(CAT_FEATURES).tolist()
feature_names = NUM_FEATURES + cat_feature_names
print(f"Total post-encoding features: {len(feature_names)}")

# Transform test data and wrap in DataFrame so SHAP carries feature names
X_test_arr = preprocessor.transform(X_test)
X_test_transformed = pd.DataFrame(X_test_arr, columns=feature_names)

# Stratified subsample for SHAP
SHAP_SAMPLE = 5_000
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X_test_transformed), SHAP_SAMPLE, replace=False)
X_shap = X_test_transformed.iloc[sample_idx]
y_shap = y_test.iloc[sample_idx].values

explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer(X_shap)
print(f"SHAP values computed for {SHAP_SAMPLE:,} samples")""",
        ),
        M("c06", "## 3. Global Feature Importance"),
        C(
            "c07",
            """\
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance_df = (
    pd.DataFrame({"feature": feature_names, "mean_|SHAP|": mean_abs_shap})
    .sort_values("mean_|SHAP|", ascending=False)
    .head(20)
    .reset_index(drop=True)
)

plt.figure(figsize=(10, 7))
sns.barplot(data=importance_df, x="mean_|SHAP|", y="feature", palette="viridis")
plt.title("Top 20 Features — Mean |SHAP| Value", fontsize=13, fontweight="bold")
plt.xlabel("Mean |SHAP| Value")
plt.ylabel("")
plt.tight_layout()
plt.savefig("data/processed/fig_shap_importance.png", dpi=150, bbox_inches="tight")
plt.show()

print("\\nTop 10:")
print(importance_df.head(10).to_string(index=False))""",
        ),
        M(
            "c08",
            """\
## 4. SHAP Beeswarm Summary

Each dot is one prediction. **Colour** = feature value (red = high, blue = low).
**Horizontal position** = SHAP value (positive → pushes toward *attack*, negative → *normal*).""",
        ),
        C(
            "c09",
            """\
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.title("SHAP Beeswarm — Global Feature Impact", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("data/processed/fig_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()""",
        ),
        M(
            "c10",
            """\
## 5. Waterfall Plots — Individual Predictions

Waterfall plots decompose a single prediction into per-feature contributions,
starting from the model's base rate (E[f(x)]) and ending at the final prediction.

We select one correctly classified **normal** flow and one correctly classified **attack** flow.""",
        ),
        C(
            "c11",
            """\
# Determine predicted labels from SHAP values (log-odds space: >0 → attack)
log_odds = shap_values.values.sum(axis=1) + explainer.expected_value
y_pred_shap = (log_odds > 0).astype(int)

correct_normal = np.where((y_shap == 0) & (y_pred_shap == 0))[0]
correct_attack = np.where((y_shap == 1) & (y_pred_shap == 1))[0]

normal_i = correct_normal[0]
attack_i = correct_attack[0]

# Normal flow
shap.plots.waterfall(shap_values[normal_i], max_display=14, show=False)
plt.title("Normal Flow — Prediction Breakdown", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("data/processed/fig_shap_waterfall_normal.png", dpi=150, bbox_inches="tight")
plt.show()

# Attack flow
shap.plots.waterfall(shap_values[attack_i], max_display=14, show=False)
plt.title("Attack Flow — Prediction Breakdown", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("data/processed/fig_shap_waterfall_attack.png", dpi=150, bbox_inches="tight")
plt.show()""",
        ),
        M(
            "c12",
            """\
## 6. Dependence Plot — Top Feature

Shows how the top feature's SHAP value varies with its raw value.
Colour encodes the feature SHAP most interacts with (auto-selected).""",
        ),
        C(
            "c13",
            """\
top_feature = importance_df.iloc[0]["feature"]
top_idx = feature_names.index(top_feature)

shap.plots.scatter(shap_values[:, top_idx], color=shap_values, show=False)
plt.title(f"SHAP Dependence: {top_feature}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("data/processed/fig_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Top feature : {top_feature}")
print(f"Mean |SHAP| : {importance_df.iloc[0]['mean_|SHAP|']:.4f}")""",
        ),
    ]
)


# ===========================================================================
# Write notebooks
# ===========================================================================
notebooks = {
    "01_eda_preprocessing.ipynb": NB01,
    "02_modeling.ipynb": NB02,
    "03_explainability.ipynb": NB03,
}

for filename, notebook in notebooks.items():
    with open(filename, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Written: {filename}")

print("\nDone.")
