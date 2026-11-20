"""Fairness check by sex and by age group. Run with:  python src/fairness.py

Writes outputs/results/fairness_by_group.csv and outputs/figures/fairness.png

Dataset A: out-of-fold predictions (5-fold CV with the selected model), so no person is scored by a model that saw them.
Dataset B: the model trained on A, scored on Framingham (never used for training).
Both use the same flag cutoff as the app (the score that gives 75% recall on A's out-of-fold predictions).
Per group: how many people, disease rate, share flagged, recall (true-positive rate), false-positive rate, precision.
Gap = largest minus smallest value across the groups of one attribute (a gap near 0 means similar treatment).
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import preprocessing as pp  # noqa: E402

RESULTS, FIGURES = ROOT / "outputs" / "results", ROOT / "outputs" / "figures"
AGE_BINS = [0, 45, 55, 65, 200]
AGE_LABELS = ["under 45", "45-54", "55-64", "65 and over"]


def group_metrics(y, flag):
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return {"n": len(y), "disease_rate_pct": round(y.mean() * 100, 1), "flagged_pct": round(flag.mean() * 100, 1),
            "recall_pct": round(tp / pos * 100, 1) if pos else np.nan,
            "false_positive_rate_pct": round(fp / neg * 100, 1) if neg else np.nan,
            "precision_pct": round(tp / (tp + fp) * 100, 1) if tp + fp else np.nan}


def main():
    info = json.loads((RESULTS / "model_info.json").read_text())
    best = joblib.load(ROOT / "models" / info["models"][info["best_model"]])
    cutoff = info["cutoffs"][info["best_model"]]["recall75"]
    feats = info["features"]
    rows = []
    for name, f in [("A (out-of-fold)", "dataset_A.csv"), ("B (trained on A)", "dataset_B.csv")]:
        df = pd.read_csv(ROOT / "data" / "processed" / f)
        X, y = pp.get_X(df, feats), df[pp.TARGET].values
        if name.startswith("A"):
            score = cross_val_predict(clone(best), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                      method="predict_proba")[:, 1]
        else:
            score = best.predict_proba(X)[:, 1]
        flag = score >= cutoff
        groups = {"sex": df["sex"].map({0: "female", 1: "male"}),
                  "age group": pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str)}
        for attr, g in groups.items():
            for level in sorted(g.unique(), key=lambda v: (AGE_LABELS.index(v) if v in AGE_LABELS else 0, v)):
                m = (g == level).values
                rows.append({"dataset": name, "attribute": attr, "group": level, **group_metrics(y[m], flag[m])})
    out = pd.DataFrame(rows)
    gaps = []
    for (ds, attr), d in out[out.n >= 30].groupby(["dataset", "attribute"]):
        gaps.append({"dataset": ds, "attribute": attr, "group": "GAP (max - min)", "n": d.n.sum(),
                     "recall_pct": round(d.recall_pct.max() - d.recall_pct.min(), 1),
                     "false_positive_rate_pct": round(d.false_positive_rate_pct.max() - d.false_positive_rate_pct.min(), 1)})
    result = pd.concat([out, pd.DataFrame(gaps)], ignore_index=True)
    result.to_csv(RESULTS / "fairness_by_group.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (ds, d) in zip(axes, out.groupby("dataset", sort=True)):
        d = d.reset_index(drop=True)
        x = np.arange(len(d)); w = 0.38
        ax.bar(x - w / 2, d.recall_pct, w, label="recall (true-positive rate)", color="#4E79A7")
        ax.bar(x + w / 2, d.false_positive_rate_pct, w, label="false-positive rate", color="#F28E2B")
        ax.set_xticks(x); ax.set_xticklabels([f"{a}\n{g}" for a, g in zip(d.attribute, d.group)], fontsize=8)
        ax.set_title(ds); ax.set_ylabel("%"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fairness check: error rates by sex and age group (same flag cutoff as the app)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIGURES / "fairness.png", dpi=150); plt.close(fig)
    return result


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(main().to_string(index=False))
