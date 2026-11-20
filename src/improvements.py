"""Improvements on top of notebook 02: a soft-voting ensemble, a class-imbalance experiment and a
baseline-vs-improved summary.

Run after notebook 02 (it reads the per-fold settings and the comparison table that notebook 02 writes):

    python src/improvements.py

Notebook 03 calls the same three steps one by one and explains the results.

1. Soft-voting ensemble (Exp 4)
   The average of the predicted probabilities of Logistic Regression, Random Forest and Gradient Boosting.
   It goes through the same nested 5-fold CV on A as the single models: in each outer fold every member uses the
   setting that the inner search chose in that fold (read from exp1_cv_fold_metrics.csv), so no outer test fold
   is ever used for tuning. Then it is trained on all of A and tested on B, like the other models (Exp 2).
   The ensemble is added to the model selection, with the same rule as notebook 02.

2. Class-imbalance experiment (Exp 5)
   Framingham (B) has 15.2% positives. Inside Framingham's 5-fold CV (the Exp 3 setting) each model is trained
   with: no treatment, class_weight="balanced", and SMOTE (synthetic positives, made inside the training folds only).
   The test folds are never resampled.

3. Baseline vs improved summary
   One table and one chart: baseline Logistic Regression against the tuned models, the ensemble and the best
   imbalance treatment, with the gain over the baseline.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import preprocessing as pp  # noqa: E402

warnings.filterwarnings("ignore")
FIG = ROOT / "outputs" / "figures"
RES = ROOT / "outputs" / "results"
MODELS = ROOT / "models"

SEED = 42
SPEC_TARGET = 0.70
RECALL_TARGET = 0.75
ENSEMBLE = "Soft-voting ensemble"
BASE_MODELS = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
COLORS = {"Logistic Regression": "#4E79A7", "Random Forest": "#59A14F", "Gradient Boosting": "#E15759",
          ENSEMBLE: "#B07AA1", "none": "#9AA0A6", "class_weight": "#F28E2B", "SMOTE": "#76B7B2"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


# ---------------------------------------------------------------- shared helpers
def log(msg):
    print(msg, flush=True)


def load_data():
    a = pd.read_csv(ROOT / "data" / "processed" / "dataset_A.csv")
    b = pd.read_csv(ROOT / "data" / "processed" / "dataset_B.csv")
    return a, b


def base_estimator(name, class_weight=None):
    """The three model types used in notebook 02 (same fixed settings)."""
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight=class_weight)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=150, n_jobs=2, random_state=SEED, class_weight=class_weight)
    return HistGradientBoostingClassifier(random_state=SEED, class_weight=class_weight)


GRIDS = {
    "Logistic Regression": {"clf__C": [0.1, 1.0, 10.0]},
    "Random Forest": {"clf__max_depth": [6, 10], "clf__min_samples_leaf": [20, 100]},
    "Gradient Boosting": {"clf__max_leaf_nodes": [8, 31], "clf__learning_rate": [0.05, 0.1]},
}


def member_pipeline(name, params):
    """One pipeline (impute -> one-hot -> scale for LR -> model) with the given clf__ settings."""
    pipe = pp.build_pipeline(base_estimator(name), scale=(name == "Logistic Regression"))
    return pipe.set_params(**params)


def voting_ensemble(params_by_model):
    return VotingClassifier([(n.split()[0].lower(), member_pipeline(n, params_by_model[n])) for n in BASE_MODELS],
                            voting="soft", n_jobs=1)


def recall_at_specificity(y, p, spec=SPEC_TARGET):
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[fpr <= 1 - spec].max())


def cutoff_for_recall(y, p, target=RECALL_TARGET):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr >= target)])


def youden_cutoff(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def at_cutoff(y, p, cutoff):
    pred = (np.asarray(p) >= cutoff).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    recall = tp / (tp + fn); spec = tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
    return {"recall": recall, "specificity": spec, "precision": prec, "f1": f1,
            "balanced_accuracy": balanced_accuracy_score(y, pred), "flagged_pct": pred.mean() * 100}


def boot_ci(y, p, fn, n=500, seed=SEED):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fn(y[i], p[i]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


# ---------------------------------------------------------------- 1. soft-voting ensemble
def run_ensemble(a, b):
    """Exp 4: nested CV on A (fold settings from notebook 02), then train on all of A and test on B."""
    t0 = time.time()
    XA, yA = pp.get_X(a), a[pp.TARGET].values
    XB, yB = pp.get_X(b), b[pp.TARGET].values
    folds = pd.read_csv(RES / "exp1_cv_fold_metrics.csv")
    info = json.loads((RES / "model_info.json").read_text())

    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)  # the same outer folds as notebook 02
    oof = np.zeros(len(a)); rows, fold_models = [], []
    for f, (tr, te) in enumerate(outer.split(XA, yA), 1):
        params = {n: json.loads(folds[(folds.model == n) & (folds.fold == f)].best_params.iloc[0]) for n in BASE_MODELS}
        ens = voting_ensemble(params).fit(XA.iloc[tr], yA[tr])
        p = ens.predict_proba(XA.iloc[te])[:, 1]
        oof[te] = p
        rows.append({"model": ENSEMBLE, "fold": f, "roc_auc": roc_auc_score(yA[te], p),
                     "pr_auc": average_precision_score(yA[te], p), "recall_at_70spec": recall_at_specificity(yA[te], p)})
        fold_models.append((ens, te))
        log(f"ensemble fold {f}: AUC {rows[-1]['roc_auc']:.4f}, recall@70spec {rows[-1]['recall_at_70spec']:.4f} ({time.time()-t0:.0f}s)")
    ens_folds = pd.DataFrame(rows)

    # final ensemble: members use the settings notebook 02 chose on all of A
    final = voting_ensemble(info["best_params"]).fit(XA, yA)
    pB = final.predict_proba(XB)[:, 1]
    cut = {"recall75": cutoff_for_recall(yA, oof), "youden": youden_cutoff(yA, oof)}
    lo, hi = boot_ci(yB, pB, roc_auc_score)
    test_B = {"model": ENSEMBLE, "B_roc_auc": roc_auc_score(yB, pB), "B_roc_auc_ci_low": lo, "B_roc_auc_ci_high": hi,
              "B_pr_auc": average_precision_score(yB, pB), "B_recall_at_70spec": recall_at_specificity(yB, pB),
              **{f"B_{k}_at_A_cutoff": v for k, v in at_cutoff(yB, pB, cut["recall75"]).items()}}
    log(f"ensemble trained on all of A; on B: AUC {test_B['B_roc_auc']:.4f} ({time.time()-t0:.0f}s)")

    # permutation importance on held-out folds, as in notebook 02 (for the app)
    imp = []
    for ens, te in fold_models:
        r = permutation_importance(ens, XA.iloc[te], yA[te], scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
        imp.append(r.importances_mean)
    imp = np.array(imp)
    importance = pd.DataFrame({"model": ENSEMBLE, "feature": pp.SHARED,
                               "importance_mean": imp.mean(axis=0), "importance_sd": imp.std(axis=0, ddof=1)})
    log(f"ensemble importance done ({time.time()-t0:.0f}s)")
    return {"folds": ens_folds, "oof": oof, "pred_B": pB, "test_B": test_B, "cutoffs": cut,
            "model": final, "importance": importance}


def update_selection(ens):
    """Add the ensemble to the comparison table and re-run notebook 02's selection rule with it included."""
    folds = pd.read_csv(RES / "exp1_cv_fold_metrics.csv")
    allf = pd.concat([folds[["model", "fold", "roc_auc", "recall_at_70spec"]], ens["folds"][["model", "fold", "roc_auc", "recall_at_70spec"]]])
    sel = allf.groupby("model").agg(recall70_mean=("recall_at_70spec", "mean"), recall70_sd=("recall_at_70spec", "std"),
                                    auc_mean=("roc_auc", "mean"), auc_sd=("roc_auc", "std"))
    best_recall = sel["recall70_mean"].max()
    sel["tied_with_best"] = sel["recall70_mean"] >= best_recall - 0.02
    best = sel[sel["tied_with_best"]]["auc_mean"].idxmax()
    sel["selected"] = sel.index == best
    return sel, best


def save_ensemble(ens, sel, best):
    """Write the ensemble's results, update model_info.json and the comparison table, save the model."""
    ens["folds"].round(5).to_csv(RES / "exp4_ensemble_cv_fold_metrics.csv", index=False)
    ens["importance"].round(5).to_csv(RES / "feature_importance_ensemble.csv", index=False)
    joblib.dump(ens["model"], MODELS / "soft_voting_ensemble.joblib")

    info = json.loads((RES / "model_info.json").read_text())
    info["models"][ENSEMBLE] = "soft_voting_ensemble.joblib"
    info["cutoffs"][ENSEMBLE] = ens["cutoffs"]
    info["selection_table"] = json.loads(sel.round(5).reset_index().to_json(orient="records"))
    info["selection_note"] = "the soft-voting ensemble (notebook 03) is included in the selection with the same rule"
    if info["best_model"] != best:
        info["best_model_before_ensemble"] = info["best_model"]
    info["best_model"] = best
    (RES / "model_info.json").write_text(json.dumps(info, indent=2))
    joblib.dump(joblib.load(MODELS / info["models"][best]), MODELS / "best_model.joblib")

    comp = pd.read_csv(RES / "comparison_table.csv")
    comp = comp[comp.model != ENSEMBLE]
    f = ens["folds"]; t = ens["test_B"]
    row = {"model": ENSEMBLE, "A_cv_recall_at_70spec": f.recall_at_70spec.mean(), "A_cv_roc_auc": f.roc_auc.mean(),
           "A_cv_pr_auc": f.pr_auc.mean(), "B_roc_auc": t["B_roc_auc"], "B_roc_auc_ci_low": t["B_roc_auc_ci_low"],
           "B_roc_auc_ci_high": t["B_roc_auc_ci_high"], "B_pr_auc": t["B_pr_auc"], "B_pr_auc_baseline": comp.B_pr_auc_baseline.iloc[0],
           "B_recall_at_70spec": t["B_recall_at_70spec"]}
    rule_rows = comp[comp.model.str.startswith("Rule")]
    comp = pd.concat([comp[~comp.model.str.startswith("Rule")], pd.DataFrame([row]), rule_rows], ignore_index=True)
    comp["selected"] = comp.model == best
    comp.round(4).to_csv(RES / "comparison_table.csv", index=False)
    return comp


# ---------------------------------------------------------------- 2. class-imbalance experiment
STRATEGIES = ["none", "class_weight", "SMOTE"]


def imbalance_pipeline(name, strategy):
    if strategy == "class_weight":
        return pp.build_pipeline(base_estimator(name, class_weight="balanced"), scale=(name == "Logistic Regression"))
    pipe = pp.build_pipeline(base_estimator(name), scale=(name == "Logistic Regression"))
    if strategy == "none":
        return pipe
    # SMOTE goes after imputation/encoding and before the model; imblearn applies it to training data only
    return ImbPipeline([("prep", pipe.named_steps["prep"]), ("smote", SMOTE(random_state=SEED)), ("clf", pipe.named_steps["clf"])])


def run_imbalance(b):
    """Exp 5: Framingham 5-fold CV (inner 3-fold grid search), each model x each imbalance strategy."""
    t0 = time.time()
    XB, yB = pp.get_X(b), b[pp.TARGET].values
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)  # the same folds as Exp 3
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    oof = {(m, s): np.zeros(len(b)) for m in BASE_MODELS for s in STRATEGIES}
    for f, (tr, te) in enumerate(outer.split(XB, yB), 1):
        for m in BASE_MODELS:
            for s in STRATEGIES:
                search = GridSearchCV(imbalance_pipeline(m, s), GRIDS[m], scoring="roc_auc", cv=inner, n_jobs=1)
                search.fit(XB.iloc[tr], yB[tr])
                oof[(m, s)][te] = search.predict_proba(XB.iloc[te])[:, 1]
        log(f"imbalance fold {f} done ({time.time()-t0:.0f}s)")
    rows = []
    for (m, s), p in oof.items():
        rows.append({"model": m, "strategy": s, "roc_auc": roc_auc_score(yB, p), "pr_auc": average_precision_score(yB, p),
                     "pr_auc_baseline": yB.mean(), **at_cutoff(yB, p, 0.5)})
    res = pd.DataFrame(rows)
    res.round(4).to_csv(RES / "exp5_imbalance_framingham.csv", index=False)
    return res


def plot_imbalance(res):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    x = np.arange(len(BASE_MODELS)); w = 0.26
    panels = [("recall", "Recall at the default 0.5 cutoff"), ("f1", "F1 at the default 0.5 cutoff"), ("roc_auc", "ROC-AUC (no cutoff)")]
    for ax, (col, title) in zip(axes, panels):
        for i, s in enumerate(STRATEGIES):
            vals = [res[(res.model == m) & (res.strategy == s)][col].iloc[0] for m in BASE_MODELS]
            ax.bar(x + (i - 1) * w, vals, w, color=COLORS[s], label=s)
            for xi, v in zip(x + (i - 1) * w, vals):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(BASE_MODELS, fontsize=9); ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1 if col != "roc_auc" else 0.85)
    axes[0].legend(frameon=False, title="imbalance treatment", fontsize=8)
    fig.suptitle("Exp 5: class imbalance in Framingham (15.2% positive), 5-fold CV", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "imbalance_experiment.png", dpi=130, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------- 3. baseline vs improved
def build_summary(comp, imb):
    """One row per step, with the gain over the baseline (Logistic Regression)."""
    exp3 = pd.read_csv(RES / "exp3_framingham_reference.csv")
    base = comp[comp.model == "Logistic Regression"].iloc[0]
    rows = []
    for m in BASE_MODELS + [ENSEMBLE]:
        r = comp[comp.model == m].iloc[0]
        step = "Baseline" if m == "Logistic Regression" else ("Ensembling" if m == ENSEMBLE else "Tuned non-linear model")
        rows.append({"step": step, "model": m,
                     "A_cv_roc_auc": r.A_cv_roc_auc, "A_gain_vs_baseline": r.A_cv_roc_auc - base.A_cv_roc_auc,
                     "A_cv_recall_at_70spec": r.A_cv_recall_at_70spec,
                     "B_roc_auc": r.B_roc_auc, "B_gain_vs_baseline": r.B_roc_auc - base.B_roc_auc, "B_pr_auc": r.B_pr_auc})
    summary = pd.DataFrame(rows)

    none_lr = imb[(imb.model == "Logistic Regression") & (imb.strategy == "none")].iloc[0]
    imb_rows = []
    for _, r in imb.iterrows():
        imb_rows.append({"model": r.model, "strategy": r.strategy, "recall_at_0.5": r.recall, "f1_at_0.5": r.f1,
                         "balanced_accuracy": r.balanced_accuracy, "roc_auc": r.roc_auc,
                         "recall_gain_vs_baseline": r.recall - none_lr.recall, "f1_gain_vs_baseline": r.f1 - none_lr.f1})
    imb_summary = pd.DataFrame(imb_rows)
    summary.round(4).to_csv(RES / "baseline_vs_improved.csv", index=False)
    imb_summary.round(4).to_csv(RES / "baseline_vs_improved_imbalance.csv", index=False)
    return summary, imb_summary, exp3


def plot_summary(summary, imb):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    labels = ["LR\n(baseline)", "Random\nForest", "Gradient\nBoosting", "Soft-voting\nensemble"]
    cols = [COLORS[m] for m in summary.model]
    for ax, col, title, lim in [(axes[0], "A_cv_roc_auc", "A: nested 5-fold CV, ROC-AUC", (0.75, 0.82)),
                                (axes[1], "B_roc_auc", "Trained on A, tested on B, ROC-AUC", (0.6, 0.72))]:
        vals = summary[col].values
        ax.bar(labels, vals, color=cols)
        ax.axhline(vals[0], color="#555555", linestyle="--", linewidth=1, label="baseline")
        for i, v in enumerate(vals):
            gain = v - vals[0]
            ax.text(i, v + (lim[1] - lim[0]) * 0.03, f"{v:.3f}" + ("" if i == 0 else f"\n({gain:+.3f})"), ha="center", fontsize=8)
        ax.set_ylim(*lim); ax.set_title(title, fontsize=10); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[2]
    lr = imb[imb.model == "Logistic Regression"].set_index("strategy").loc[STRATEGIES]
    vals = list(lr.recall)
    f1s = list(lr.f1)
    lab = ["LR, no treatment\n(baseline)", "LR +\nclass_weight", "LR +\nSMOTE"]
    x = np.arange(len(lab))
    ax.bar(x - 0.18, vals, 0.36, color="#4E79A7", label="recall at 0.5")
    ax.bar(x + 0.18, f1s, 0.36, color="#F28E2B", label="F1 at 0.5")
    for xi, v, f in zip(x, vals, f1s):
        ax.text(xi - 0.18, v + 0.015, f"{v:.2f}", ha="center", fontsize=8); ax.text(xi + 0.18, f + 0.015, f"{f:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.set_ylim(0, 1)
    ax.set_title("Framingham (15% positive): imbalance treatment", fontsize=10); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Baseline vs improved", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "baseline_vs_improved.png", dpi=130, bbox_inches="tight")
    return fig


def plot_roc_with_ensemble(a, b, ens):
    """ROC curves of the three models and the ensemble, on A (out-of-fold) and on B."""
    import joblib as jl
    info = json.loads((RES / "model_info.json").read_text())
    XB, yB = pp.get_X(b), b[pp.TARGET].values
    yA = a[pp.TARGET].values
    folds = pd.read_csv(RES / "exp1_cv_fold_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name in BASE_MODELS:
        p = jl.load(MODELS / info["models"][name]).predict_proba(XB)[:, 1]
        fpr, tpr, _ = roc_curve(yB, p)
        axes[1].plot(fpr, tpr, color=COLORS[name], label=f"{name} (AUC {roc_auc_score(yB, p):.3f})")
        axes[0].plot([], [], color=COLORS[name], label=f"{name} (mean fold AUC {folds[folds.model == name].roc_auc.mean():.3f})")
    fpr, tpr, _ = roc_curve(yA, ens["oof"])
    axes[0].plot(fpr, tpr, color=COLORS[ENSEMBLE], linewidth=2, label=f"{ENSEMBLE} (out-of-fold AUC {roc_auc_score(yA, ens['oof']):.3f})")
    fpr, tpr, _ = roc_curve(yB, ens["pred_B"])
    axes[1].plot(fpr, tpr, color=COLORS[ENSEMBLE], linewidth=2, label=f"{ENSEMBLE} (AUC {roc_auc_score(yB, ens['pred_B']):.3f})")
    for ax, t in zip(axes, ["A: ensemble out-of-fold (nested CV)", "Trained on A, tested on B"]):
        ax.plot([0, 1], [0, 1], color="#BBBBBB", linestyle="--")
        ax.set_xlabel("false positive rate"); ax.set_ylabel("recall"); ax.set_title(t, fontsize=10); ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "roc_ensemble.png", dpi=130, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------- run everything
def main():
    a, b = load_data()
    ens = run_ensemble(a, b)
    sel, best = update_selection(ens)
    comp = save_ensemble(ens, sel, best)
    log(f"selected model with the ensemble included: {best}")
    plot_roc_with_ensemble(a, b, ens)
    imb = run_imbalance(b)
    plot_imbalance(imb)
    summary, imb_summary, _ = build_summary(comp, imb)
    plot_summary(summary, imb)
    log(summary.round(3).to_string(index=False))
    log(imb_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
