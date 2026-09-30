"""Metrics, threshold analysis, error analysis, fairness analysis and plots."""
import logging
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from src import config

logger = logging.getLogger(__name__)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> Dict[str, float]:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
    }


def threshold_table(y_true, proba, thresholds=None) -> pd.DataFrame:
    """Effect of each threshold on precision, recall, errors and business cost."""
    thresholds = np.arange(0.05, 0.96, 0.05) if thresholds is None else thresholds
    y_true = np.asarray(y_true)
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": round(float(t), 2),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn),
            "false_positives": fp, "false_negatives": fn,
            "high_risk_count": int(pred.sum()),
            "high_risk_share": pred.mean(),
            "expected_cost": config.FALSE_NEGATIVE_COST * fn + config.FALSE_POSITIVE_COST * fp,
        })
    return pd.DataFrame(rows)


def choose_thresholds(y_true, proba) -> Dict[str, float]:
    """High-risk cut-off minimises expected cost. Low-risk cut-off keeps missed
    defaulters below the allowed miss rate."""
    grid = np.arange(0.01, 0.99, 0.01)
    table = threshold_table(y_true, proba, grid)
    high = float(table.loc[table.expected_cost.idxmin(), "threshold"])

    y_true = np.asarray(y_true)
    defaulter_scores = np.sort(proba[y_true == 1])
    low = float(np.quantile(defaulter_scores, config.LOW_RISK_MISS_RATE))
    if low >= high:
        low = high / 2
    return {"low_risk": round(low, 3), "high_risk": high}


def risk_category(prob: float, thresholds: Dict[str, float]) -> str:
    if prob >= thresholds["high_risk"]:
        return "High Risk"
    if prob >= thresholds["low_risk"]:
        return "Medium Risk"
    return "Low Risk"


def error_analysis(X_test: pd.DataFrame, y_test, proba, threshold: float) -> pd.DataFrame:
    """Labels each test row as TP/TN/FP/FN and adds the predicted probability."""
    df = X_test.copy()
    df["actual"] = np.asarray(y_test)
    df["probability"] = proba
    df["predicted"] = (proba >= threshold).astype(int)
    df["outcome"] = np.select(
        [(df.actual == 1) & (df.predicted == 1), (df.actual == 0) & (df.predicted == 0),
         (df.actual == 0) & (df.predicted == 1)], ["TP", "TN", "FP"], default="FN")
    df["DEBTINC_missing"] = df["DEBTINC"].isna()
    return df


def error_summary(errors: pd.DataFrame) -> pd.DataFrame:
    cols = ["probability", "LOAN", "VALUE", "DEBTINC", "DELINQ", "DEROG", "CLAGE", "NINQ"]
    summary = errors.groupby("outcome")[cols].median().round(2)
    summary.insert(0, "count", errors.outcome.value_counts())
    summary.insert(1, "share_DEBTINC_missing", errors.groupby("outcome").DEBTINC_missing.mean().round(2))
    return summary


def group_metrics(X_test: pd.DataFrame, y_test, proba, threshold: float, column: str) -> pd.DataFrame:
    """Error rates per group. Small groups are flagged because rates are unstable."""
    df = pd.DataFrame({"group": X_test[column].fillna("Missing").values,
                       "y": np.asarray(y_test), "pred": (proba >= threshold).astype(int)})
    rows = []
    for name, g in df.groupby("group"):
        pos, neg = (g.y == 1).sum(), (g.y == 0).sum()
        rows.append({
            "column": column, "group": name, "n": len(g),
            "actual_default_rate": g.y.mean(),
            "flagged_high_risk_rate": g.pred.mean(),
            "recall": ((g.pred == 1) & (g.y == 1)).sum() / pos if pos else np.nan,
            "false_positive_rate": ((g.pred == 1) & (g.y == 0)).sum() / neg if neg else np.nan,
            "small_group": len(g) < 50,
        })
    return pd.DataFrame(rows).round(3)


def _save(name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / name, dpi=110)
    plt.close()


def plot_roc_pr(y_true, probas: Dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
        ax[1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y_true, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].axhline(np.mean(y_true), color="k", ls="--", lw=0.8)
    ax[1].set(title="Precision-recall curve (test set)", xlabel="Recall", ylabel="Precision")
    ax[0].legend(); ax[1].legend()
    _save("roc_pr_curves.png")


def plot_confusion(y_true, proba, threshold: float) -> None:
    cm = confusion_matrix(y_true, (proba >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Repaid", "Default"],
           yticklabels=["Repaid", "Default"], xlabel="Predicted", ylabel="Actual",
           title=f"Confusion matrix (threshold {threshold:.2f})")
    _save("confusion_matrix.png")


def plot_calibration(y_true, probas: Dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in probas.items():
        frac, mean_pred = calibration_curve(y_true, p, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=f"{name} (Brier {brier_score_loss(y_true, p):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
    ax.set(title="Calibration curve (test set)", xlabel="Mean predicted probability",
           ylabel="Observed default rate")
    ax.legend(fontsize=8)
    _save("calibration_curve.png")


def plot_thresholds(table: pd.DataFrame, thresholds: Dict[str, float]) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(table.threshold, table.precision, label="Precision")
    ax[0].plot(table.threshold, table.recall, label="Recall")
    ax[1].plot(table.threshold, table.expected_cost, color="tab:red")
    for a in ax:
        a.axvline(thresholds["high_risk"], color="k", ls="--", label="High-risk threshold")
        a.axvline(thresholds["low_risk"], color="gray", ls=":", label="Low-risk threshold")
        a.set_xlabel("Probability threshold")
    ax[0].set_title("Precision and recall"); ax[0].legend()
    ax[1].set_title("Expected cost (FN x5 + FP x1), cross-validated")
    _save("threshold_analysis.png")


def plot_error_probabilities(errors: pd.DataFrame, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for outcome, color in [("TN", "tab:green"), ("TP", "tab:blue"), ("FP", "tab:orange"), ("FN", "tab:red")]:
        ax.hist(errors.loc[errors.outcome == outcome, "probability"], bins=25, alpha=0.6,
                label=f"{outcome} ({(errors.outcome == outcome).sum()})", color=color)
    ax.axvline(threshold, color="k", ls="--")
    ax.set(title="Predicted probability by outcome (test set)", xlabel="Predicted default probability",
           ylabel="Applicants")
    ax.legend()
    _save("error_probabilities.png")
