"""Global and individual explanations (feature importance, permutation importance, SHAP)."""
import logging
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src import config

logger = logging.getLogger(__name__)

LABELS = {
    "DEBTINC": "Debt-to-income ratio", "DEBTINC_MISSING": "Debt-to-income not reported",
    "DELINQ": "Delinquent credit lines", "DEROG": "Derogatory reports",
    "NEGATIVE_MARKS": "Total delinquencies and derogatory reports",
    "HAS_NEGATIVE_MARKS": "Has any negative credit mark",
    "CLAGE": "Age of oldest credit line (months)", "CLNO": "Number of credit lines",
    "NINQ": "Recent credit inquiries", "INQUIRIES_PER_LINE": "Inquiries per credit line",
    "LOAN": "Loan amount", "LOG_LOAN": "Loan amount", "MORTDUE": "Mortgage still owed",
    "LOG_MORTDUE": "Mortgage still owed", "VALUE": "Property value", "LOG_VALUE": "Property value",
    "VALUE_MISSING": "Property value not reported", "LTV": "Loan-to-value ratio",
    "MORTGAGE_TO_VALUE": "Mortgage-to-value ratio", "TOTAL_LEVERAGE": "Total debt against property value",
    "EQUITY": "Home equity", "YOJ": "Years at current job", "YOJ_MISSING": "Years at job not reported",
    "DEROG_MISSING": "Derogatory reports not reported", "DELINQ_MISSING": "Delinquencies not reported",
    "CLAGE_MISSING": "Credit age not reported", "NINQ_MISSING": "Inquiries not reported",
    "CLNO_MISSING": "Credit line count not reported",
}


def readable(feature: str) -> str:
    """Turn a pipeline feature name such as 'cat__JOB_Sales' into plain language."""
    name = feature.split("__", 1)[-1]
    if name in LABELS:
        return LABELS[name]
    for col, title in [("JOB", "Job"), ("REASON", "Loan reason")]:
        if name.startswith(col + "_"):
            return f"{title}: {name[len(col) + 1:]}"
    return name


def transform_features(pipeline, X: pd.DataFrame):
    """Return (engineered frame, model-ready frame with feature names)."""
    engineered = pipeline.named_steps["features"].transform(X)
    matrix = pipeline.named_steps["prep"].transform(engineered)
    names = pipeline.named_steps["prep"].get_feature_names_out()
    return engineered, pd.DataFrame(matrix, columns=names, index=X.index)


def _build_explainer(pipeline, background: pd.DataFrame):
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        return shap.LinearExplainer(model, background)
    return shap.TreeExplainer(model)


def _positive_class_values(values) -> np.ndarray:
    """SHAP output shape differs between model types; keep the default-class part."""
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values


class Explainer:
    """Wraps a fitted pipeline and explains single applicants with SHAP."""

    def __init__(self, pipeline, background_X: pd.DataFrame):
        self.pipeline = pipeline
        _, background = transform_features(pipeline, background_X)
        self.shap = _build_explainer(pipeline, background)

    def shap_values(self, X: pd.DataFrame):
        engineered, matrix = transform_features(self.pipeline, X)
        values = _positive_class_values(self.shap.shap_values(matrix))
        return engineered, matrix, values

    def explain_one(self, applicant: pd.DataFrame, top_n: int = 4) -> Dict[str, List[dict]]:
        engineered, matrix, values = self.shap_values(applicant)
        contrib = pd.Series(values[0], index=matrix.columns)
        # merge duplicated concepts such as LOAN and LOG_LOAN under one label
        by_label = contrib.groupby([readable(c) for c in contrib.index]).sum()

        def describe(label: str) -> str:
            for col in matrix.columns:
                if readable(col) == label:
                    raw = col.split("__", 1)[-1]
                    if raw in engineered.columns:
                        v = engineered[raw].iloc[0]
                        if pd.isna(v):
                            return "not available"
                        if raw.endswith("_MISSING"):
                            return "yes" if v == 1 else "no"
                        return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"
                    return "yes"
            return ""

        def pack(series):
            return [{"factor": k, "value": describe(k), "impact": round(float(v), 3)}
                    for k, v in series.items()]

        return {"increase": pack(by_label[by_label > 0].sort_values(ascending=False).head(top_n)),
                "decrease": pack(by_label[by_label < 0].sort_values().head(top_n))}


def built_in_importance(pipeline) -> pd.Series:
    model = pipeline.named_steps["model"]
    names = [readable(n) for n in pipeline.named_steps["prep"].get_feature_names_out()]
    values = model.feature_importances_ if hasattr(model, "feature_importances_") else np.abs(model.coef_[0])
    return pd.Series(values, index=names).groupby(level=0).sum().sort_values(ascending=False)


def plot_importances(pipeline, X_test: pd.DataFrame, y_test, n_top: int = 12) -> pd.DataFrame:
    """Saves built-in and permutation importance plots. Permutation importance is
    computed on the original applicant columns using the test set."""
    built_in = built_in_importance(pipeline).head(n_top)[::-1]
    built_in.plot.barh(figsize=(7, 5), title="Model feature importance")
    _save("feature_importance.png")

    result = permutation_importance(pipeline, X_test, y_test, scoring="average_precision",
                                    n_repeats=10, random_state=config.RANDOM_STATE)
    perm = pd.DataFrame({"feature": X_test.columns, "importance": result.importances_mean,
                         "std": result.importances_std}).sort_values("importance", ascending=False)
    top = perm.head(n_top)[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top.feature, top.importance, xerr=top["std"])
    plt.title("Permutation importance (drop in PR-AUC, test set)")
    _save("permutation_importance.png")
    return perm


def plot_shap_summary(explainer: Explainer, X_test: pd.DataFrame) -> pd.DataFrame:
    """Saves a SHAP beeswarm plot and returns mean absolute SHAP per feature."""
    _, matrix, values = explainer.shap_values(X_test)
    display = matrix.copy()
    display.columns = [readable(c) for c in matrix.columns]
    shap.summary_plot(values, display, show=False, max_display=12)
    _save("shap_summary.png")
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=display.columns)
    return mean_abs.groupby(level=0).sum().sort_values(ascending=False).rename("mean_abs_shap").reset_index()


def plot_applicant(explanation: Dict[str, List[dict]], title: str, filename: str) -> None:
    items = explanation["increase"][::-1] + explanation["decrease"][::-1]
    plt.figure(figsize=(7, 4))
    plt.barh([i["factor"] for i in items], [i["impact"] for i in items],
             color=["tab:red" if i["impact"] > 0 else "tab:green" for i in items])
    plt.axvline(0, color="k", lw=0.8)
    plt.title(title, fontsize=10)
    plt.xlabel("Effect on default score (red raises risk, green lowers it)")
    _save(filename)


def _save(name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / name, dpi=110, bbox_inches="tight")
    plt.close()
