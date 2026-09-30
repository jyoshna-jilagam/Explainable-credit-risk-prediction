"""Streamlit app: applicant scoring and model evaluation.  Run: streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

from src import config
from src.predict import load_artifacts, predict_applicant

st.set_page_config(page_title="Credit Risk Model", layout="wide")
FIG = config.FIGURES_DIR
REPORTS = config.REPORTS_DIR

if not config.MODEL_PATH.exists():
    st.error("Model file not found. Run `python -m src.train` first.")
    st.stop()

_, settings, _ = load_artifacts()
page = st.sidebar.radio("Section", ["Applicant Scoring", "Model Evaluation"])
st.sidebar.caption("Educational project. Not a production lending system.")


def number_input(label: str, column: str, integer: bool = False, optional: bool = False):
    low, mid, high = settings["input_ranges"][column]
    left, right = st.columns([3, 1])
    unknown = right.checkbox("Unknown", key=f"unk_{column}") if optional else False
    if unknown:
        left.caption(f"{label}: not provided")
        return None
    if integer:
        return left.number_input(label, int(low), int(high * 2), int(mid), key=column)
    return left.number_input(label, float(low), float(high * 2), float(mid), key=column)


def scoring_page() -> None:
    st.title("Loan Default Risk")
    st.write("Enter applicant details. Fields marked Unknown can be left blank, "
             "but a missing value is itself treated as a risk signal by the model.")
    col_a, col_b = st.columns(2)
    with col_a:
        loan = number_input("Requested loan amount", "LOAN", True)
        mortdue = number_input("Mortgage amount still owed", "MORTDUE", True, True)
        value = number_input("Current property value", "VALUE", True, True)
        reason = st.selectbox("Loan reason", settings["categories"]["REASON"])
        job = st.selectbox("Job category", settings["categories"]["JOB"])
        yoj = number_input("Years at current job", "YOJ", False, True)
    with col_b:
        derog = number_input("Major derogatory reports", "DEROG", True, True)
        delinq = number_input("Delinquent credit lines", "DELINQ", True, True)
        clage = number_input("Age of oldest credit line (months)", "CLAGE", False, True)
        ninq = number_input("Recent credit inquiries", "NINQ", True, True)
        clno = number_input("Number of credit lines", "CLNO", True, True)
        debtinc = number_input("Debt-to-income ratio (%)", "DEBTINC", False, True)

    if not st.button("Assess risk", type="primary"):
        return
    result = predict_applicant({
        "LOAN": loan, "MORTDUE": mortdue, "VALUE": value, "REASON": reason, "JOB": job, "YOJ": yoj,
        "DEROG": derog, "DELINQ": delinq, "CLAGE": clage, "NINQ": ninq, "CLNO": clno, "DEBTINC": debtinc})

    prob, category, limits = result["probability"], result["risk_category"], result["thresholds"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Default probability", f"{prob:.1%}")
    c2.metric("Risk category", category)
    c3.metric("Review threshold", f"{limits['high_risk']:.0%}")
    st.progress(min(prob, 1.0))
    st.caption(f"Low risk is below {limits['low_risk']:.0%}, high risk is {limits['high_risk']:.0%} or above. "
               "Both cut-offs come from the threshold analysis on training data.")

    st.subheader("Why this result")
    left, right = st.columns(2)
    for column, key, title in [(left, "increase", "Factors raising risk"),
                               (right, "decrease", "Factors lowering risk")]:
        column.markdown(f"**{title}**")
        items = result["explanation"][key]
        if not items:
            column.write("None of note")
        for item in items:
            column.write(f"- {item['factor']} (value: {item['value']}, effect {item['impact']:+.2f})")
    st.caption("Effects are SHAP values on the model's internal score. Larger numbers mean a stronger push.")


def evaluation_page() -> None:
    st.title("Model Evaluation")
    m = settings["test_metrics"]
    st.write(f"Final model: **{settings['model_name']}**, calibrated with **{settings['calibration_method']}** scaling. "
             f"Metrics below are from the untouched 20% test set at the chosen threshold "
             f"({settings['thresholds']['high_risk']:.2f}).")
    cols = st.columns(5)
    for col, key in zip(cols, ["roc_auc", "pr_auc", "precision", "recall", "brier"]):
        col.metric(key.replace("_", " ").upper(), f"{m[key]:.3f}")

    tabs = st.tabs(["Model comparison", "Curves", "Calibration", "Thresholds", "Explainability", "Errors and fairness"])
    with tabs[0]:
        st.caption("Five-fold cross-validation on the training set")
        st.dataframe(pd.read_csv(REPORTS / "model_results.csv"), hide_index=True)
        st.caption("Tuned models (selection by cross-validated PR-AUC)")
        st.dataframe(pd.read_csv(REPORTS / "tuned_results.csv"), hide_index=True)
        st.caption("Class imbalance strategies")
        st.dataframe(pd.read_csv(REPORTS / "imbalance_results.csv"), hide_index=True)
    with tabs[1]:
        st.image(str(FIG / "roc_pr_curves.png"))
        st.image(str(FIG / "confusion_matrix.png"), width=380)
    with tabs[2]:
        st.image(str(FIG / "calibration_curve.png"), width=480)
        st.dataframe(pd.read_csv(REPORTS / "calibration_results.csv"), hide_index=True)
    with tabs[3]:
        st.image(str(FIG / "threshold_analysis.png"))
        st.dataframe(pd.read_csv(REPORTS / "threshold_analysis.csv"), hide_index=True)
    with tabs[4]:
        a, b = st.columns(2)
        a.image(str(FIG / "feature_importance.png"))
        b.image(str(FIG / "permutation_importance.png"))
        st.image(str(FIG / "shap_summary.png"), width=650)
    with tabs[5]:
        st.image(str(FIG / "error_probabilities.png"), width=560)
        st.dataframe(pd.read_csv(REPORTS / "error_summary.csv"), hide_index=True)
        st.caption("Error rates by group (groups under 50 applicants are unreliable)")
        st.dataframe(pd.read_csv(REPORTS / "fairness_results.csv"), hide_index=True)


scoring_page() if page == "Applicant Scoring" else evaluation_page()
