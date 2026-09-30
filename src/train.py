"""End-to-end training script. Run from the project root:  python -m src.train"""
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import config, evaluate, explain
from src.calibration import (CalibratedModel, apply_calibrator, cross_fitted_calibration,
                             fit_calibrator)
from src.data_preprocessing import build_pipeline, leakage_check, load_data, split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train")

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "precision": "precision",
           "recall": "recall", "f1": "f1", "balanced_accuracy": "balanced_accuracy",
           "brier": "neg_brier_score"}
SEED = config.RANDOM_STATE


def get_models(pos_weight: float) -> dict:
    """Model name -> (estimator, is_linear)."""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=3000, class_weight="balanced"), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                 class_weight="balanced", random_state=SEED), False),
        "Random Forest": (RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                 class_weight="balanced", n_jobs=-1, random_state=SEED), False),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED), False),
        "XGBoost": (XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                  scale_pos_weight=pos_weight, eval_metric="logloss",
                                  n_jobs=1, random_state=SEED), False),
    }


def cv_scores(pipeline, X, y, cv) -> dict:
    res = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=1)
    out = {k: res[f"test_{k}"].mean() for k in SCORING}
    out["brier"] = -out["brier"]
    out["fit_seconds"] = res["fit_time"].mean()
    return out


def compare_models(models, X, y, cv) -> pd.DataFrame:
    rows = []
    for name, (est, linear) in models.items():
        logger.info("Cross-validating %s", name)
        rows.append({"model": name, **cv_scores(build_pipeline(clone(est), linear), X, y, cv)})
    return pd.DataFrame(rows).round(4)


def imbalance_study(X, y, cv) -> pd.DataFrame:
    """Same model under three imbalance strategies. SMOTE sits inside the pipeline,
    so it is applied to training folds only."""
    setups = {
        "Logistic Regression": (lambda w: LogisticRegression(max_iter=3000, class_weight=w), True),
        "Random Forest": (lambda w: RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight=w,
                                                           n_jobs=-1, random_state=SEED), False),
    }
    rows = []
    for name, (make, linear) in setups.items():
        for strategy in ["none", "class_weight", "smote"]:
            weight = "balanced" if strategy == "class_weight" else None
            sampler = SMOTE(random_state=SEED) if strategy == "smote" else None
            logger.info("Imbalance study: %s / %s", name, strategy)
            pipe = build_pipeline(make(weight), linear, sampler)
            rows.append({"model": name, "strategy": strategy, **cv_scores(pipe, X, y, cv)})
    return pd.DataFrame(rows).round(4)


def tuning_setups(pos_weight: float) -> dict:
    """Name -> (search object factory args). Small searches to keep runtime reasonable."""
    return {
        "Logistic Regression": ("grid", LogisticRegression(max_iter=3000, class_weight="balanced"), True,
                                {"model__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": ("random", RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED), False,
                          {"model__n_estimators": [200, 300], "model__max_depth": [8, 12, None],
                           "model__min_samples_leaf": [1, 3, 5], "model__max_features": ["sqrt", 0.5]}),
        "Gradient Boosting": ("random", GradientBoostingClassifier(random_state=SEED), False,
                              {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.05, 0.1],
                               "model__max_depth": [3, 4, 5], "model__subsample": [0.8, 1.0],
                               "model__min_samples_leaf": [1, 5]}),
        "XGBoost": ("random", XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss",
                                            n_jobs=1, random_state=SEED), False,
                    {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.05, 0.1],
                     "model__max_depth": [3, 5, 7], "model__subsample": [0.8, 1.0],
                     "model__colsample_bytree": [0.7, 1.0], "model__min_child_weight": [1, 3]}),
    }


def tune_models(setups, X, y, X_test, y_test, cv):
    rows, fitted = [], {}
    for name, (kind, est, linear, grid) in setups.items():
        logger.info("Tuning %s", name)
        pipe = build_pipeline(est, linear)
        if kind == "grid":
            search = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, n_jobs=1)
        else:
            search = RandomizedSearchCV(pipe, grid, n_iter=8, scoring="average_precision", cv=cv,
                                        random_state=SEED, n_jobs=1)
        search.fit(X, y)
        test = evaluate.compute_metrics(y_test, search.predict_proba(X_test)[:, 1], 0.5)
        rows.append({"model": name, "cv_pr_auc": search.best_score_,
                     "best_params": json.dumps({k.replace("model__", ""): v for k, v in search.best_params_.items()},
                                               default=str),
                     **{f"test_{k}": v for k, v in test.items()}})
        fitted[name] = search.best_estimator_
    return pd.DataFrame(rows).round(4), fitted


def select_final_model(tuned: pd.DataFrame) -> str:
    """Best CV PR-AUC, but among models within the margin prefer the simpler one."""
    top = tuned.cv_pr_auc.max()
    close = tuned[tuned.cv_pr_auc >= top - config.SELECTION_MARGIN].model.tolist()
    return min(close, key=config.SIMPLICITY_ORDER.index)


def main() -> None:
    start = time.time()
    for folder in (config.MODELS_DIR, config.FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    reports = config.REPORTS_DIR
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=SEED)

    # 1. data and leakage check
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    logger.info("Train %s / Test %s, default rate %.3f / %.3f", X_train.shape, X_test.shape,
                y_train.mean(), y_test.mean())
    leakage_check(X_train, X_test).to_csv(reports / "leakage_check.csv")
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    # 2. logistic regression baseline (test set used for reporting only)
    baseline = build_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced"), True).fit(X_train, y_train)
    base_proba = baseline.predict_proba(X_test)[:, 1]
    base_metrics = evaluate.compute_metrics(y_test, base_proba, 0.5)
    (reports / "baseline_logreg.txt").write_text(
        json.dumps({k: round(v, 4) for k, v in base_metrics.items()}, indent=2) + "\n\n" +
        classification_report(y_test, (base_proba >= 0.5).astype(int), target_names=["Repaid", "Default"]) +
        "\nConfusion matrix [[TN, FP], [FN, TP]]:\n" +
        str(evaluate.confusion_matrix(y_test, (base_proba >= 0.5).astype(int))))

    # 3. model comparison with cross-validation on the training set
    models = get_models(pos_weight)
    comparison = compare_models(models, X_train, y_train, cv)
    comparison.to_csv(reports / "model_results.csv", index=False)
    print(comparison.to_string(index=False))

    # 4. class imbalance study
    imbalance = imbalance_study(X_train, y_train, cv)
    imbalance.to_csv(reports / "imbalance_results.csv", index=False)
    print(imbalance.to_string(index=False))

    # 5. hyperparameter tuning and model selection
    tuned, fitted = tune_models(tuning_setups(pos_weight), X_train, y_train, X_test, y_test, cv)
    tuned.to_csv(reports / "tuned_results.csv", index=False)
    print(tuned.to_string(index=False))
    final_name = select_final_model(tuned)
    logger.info("Selected final model: %s", final_name)
    final_pipe = clone(fitted[final_name])

    # 6. probability calibration (choice made with cross-validated predictions on train)
    oof_raw = cross_val_predict(final_pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    fit_pipe = clone(final_pipe).fit(X_train, y_train)
    test_raw = fit_pipe.predict_proba(X_test)[:, 1]
    cal_rows, cal_oof, cal_test = [], {}, {}
    for method in ["raw", "sigmoid", "isotonic"]:
        cal_oof[method] = cross_fitted_calibration(method, oof_raw, y_train)
        calibrator = None if method == "raw" else fit_calibrator(method, oof_raw, y_train)
        cal_test[method] = test_raw if method == "raw" else apply_calibrator(method, calibrator, test_raw)
        cal_rows.append({
            "method": method,
            "cv_brier": evaluate.brier_score_loss(y_train, cal_oof[method]),
            "cv_log_loss": log_loss(y_train, np.clip(cal_oof[method], 1e-6, 1 - 1e-6)),
            "test_brier": evaluate.brier_score_loss(y_test, cal_test[method]),
            "test_log_loss": log_loss(y_test, np.clip(cal_test[method], 1e-6, 1 - 1e-6)),
        })
    calibration = pd.DataFrame(cal_rows).round(4)
    calibration.to_csv(reports / "calibration_results.csv", index=False)
    print(calibration.to_string(index=False))
    best_method = calibration.loc[calibration.cv_brier.idxmin(), "method"]
    logger.info("Calibration method: %s", best_method)
    evaluate.plot_calibration(y_test, {"Raw": cal_test["raw"], "Sigmoid": cal_test["sigmoid"],
                                       "Isotonic": cal_test["isotonic"]})

    calibrator = None if best_method == "raw" else fit_calibrator(best_method, oof_raw, y_train)
    model = CalibratedModel(fit_pipe, best_method, calibrator)

    # 7. threshold selection on cross-validated calibrated probabilities (train only)
    thresholds = evaluate.choose_thresholds(y_train, cal_oof[best_method])
    table = evaluate.threshold_table(y_train, cal_oof[best_method])
    table.round(4).to_csv(reports / "threshold_analysis.csv", index=False)
    evaluate.plot_thresholds(evaluate.threshold_table(y_train, cal_oof[best_method], np.arange(0.02, 0.9, 0.01)), thresholds)
    print(table.round(3).to_string(index=False))
    logger.info("Thresholds: %s", thresholds)

    # 8. final test evaluation
    proba = model.predict_proba(X_test)[:, 1]
    cut = thresholds["high_risk"]
    final_metrics = evaluate.compute_metrics(y_test, proba, cut)
    at_half = evaluate.compute_metrics(y_test, proba, 0.5)
    (reports / "final_classification_report.txt").write_text(
        classification_report(y_test, (proba >= cut).astype(int), target_names=["Repaid", "Default"]) +
        "\nConfusion matrix [[TN, FP], [FN, TP]]:\n" + str(evaluate.confusion_matrix(y_test, (proba >= cut).astype(int))))
    evaluate.plot_roc_pr(y_test, {"Logistic Regression baseline": base_proba, f"{final_name} (calibrated)": proba})
    evaluate.plot_confusion(y_test, proba, cut)
    print("Final metrics at threshold", cut, {k: round(v, 4) for k, v in final_metrics.items()})

    # 9. explainability
    explainer = explain.Explainer(fit_pipe, X_train.sample(200, random_state=SEED))
    perm = explain.plot_importances(fit_pipe, X_test, y_test)
    perm.round(4).to_csv(reports / "permutation_importance.csv", index=False)
    shap_table = explain.plot_shap_summary(explainer, X_test)
    shap_table.round(4).to_csv(reports / "shap_importance.csv", index=False)
    errors = evaluate.error_analysis(X_test, y_test, proba, cut)
    example_rows = {"caught_default": errors[errors.outcome == "TP"].probability.idxmax(),
                    "missed_default": errors[errors.outcome == "FN"].probability.idxmin(),
                    "safe_applicant": errors[errors.outcome == "TN"].probability.idxmin()}
    examples = {}
    for label, idx in example_rows.items():
        expl = explainer.explain_one(X_test.loc[[idx]])
        p = float(errors.loc[idx, "probability"])
        examples[label] = {"probability": round(p, 4), "risk": evaluate.risk_category(p, thresholds), **expl}
        explain.plot_applicant(expl, f"{label.replace('_', ' ').title()}: default probability {p:.1%}",
                               f"applicant_{label}.png")
    (reports / "example_explanations.json").write_text(json.dumps(examples, indent=2))

    # 10. error analysis and fairness
    errors.to_csv(reports / "test_predictions.csv")
    evaluate.error_summary(errors).to_csv(reports / "error_summary.csv")
    evaluate.plot_error_probabilities(errors, cut)
    fair_X = X_test.copy()
    fair_X["DEBTINC_REPORTED"] = np.where(fair_X.DEBTINC.isna(), "Missing", "Reported")
    fairness = pd.concat([evaluate.group_metrics(fair_X, y_test, proba, cut, c)
                          for c in ["JOB", "REASON", "DEBTINC_REPORTED"]])
    fairness.to_csv(reports / "fairness_results.csv", index=False)

    # 11. save artefacts
    joblib.dump(model, config.MODEL_PATH, compress=3)
    settings = {
        "model_name": final_name, "calibration_method": best_method, "thresholds": thresholds,
        "cost_ratio_fn_to_fp": config.FALSE_NEGATIVE_COST / config.FALSE_POSITIVE_COST,
        "best_params": json.loads(tuned.set_index("model").loc[final_name, "best_params"]),
        "test_metrics": {k: round(v, 4) for k, v in final_metrics.items()},
        "test_metrics_at_0.5": {k: round(v, 4) for k, v in at_half.items()},
        "baseline_test_metrics": {k: round(v, 4) for k, v in base_metrics.items()},
        "input_ranges": {c: [float(X_train[c].min()), float(X_train[c].median()), float(X_train[c].max())]
                         for c in config.NUMERIC_COLS},
        "categories": {c: sorted(X_train[c].dropna().unique().tolist()) for c in config.CATEGORICAL_COLS},
    }
    config.SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    logger.info("Done in %.1f minutes", (time.time() - start) / 60)


if __name__ == "__main__":
    main()
