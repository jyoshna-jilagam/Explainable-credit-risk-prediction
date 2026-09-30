"""Probability calibration helpers (sigmoid and isotonic)."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src import config


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(method: str, scores: np.ndarray, y: np.ndarray):
    """Learn a mapping from raw model scores to calibrated probabilities."""
    if method == "sigmoid":
        return LogisticRegression(C=1e6).fit(_logit(scores), y)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(scores, y)
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(method: str, calibrator, scores: np.ndarray) -> np.ndarray:
    if method == "sigmoid":
        return calibrator.predict_proba(_logit(scores))[:, 1]
    if method == "isotonic":
        return calibrator.predict(scores)
    return scores


def cross_fitted_calibration(method: str, scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calibrated probabilities for every row, each produced by a calibrator that
    did not see that row. Gives an honest estimate of calibration quality."""
    if method == "raw":
        return scores
    out = np.zeros(len(scores))
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    y = np.asarray(y)
    for fit_idx, val_idx in skf.split(scores, y):
        cal = fit_calibrator(method, scores[fit_idx], y[fit_idx])
        out[val_idx] = apply_calibrator(method, cal, scores[val_idx])
    return out


class CalibratedModel:
    """Fitted pipeline plus a probability calibrator. Exposes predict_proba."""

    def __init__(self, pipeline, method: str, calibrator):
        self.pipeline = pipeline
        self.method = method
        self.calibrator = calibrator

    def predict_proba(self, X) -> np.ndarray:
        raw = self.pipeline.predict_proba(X)[:, 1]
        p = apply_calibrator(self.method, self.calibrator, raw)
        return np.column_stack([1 - p, p])
