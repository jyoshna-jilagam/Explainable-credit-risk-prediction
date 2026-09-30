"""Scoring a single applicant with the saved model.  Usage: python -m src.predict"""
import json
import logging
from functools import lru_cache
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src import config, evaluate
from src.data_preprocessing import load_data
from src.explain import Explainer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the saved model, its settings and a SHAP explainer (cached)."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run 'python -m src.train' first.")
    model = joblib.load(config.MODEL_PATH)
    settings = json.loads(config.SETTINGS_PATH.read_text())
    background = load_data()[config.RAW_FEATURES].sample(200, random_state=config.RANDOM_STATE)
    return model, settings, Explainer(model.pipeline, background)


def predict_applicant(applicant: Dict[str, Optional[object]]) -> dict:
    """Return default probability, risk category and the main risk drivers.
    Unknown values may be given as None; the pipeline handles them."""
    model, settings, explainer = load_artifacts()
    row = pd.DataFrame([{col: applicant.get(col, np.nan) for col in config.RAW_FEATURES}])
    row[config.NUMERIC_COLS] = row[config.NUMERIC_COLS].astype(float)

    probability = float(model.predict_proba(row)[0, 1])
    thresholds = settings["thresholds"]
    return {
        "probability": probability,
        "risk_category": evaluate.risk_category(probability, thresholds),
        "flagged_for_review": probability >= thresholds["high_risk"],
        "thresholds": thresholds,
        "explanation": explainer.explain_one(row),
    }


if __name__ == "__main__":
    sample = {"LOAN": 15000, "MORTDUE": 60000, "VALUE": 85000, "REASON": "DebtCon", "JOB": "Other",
              "YOJ": 5, "DEROG": 1, "DELINQ": 2, "CLAGE": 90, "NINQ": 3, "CLNO": 15, "DEBTINC": None}
    print(json.dumps(predict_applicant(sample), indent=2))
