"""Project-wide constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "hmeq.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "credit_risk_pipeline.pkl"
SETTINGS_PATH = MODELS_DIR / "model_settings.json"

TARGET = "BAD"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

NUMERIC_COLS = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ",
                "CLAGE", "NINQ", "CLNO", "DEBTINC"]
CATEGORICAL_COLS = ["REASON", "JOB"]
RAW_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

# columns whose "was missing" status is turned into a feature (chosen from EDA)
MISSING_FLAG_COLS = ["VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]

# business assumption: approving a defaulter costs 5x more than rejecting a good applicant
FALSE_NEGATIVE_COST = 5.0
FALSE_POSITIVE_COST = 1.0
# low-risk band: at most this share of real defaulters may fall below the low threshold
LOW_RISK_MISS_RATE = 0.05

# if several tuned models are within this PR-AUC margin, the simpler one is preferred
SELECTION_MARGIN = 0.01
SIMPLICITY_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest",
                    "Gradient Boosting", "XGBoost"]
