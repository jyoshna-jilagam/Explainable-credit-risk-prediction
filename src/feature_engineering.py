"""Feature engineering. The transformer is stateless (nothing is learned from data),
so it can be applied safely before or after the train/test split."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src import config

ENGINEERED_COLS = [
    "LTV", "MORTGAGE_TO_VALUE", "TOTAL_LEVERAGE", "EQUITY", "NEGATIVE_MARKS",
    "HAS_NEGATIVE_MARKS", "INQUIRIES_PER_LINE", "LOG_LOAN", "LOG_MORTDUE", "LOG_VALUE",
]
FLAG_COLS = [f"{c}_MISSING" for c in config.MISSING_FLAG_COLS]


def numeric_columns_after_fe() -> list:
    return config.NUMERIC_COLS + ENGINEERED_COLS + FLAG_COLS


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den).replace([np.inf, -np.inf], np.nan)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds credit-risk features and missingness indicators to the raw data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for col in config.MISSING_FLAG_COLS:
            df[f"{col}_MISSING"] = df[col].isna().astype(int)

        df["LTV"] = _ratio(df["LOAN"], df["VALUE"])
        df["MORTGAGE_TO_VALUE"] = _ratio(df["MORTDUE"], df["VALUE"])
        df["TOTAL_LEVERAGE"] = _ratio(df["LOAN"] + df["MORTDUE"], df["VALUE"])
        df["EQUITY"] = df["VALUE"] - df["MORTDUE"]

        df["NEGATIVE_MARKS"] = df[["DEROG", "DELINQ"]].sum(axis=1, min_count=1)
        df["HAS_NEGATIVE_MARKS"] = (df["NEGATIVE_MARKS"] > 0).astype(int)
        df["INQUIRIES_PER_LINE"] = _ratio(df["NINQ"], df["CLNO"] + 1)

        for col in ["LOAN", "MORTDUE", "VALUE"]:
            df[f"LOG_{col}"] = np.log1p(df[col])
        return df
