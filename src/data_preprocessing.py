"""Loading, splitting and preprocessing. All fitted steps live inside a pipeline,
so they are learned from training folds only."""
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.feature_engineering import FeatureEngineer, numeric_columns_after_fe

logger = logging.getLogger(__name__)


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the HMEQ csv and drop exact duplicate rows."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Dataset not found at {path}. Place hmeq.csv in the data folder.") from err
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Loaded %d rows (%d duplicates removed)", len(df), before - len(df))
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split with a fixed seed."""
    X, y = df[config.RAW_FEATURES], df[config.TARGET]
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y,
                            random_state=config.RANDOM_STATE)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clips each numeric column to bounds learned on the training data.
    Used only for the linear model, which is sensitive to extreme values."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


def build_preprocessor(linear_model: bool = False) -> ColumnTransformer:
    """Median imputation for numbers, 'Missing' category + one-hot for text columns.
    Linear models additionally get outlier clipping and scaling."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if linear_model:
        steps += [("clip", QuantileClipper()), ("scale", StandardScaler())]
    numeric = Pipeline(steps)

    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, numeric_columns_after_fe()),
        ("cat", categorical, config.CATEGORICAL_COLS),
    ])


def build_pipeline(model, linear_model: bool = False, sampler: Optional[object] = None) -> Pipeline:
    """features -> preprocessing -> (optional resampler) -> model."""
    steps = [("features", FeatureEngineer()), ("prep", build_preprocessor(linear_model))]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return Pipeline(steps)


def leakage_check(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Shows that the imputation values learned by the pipeline come from the
    training set only, and differ from what a full-data fit would learn."""
    fe = FeatureEngineer()
    cols = config.NUMERIC_COLS
    train_medians = fe.transform(X_train)[cols].median()
    full_medians = fe.transform(pd.concat([X_train, X_test]))[cols].median()

    prep = build_preprocessor().fit(fe.transform(X_train))
    learned = prep.named_transformers_["num"].named_steps["impute"].statistics_[:len(cols)]

    result = pd.DataFrame({"pipeline_median": learned, "train_only_median": train_medians,
                           "train_plus_test_median": full_medians}, index=cols)
    result["uses_train_only"] = np.isclose(result.pipeline_median, result.train_only_median)
    return result.round(3)
