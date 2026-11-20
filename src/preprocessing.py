"""Reusable pieces for the cardiovascular risk-screening project.

Contents
--------
* column groups for the 8 shared attributes
* cleaning step functions for the Cardio data (one rule at a time, each returns a log entry)
* cholesterol / glucose binning: clinical bands, and a quantile-matched version (experiment 2c)
* the model pipeline builder (imputation fitted on dataset A only, one-hot for levels,
  scaling only when asked for, i.e. for Logistic Regression)
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------------------
# Column groups
# --------------------------------------------------------------------------------------
NUMERIC = ["age", "sysbp", "diabp", "bmi"]
BINARY = ["sex", "smoker"]  # sex: 1 = male, 0 = female
LEVELS = ["chol_level", "gluc_level"]  # 1 = normal, 2 = above normal, 3 = well above normal
SHARED = ["age", "sex", "sysbp", "diabp", "bmi", "smoker", "chol_level", "gluc_level"]
TARGET = "target"

# Clinical bands in mg/dl (lower edge of level 2 and of level 3).
# Cholesterol: <200 desirable, 200-239 borderline high, >=240 high (NCEP ATP III, JAMA 2001;285:2486-2497).
# Glucose: <100 normal, 100-125 prediabetes, >=126 diabetes range (ADA Standards of Care, Section 2; these are fasting values,
# and whether Framingham's glucose was fasting is not documented in the file).
CHOL_CUTOFFS = (200.0, 240.0)
GLUC_CUTOFFS = (100.0, 126.0)

# Plausibility limits used to remove impossible Cardio rows (one rule at a time).
PLAUSIBLE = [
    ("height outside 130-210 cm", lambda d: ~d["height"].between(130, 210)),
    ("weight outside 35-200 kg", lambda d: ~d["weight"].between(35, 200)),
    ("BMI outside 14-60", lambda d: ~d["bmi"].between(14, 60)),
    ("systolic BP outside 70-250 mmHg", lambda d: ~d["ap_hi"].between(70, 250)),
    ("diastolic BP outside 40-150 mmHg", lambda d: ~d["ap_lo"].between(40, 150)),
    ("systolic not above diastolic", lambda d: d["ap_hi"] <= d["ap_lo"]),
]


# --------------------------------------------------------------------------------------
# Cleaning step functions (Cardio)
# --------------------------------------------------------------------------------------
def log_entry(dataset, step, detail, removed, remaining):
    return {"dataset": dataset, "step": step, "detail": detail,
            "rows_removed": int(removed), "rows_remaining": int(remaining)}


def drop_duplicate_rows(df, ignore=("id",)):
    """Drop rows identical in every column except the ignored ones (the row id)."""
    cols = [c for c in df.columns if c not in ignore]
    out = df.drop_duplicates(subset=cols, keep="first").copy()
    return out, len(df) - len(out)


def age_in_years(days):
    """Cardio stores age in days; Framingham stores whole years, so floor to whole years."""
    return np.floor(pd.to_numeric(days) / 365.25).astype(int)


def compute_bmi(weight_kg, height_cm):
    return weight_kg / (height_cm / 100.0) ** 2


def remove_impossible(df, name, mask_fn):
    """Remove the rows a single plausibility rule flags. Returns (frame, rows_removed)."""
    mask = mask_fn(df)
    return df.loc[~mask].copy(), int(mask.sum())


# --------------------------------------------------------------------------------------
# Binning cholesterol and glucose (mg/dl) into levels 1-3
# --------------------------------------------------------------------------------------
def bin_by_cutoffs(values, cutoffs):
    """Level 1 below cutoffs[0], level 2 from cutoffs[0] up to cutoffs[1], level 3 from cutoffs[1]. NaN stays NaN."""
    v = pd.to_numeric(values, errors="coerce")
    level = pd.Series(1.0, index=v.index)
    level[v >= cutoffs[0]] = 2.0
    level[v >= cutoffs[1]] = 3.0
    level[v.isna()] = np.nan
    return level


def quantile_cutoffs(values, shares):
    """Cut points that give (approximately) the requested level shares, e.g. shares=(0.75, 0.14, 0.11).

    Used only for experiment 2c. It uses B's feature values and A's level shares, never B's labels.
    """
    v = pd.to_numeric(values, errors="coerce").dropna()
    cumulative = np.cumsum(shares)[:-1]
    return tuple(float(x) for x in np.quantile(v, cumulative))


def level_shares(levels):
    s = pd.Series(levels).dropna().astype(int).value_counts(normalize=True).reindex([1, 2, 3]).fillna(0.0)
    return s.round(4).to_dict()


# --------------------------------------------------------------------------------------
# Model pipeline
# --------------------------------------------------------------------------------------
def get_X(df, features=None):
    """Feature frame as floats (levels and binary columns included), so NaN is allowed everywhere."""
    features = list(features or SHARED)
    return df[features].astype(float)


def build_pipeline(model, features=None, scale=False):
    """Impute (fitted on the training data only) -> one-hot the levels -> optional scaling -> model."""
    features = list(features or SHARED)
    num = [c for c in NUMERIC if c in features]
    binary = [c for c in BINARY if c in features]
    levels = [c for c in LEVELS if c in features]

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), num)]
    if binary:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), binary))
    if levels:
        transformers.append(("lvl", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(categories=[[1.0, 2.0, 3.0]] * len(levels),
                                     handle_unknown="ignore", sparse_output=False)),
        ]), levels))
    prep = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("prep", prep), ("clf", model)])
