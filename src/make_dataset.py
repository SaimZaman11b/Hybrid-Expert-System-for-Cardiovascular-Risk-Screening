"""Raw -> processed. Run with:  python src/make_dataset.py

Dataset A (train): Cardiovascular Disease dataset (Kaggle, sulianova), file cardio_train.csv
Dataset B (test):  Framingham Heart Study (Kaggle copy), file framingham.csv

Writes data/processed/dataset_A.csv, dataset_B.csv and outputs/results/cleaning_log.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import preprocessing as pp  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "outputs" / "results"
OUT_COLS = pp.SHARED + [pp.TARGET, "source"]


def make_dataset_a(log):
    df = pd.read_csv(RAW / "cardio_train.csv", sep=";")
    log.append(pp.log_entry("A", "load", "cardio_train.csv (semicolon separated)", 0, len(df)))

    df, n = pp.drop_duplicate_rows(df)
    log.append(pp.log_entry("A", "drop duplicates", "identical in every column except id", n, len(df)))

    df["age"] = pp.age_in_years(df["age"])
    df["bmi"] = pp.compute_bmi(df["weight"], df["height"])
    log.append(pp.log_entry("A", "convert", "age days -> whole years (floor of days/365.25); BMI = weight / height^2", 0, len(df)))

    for name, rule in pp.PLAUSIBLE:
        df, n = pp.remove_impossible(df, name, rule)
        log.append(pp.log_entry("A", "remove impossible", name, n, len(df)))

    # rename and recode; Cardio gender 2 is the taller group on average, so 2 = male
    df = df.rename(columns={"ap_hi": "sysbp", "ap_lo": "diabp", "smoke": "smoker",
                            "cholesterol": "chol_level", "gluc": "gluc_level", "cardio": pp.TARGET})
    df["sex"] = (df["gender"] == 2).astype(int)
    log.append(pp.log_entry("A", "rename/recode", "gender 2 -> sex 1 (male), 1 -> 0; renamed BP, smoker, level and target columns", 0, len(df)))

    dropped = ["id", "gender", "height", "weight", "alco", "active"]
    df = df.drop(columns=dropped)
    log.append(pp.log_entry("A", "drop attributes", "not in Framingham: " + ", ".join(dropped), 0, len(df)))

    df["source"] = "cardio"
    for c in ["age", "sex", "smoker", "chol_level", "gluc_level", pp.TARGET]:
        df[c] = df[c].astype(int)
    return df[OUT_COLS].reset_index(drop=True)


def make_dataset_b(log):
    raw = pd.read_csv(RAW / "framingham.csv", na_values=["NA"])
    log.append(pp.log_entry("B", "load", "framingham.csv ('NA' read as missing)", 0, len(raw)))

    df = pd.DataFrame({
        "age": raw["age"].astype(int),
        "sex": raw["male"].astype(int),
        "sysbp": raw["sysBP"],
        "diabp": raw["diaBP"],
        "bmi": raw["BMI"],
        "smoker": raw["currentSmoker"].astype(int),
        "chol_level": pp.bin_by_cutoffs(raw["totChol"], pp.CHOL_CUTOFFS).astype("Int64"),
        "gluc_level": pp.bin_by_cutoffs(raw["glucose"], pp.GLUC_CUTOFFS).astype("Int64"),
        pp.TARGET: raw["TenYearCHD"].astype(int),
    })
    df["source"] = "framingham"
    log.append(pp.log_entry("B", "keep/rename", "kept the shared columns; male->sex, sysBP->sysbp, diaBP->diabp, currentSmoker->smoker, TenYearCHD->target", 0, len(df)))
    log.append(pp.log_entry("B", "bin", f"totChol mg/dl -> level using clinical bands {pp.CHOL_CUTOFFS}; glucose mg/dl -> level using {pp.GLUC_CUTOFFS}", 0, len(df)))
    for c in ["bmi", "chol_level", "gluc_level"]:
        log.append(pp.log_entry("B", "missing kept", f"{c}: {int(df[c].isna().sum())} missing values left as missing (imputed later, using A only)", 0, len(df)))

    # report, do not remove: rows outside the plausibility limits used for A
    checks = {
        "systolic BP outside 70-250": ~df["sysbp"].between(70, 250),
        "diastolic BP outside 40-150": ~df["diabp"].between(40, 150),
        "BMI outside 14-60 (missing ignored)": df["bmi"].notna() & ~df["bmi"].between(14, 60),
        "systolic not above diastolic": df["sysbp"] <= df["diabp"],
    }
    for name, mask in checks.items():
        log.append(pp.log_entry("B", "out-of-range (reported, kept)", name + f": {int(mask.sum())} rows", 0, len(df)))
    return df[OUT_COLS].reset_index(drop=True)


def final_checks(a, b):
    assert list(a.columns) == list(b.columns) == OUT_COLS, "column names/order differ"
    for c in pp.SHARED + [pp.TARGET]:
        assert pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]), f"{c} not numeric"
    for c in pp.LEVELS:
        assert set(a[c].dropna().unique()) <= {1, 2, 3} and set(b[c].dropna().unique()) <= {1, 2, 3}, f"{c} outside 1-3"
    assert a[pp.SHARED].isna().sum().sum() == 0, "missing values in A"
    assert set(a[pp.TARGET].unique()) <= {0, 1} and set(b[pp.TARGET].unique()) <= {0, 1}
    assert set(a["sex"].unique()) <= {0, 1} and set(b["sex"].unique()) <= {0, 1}
    assert set(a["smoker"].unique()) <= {0, 1} and set(b["smoker"].unique()) <= {0, 1}
    assert len(b) == 4238, "Framingham rows must all be kept"
    return True


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    log = []
    a = make_dataset_a(log)
    b = make_dataset_b(log)
    final_checks(a, b)
    log.append(pp.log_entry("A+B", "final checks", "identical columns and numeric types; levels only 1-3; no missing values in A; B keeps all rows", 0, len(a) + len(b)))
    a.to_csv(PROCESSED / "dataset_A.csv", index=False)
    b.to_csv(PROCESSED / "dataset_B.csv", index=False)
    log_df = pd.DataFrame(log)
    log_df.to_csv(RESULTS / "cleaning_log.csv", index=False)
    return a, b, log_df


if __name__ == "__main__":
    a, b, log_df = main()
    pd.set_option("display.width", 200, "display.max_colwidth", 110)
    print(log_df.to_string(index=False))
    print(f"\nA: {a.shape}, positive rate {a['target'].mean():.3f}")
    print(f"B: {b.shape}, positive rate {b['target'].mean():.3f}")
