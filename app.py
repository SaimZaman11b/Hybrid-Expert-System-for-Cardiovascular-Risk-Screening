"""Cardiovascular risk-screening demo (Streamlit).   Run with:  streamlit run app.py

Not a medical device. The score is the model's estimate of "cardiovascular disease present at an exam",
as defined in the Cardio training data. It is not a 10-year risk and is not calibrated for lower-risk groups.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import preprocessing as pp  # noqa: E402
import rules as rl  # noqa: E402

FIGURES = ROOT / "outputs" / "figures"
RESULTS = ROOT / "outputs" / "results"


# ---------------------------------------------------------------- logic (no Streamlit calls)
def parse_optional(text):
    """'' -> None (unknown); '212' -> 212.0; anything else raises ValueError."""
    text = (text or "").strip().replace(",", ".")
    if not text:
        return None
    value = float(text)
    if value <= 0:
        raise ValueError("must be positive")
    return value


def levels_from_mgdl(chol_mgdl, gluc_mgdl, bands):
    """Convert mg/dl to level 1-3 with the clinical bands stored in model_info.json. Unknown stays NaN."""
    chol = pp.bin_by_cutoffs(pd.Series([chol_mgdl], dtype=float),
                             (bands["chol_level_2_from"], bands["chol_level_3_from"])).iloc[0]
    gluc = pp.bin_by_cutoffs(pd.Series([gluc_mgdl], dtype=float),
                             (bands["gluc_level_2_from"], bands["gluc_level_3_from"])).iloc[0]
    return chol, gluc


def build_patient(age, sex, sysbp, diabp, bmi, smoker, chol_level, gluc_level):
    return {"age": age, "sex": 1 if sex == "Male" else 0, "sysbp": sysbp, "diabp": diabp, "bmi": bmi,
            "smoker": 1 if smoker else 0, "chol_level": chol_level, "gluc_level": gluc_level}


def predict_score(model, patient, features):
    row = pd.DataFrame([patient])[features].astype(float)
    return float(model.predict_proba(row)[:, 1][0])


# ---------------------------------------------------------------- cached loaders
@st.cache_data
def load_info():
    return json.loads((RESULTS / "model_info.json").read_text())


@st.cache_resource
def load_model(file_name):
    return joblib.load(ROOT / "models" / file_name)


@st.cache_data
def load_csv(name):
    return pd.read_csv(RESULTS / name)


# ---------------------------------------------------------------- page
def main():
    st.set_page_config(page_title="Cardiovascular risk screening", page_icon="🫀", layout="wide")
    info = load_info()
    st.title("Cardiovascular risk screening demo")
    st.warning("Student project, not a medical device. Do not use it to make medical decisions.")
    tab_screen, tab_results = st.tabs(["Screening", "Results"])

    with tab_screen:
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            st.subheader("Measurements")
            model_names = list(info["models"])
            default = model_names.index(info["best_model"])
            model_name = st.selectbox("Model", model_names, index=default,
                                      help=f"Default is the model selected in cross-validation: {info['best_model']}.")
            c1, c2 = st.columns(2)
            age = c1.number_input("Age (years)", min_value=18, max_value=100, value=50, step=1)
            sex = c2.selectbox("Sex", ["Female", "Male"])
            c3, c4 = st.columns(2)
            sysbp = c3.number_input("Systolic BP (mmHg)", min_value=60, max_value=260, value=120, step=1)
            diabp = c4.number_input("Diastolic BP (mmHg)", min_value=30, max_value=160, value=80, step=1)
            c5, c6 = st.columns(2)
            bmi = c5.number_input("BMI (kg/m²)", min_value=12.0, max_value=70.0, value=25.0, step=0.1)
            smoker = c6.checkbox("Current smoker", value=False)
            st.caption("Cholesterol and glucose are optional. Leave blank if unknown; the model then uses the most common level in the training data.")
            c7, c8 = st.columns(2)
            chol_text = c7.text_input("Total cholesterol (mg/dl)", value="", placeholder="e.g. 210")
            gluc_text = c8.text_input("Glucose (mg/dl)", value="", placeholder="e.g. 90")

        problems, chol_mgdl, gluc_mgdl = [], None, None
        try:
            chol_mgdl = parse_optional(chol_text)
        except ValueError:
            problems.append("Cholesterol must be a positive number or left blank.")
        try:
            gluc_mgdl = parse_optional(gluc_text)
        except ValueError:
            problems.append("Glucose must be a positive number or left blank.")
        if sysbp <= diabp:
            problems.append("Systolic pressure must be higher than diastolic pressure.")

        with right:
            st.subheader("Result")
            for p in problems:
                st.error(p)
            if not problems:
                if age < 29 or age > 64:
                    st.info("The training data covers ages 29 to 64, so the estimate for this age is an extrapolation.")
                chol_level, gluc_level = levels_from_mgdl(chol_mgdl, gluc_mgdl, info["level_bands_mgdl"])
                patient = build_patient(age, sex, sysbp, diabp, bmi, smoker, chol_level, gluc_level)
                model = load_model(info["models"][model_name])
                score = predict_score(model, patient, info["features"])
                cutoff = info["cutoffs"][model_name]["recall75"]
                flagged = score >= cutoff

                m1, m2 = st.columns(2)
                m1.metric("Model score", f"{score:.1%}")
                m2.metric("Flag cutoff", f"{cutoff:.1%}")
                if flagged:
                    st.error("**Flagged for follow-up.** The score is at or above the cutoff.")
                else:
                    st.success("**Not flagged.** The score is below the cutoff.")
                st.caption(
                    "Score = the model's estimate that cardiovascular disease is present, as defined in the Cardio training data (about 50% of that data). "
                    f"The cutoff is the score that caught {int(0.75*100)}% of the people with disease in cross-validation on that data. "
                    "In a lower-risk group most people flagged this way will not have the disease, and the score is not calibrated for such groups.")

                shown = {"Cholesterol level": chol_level, "Glucose level": gluc_level}
                text = ", ".join(f"{k}: {'unknown' if pd.isna(v) else int(v)}" for k, v in shown.items())
                st.write(f"**Levels used by the model** (from mg/dl, bands in the Results tab): {text}")

                st.markdown("**Expert system: forward-chaining reasoning**")
                result = rl.infer(patient)
                banner = {"High": st.error, "Medium": st.warning, "Low": st.success}[result.band]
                banner(f"**Rule-based risk band: {result.band}.** {result.action}")
                if flagged == (result.band != "Low"):
                    st.info("The model and the rules agree.")
                else:
                    st.warning("The model and the rules disagree (the rules alert on Medium or High). Refer for review.")
                layer_titles = {1: "Layer 1: measurements become facts", 2: "Layer 2: facts combine into new facts",
                                3: "Layer 3: facts give a risk band", 4: "Layer 4: default band",
                                5: "Layer 5: band gives the recommended action"}
                for layer, title in layer_titles.items():
                    steps = [t for t in result.trace if t["layer"] == layer]
                    if steps:
                        st.markdown(f"*{title}*")
                        for t in steps:
                            used = ", ".join(t["because"]) if t["because"] else "no other band was reached"
                            st.markdown(f"- **{t['rule']}**: {t['text']} (from: {used}) → `{t['concludes']}`")
                st.caption("A rule fires only when all its conditions are already facts, so each layer builds on the one before. "
                           "A rule never fires on a missing value. Rule thresholds are common clinical cut-offs (see src/rules.py).")

                st.markdown(f"**What the {model_name} model relies on** (permutation importance on held-out folds)")
                imp = load_csv("feature_importance.csv")
                if (RESULTS / "feature_importance_ensemble.csv").exists():
                    imp = pd.concat([imp, load_csv("feature_importance_ensemble.csv")], ignore_index=True)
                imp = imp[imp.model == model_name].set_index("feature")["importance_mean"].sort_values()
                st.bar_chart(imp, horizontal=True, height=280)

    with tab_results:
        st.subheader("Model comparison")
        comp = load_csv("comparison_table.csv")
        comp["model"] = np.where(comp["selected"] == True, comp["model"] + " (selected)", comp["model"])  # noqa: E712
        st.dataframe(comp.drop(columns="selected").set_index("model").T.astype(float).round(3), width="stretch")
        st.caption("A_cv_* are nested 5-fold cross-validation results on dataset A. B_* are the same models trained on A and tested on Framingham (B) "
                   "unless the name says otherwise. On B the headline measures are ROC-AUC and PR-AUC; B's PR-AUC baseline is its positive rate.")
        bands = info["level_bands_mgdl"]
        st.write(f"Level bands used for cholesterol and glucose (mg/dl): cholesterol level 2 from {bands['chol_level_2_from']:.0f}, "
                 f"level 3 from {bands['chol_level_3_from']:.0f}; glucose level 2 from {bands['gluc_level_2_from']:.0f}, level 3 from {bands['gluc_level_3_from']:.0f}.")
        st.write(f"Selection rule: {info['selection_rule']}.")

        st.subheader("Baseline vs improved")
        for csv_name, caption in [("baseline_vs_improved.csv", "ROC-AUC of each model and the soft-voting ensemble, with the gain over the "
                                   "Logistic Regression baseline (A: nested 5-fold CV; B: trained on A, tested on Framingham)."),
                                  ("exp5_imbalance_framingham.csv", "Class-imbalance experiment on Framingham (15.2% positive), 5-fold CV: "
                                   "recall, precision and F1 at the 0.5 cutoff for each treatment.")]:
            if (RESULTS / csv_name).exists():
                st.dataframe(load_csv(csv_name).round(3), width="stretch", hide_index=True)
                st.caption(caption)

        st.subheader("Expert system validation")
        for csv_name, caption in [("rule_band_validation.csv", "Disease rate by rule-based risk band (run src/validate_rules.py to refresh)."),
                                  ("rule_model_agreement.csv", "Rule band (Medium or High = alert) against the selected model's flag.")]:
            if (RESULTS / csv_name).exists():
                st.dataframe(load_csv(csv_name), width="stretch", hide_index=True)
                st.caption(caption)
        st.subheader("Fairness check")
        if (RESULTS / "fairness_by_group.csv").exists():
            st.dataframe(load_csv("fairness_by_group.csv"), width="stretch", hide_index=True)
            st.caption("Error rates by sex and age group at the app's flag cutoff (run src/fairness.py to refresh). A: out-of-fold predictions. "
                       "B: model trained on A. Gap = largest minus smallest value across the groups.")
        for fig, caption in [("baseline_vs_improved.png", "Baseline vs improved"),
                             ("imbalance_experiment.png", "Class-imbalance experiment on Framingham"),
                             ("roc_ensemble.png", "ROC curves with the soft-voting ensemble"),
                             ("fairness.png", "Recall and false-positive rate by group"),
                             ("model_comparison.png", "ROC-AUC and PR-AUC by experiment"),
                             ("roc_curves.png", "ROC curves"),
                             ("confusion_matrices.png", "Confusion matrices at each model's cutoff"),
                             ("feature_importance.png", "Permutation importance"),
                             ("rule_validation.png", "Rule validation")]:
            path = FIGURES / fig
            if path.exists():
                st.image(str(path), caption=caption, width="stretch")


main()
