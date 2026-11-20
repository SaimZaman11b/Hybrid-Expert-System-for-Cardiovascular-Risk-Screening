"""Rule-based expert system with a small forward-chaining inference engine.

The four classic parts of an expert system, and where they are in this file
---------------------------------------------------------------------------
1. Knowledge base       RULES (layer 1), INFERENCE_RULES (layers 2-4)
2. Working memory       the set of FACTS: the patient's measurements turn into facts, and rules add new facts
3. Inference engine     infer(): forward chaining, layer by layer, until no rule can add a new fact
4. Explanation facility Inference.trace: every rule that fired, the facts it used and the fact it added

How the chain works (forward chaining)
--------------------------------------
Layer 1  measurements -> facts            e.g. systolic BP 150          =>  high_sysbp
Layer 2  facts -> combined facts          e.g. high_sysbp + high_chol   =>  cardiometabolic_load
Layer 3  facts -> risk band (High/Medium) e.g. 2 or more risk factors   =>  band_medium
Layer 4  default band                     e.g. no other band reached    =>  band_low
Layer 5  band -> recommended action       e.g. band_high                =>  action_followup

A rule fires only when all its conditions are already facts, so layer 2 needs layer 1, and so on.
A rule NEVER fires on a missing value: an unknown attribute simply gives no fact (not applicable).
The default "Low" band has its own layer (4), so it is only tried after layer 3 has finished.

Sources: each rule carries a `source` field. Layer-1 thresholds cite a guideline (ESC/ESH 2018 for blood pressure, NCEP ATP III for
cholesterol, ADA Standards of Care for glucose, WHO for BMI, the Framingham risk profile for smoking and age). The age cut-off of 60,
the combined rules (layer 2), the band cut-offs (layer 3) and the action wording are project design choices, and are labelled that way.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

SRC_BP = ("Williams B, et al. 2018 ESC/ESH Guidelines for the management of arterial hypertension. Eur Heart J 2018;39:3021-3104. "
          "Hypertension = office systolic BP >= 140 and/or diastolic BP >= 90 mmHg.")
SRC_CHOL = ("NCEP Adult Treatment Panel III, Executive Summary of the Third Report. JAMA 2001;285:2486-2497. "
            "Total cholesterol: < 200 mg/dl desirable, 200-239 borderline high, >= 240 high. Level 2 or 3 = 200 mg/dl or more.")
SRC_GLUC = ("American Diabetes Association, Standards of Care in Diabetes, Section 2: Diagnosis and Classification of Diabetes, Diabetes Care "
            "(2026 edition, as accessed). Fasting plasma glucose: < 100 mg/dl normal, 100-125 prediabetes, >= 126 diabetes. "
            "Level 2 or 3 = 100 mg/dl or more. Whether the datasets' glucose values were fasting is not documented (a limitation).")
SRC_BMI = "World Health Organization, fact sheet 'Obesity and overweight'. Adult obesity = BMI >= 30 kg/m2."
SRC_FRAM = ("D'Agostino RB Sr, et al. General cardiovascular risk profile for use in primary care: the Framingham Heart Study. "
            "Circulation 2008;117:743-753. Age, total cholesterol, systolic BP and smoking are risk factors in the model.")
SRC_AGE = (SRC_FRAM + " The cut-off of 60 years is a project choice (no guideline sets 60 as a cardiovascular cut-off); "
           "it is supported by this project's own data (disease rate when the rule fires is higher than when it does not, on A and on B).")
SRC_COMBO = ("Project design. Combines two risk factors that the Framingham model (D'Agostino 2008, Circulation 117:743-753) uses together; "
             "it is not a guideline threshold.")
SRC_BAND = ("Project design. The cut-offs follow outputs/results/rule_count_vs_disease.csv (disease rate rises with the number of rules fired) "
            "and are checked in outputs/results/rule_band_validation.csv.")
SRC_ACTION = "Project design. The recommendation wording is illustrative, not clinical advice."


def _known(value):
    """True when the value is present (not None and not NaN)."""
    if value is None:
        return False
    try:
        return not bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


# ======================================================================================
# Layer 1: measurement rules (the original seven rules, unchanged)
# ======================================================================================
@dataclass(frozen=True)
class Rule:
    key: str          # short id; also the fact it adds, and a column name
    label: str        # text shown to the user
    feature: str      # attribute the rule looks at
    op: str           # ">=" or "=="
    threshold: float
    source: str = ""  # where the threshold comes from (every rule sets one)

    def fires(self, value):
        if not _known(value):
            return False
        value = float(value)
        return value >= self.threshold if self.op == ">=" else value == self.threshold

    def fires_series(self, series):
        s = pd.to_numeric(series, errors="coerce")
        hit = (s >= self.threshold) if self.op == ">=" else (s == self.threshold)
        return (hit & s.notna()).astype(bool)


RULES = [
    Rule("high_sysbp", "Systolic blood pressure at or above 140 mmHg", "sysbp", ">=", 140, SRC_BP),
    Rule("high_diabp", "Diastolic blood pressure at or above 90 mmHg", "diabp", ">=", 90, SRC_BP),
    Rule("high_chol", "Cholesterol above normal", "chol_level", ">=", 2, SRC_CHOL),
    Rule("high_gluc", "Glucose above normal", "gluc_level", ">=", 2, SRC_GLUC),
    Rule("obese", "BMI at or above 30 (obesity range)", "bmi", ">=", 30, SRC_BMI),
    Rule("smoker", "Current smoker", "smoker", "==", 1, SRC_FRAM),
    Rule("older", "Age 60 or older", "age", ">=", 60, SRC_AGE),
]
RULE_KEYS = [r.key for r in RULES]
RULE_LABELS = {r.key: r.label for r in RULES}


def evaluate_rules(patient):
    """Layer-1 rules that fire for one patient (a dict or Series). Returns a list of Rule objects."""
    return [r for r in RULES if r.fires(patient.get(r.feature))]


def rules_fired_matrix(df):
    """Boolean frame, one column per layer-1 rule, one row per person in df."""
    return pd.DataFrame({r.key: r.fires_series(df[r.feature]) for r in RULES}, index=df.index)


# ======================================================================================
# Layers 2-5: inference rules (facts -> new facts)
# ======================================================================================
@dataclass(frozen=True)
class InferenceRule:
    key: str                     # id of the rule
    layer: int                   # 2, 3, 4 or 5
    label: str                   # text shown in the explanation
    concludes: str               # the fact this rule adds
    all_of: tuple = ()           # every one of these facts must be present
    at_least: tuple = ()         # (n, (fact, fact, ...)): at least n of these facts must be present
    none_of: tuple = ()          # none of these facts may be present (rule order matters inside a layer: list it after the rules it depends on)
    source: str = ""  # where the threshold comes from (every rule sets one)

    def conditions_met(self, facts):
        if not all(f in facts for f in self.all_of):
            return False
        if self.at_least:
            n, pool = self.at_least
            if sum(f in facts for f in pool) < n:
                return False
        return not any(f in facts for f in self.none_of)

    def because(self, facts):
        """The facts that made the rule fire, for the explanation."""
        used = list(self.all_of)
        if self.at_least:
            used += [f for f in self.at_least[1] if f in facts]
        return used


DERIVED = ("cardiometabolic_load", "age_related_bp_risk", "metabolic_pattern")

INFERENCE_RULES = [
    # layer 2: combine layer-1 facts
    InferenceRule("R8", 2, "High systolic pressure together with above-normal cholesterol",
                  "cardiometabolic_load", all_of=("high_sysbp", "high_chol"), source=SRC_COMBO),
    InferenceRule("R9", 2, "Age 60 or older together with high systolic pressure",
                  "age_related_bp_risk", all_of=("older", "high_sysbp"), source=SRC_COMBO),
    InferenceRule("R10", 2, "Obesity together with above-normal glucose",
                  "metabolic_pattern", all_of=("obese", "high_gluc"), source=SRC_COMBO),
    # layer 3: risk band High / Medium (the Low default comes in layer 4, after these)
    InferenceRule("R11", 3, "Four or more of the seven risk factors", "band_high",
                  at_least=(4, tuple(RULE_KEYS)), source=SRC_BAND),
    InferenceRule("R12", 3, "Two combined patterns are present", "band_high",
                  at_least=(2, DERIVED), source=SRC_BAND),
    # the Medium rules are listed after the High rules and are skipped when High was already reached
    InferenceRule("R13", 3, "Two or more of the seven risk factors", "band_medium",
                  at_least=(2, tuple(RULE_KEYS)), none_of=("band_high",), source=SRC_BAND),
    InferenceRule("R14", 3, "At least one combined pattern is present", "band_medium",
                  at_least=(1, DERIVED), none_of=("band_high",), source=SRC_BAND),
    # layer 5: recommended action
    InferenceRule("R15", 5, "High risk band", "action_followup", all_of=("band_high",), source=SRC_ACTION),
    InferenceRule("R16", 5, "Medium risk band", "action_review", all_of=("band_medium",),
                  none_of=("band_high",), source=SRC_ACTION),
]
LOW_RULE = InferenceRule("R17", 4, "No other band was reached (default)", "band_low",
                         none_of=("band_medium", "band_high"), source=SRC_BAND)
ACTION_LOW = InferenceRule("R18", 5, "Low risk band", "action_routine", all_of=("band_low",), source=SRC_ACTION)

ALL_INFERENCE_RULES = INFERENCE_RULES[:7] + [LOW_RULE] + INFERENCE_RULES[7:] + [ACTION_LOW]

BANDS = {"band_high": "High", "band_medium": "Medium", "band_low": "Low"}
ACTIONS = {
    "action_followup": "Recommend follow-up with a clinician.",
    "action_review": "Recommend a routine check-up and a lifestyle review.",
    "action_routine": "No action beyond routine care.",
}
DERIVED_LABELS = {
    "cardiometabolic_load": "High systolic pressure and above-normal cholesterol together",
    "age_related_bp_risk": "Age 60 or older with high systolic pressure",
    "metabolic_pattern": "Obesity with above-normal glucose",
}


@dataclass
class Inference:
    facts: set            # every fact in working memory at the end
    trace: list           # one dict per rule that fired, in order
    band: str             # "Low", "Medium" or "High"
    action: str           # recommended action text

    @property
    def n_risk_factors(self):
        return sum(k in self.facts for k in RULE_KEYS)


def infer(patient):
    """Forward chaining for one patient (a dict or Series of attribute values).

    Layer 1 turns measurements into facts. Then layers 2 to 5 are run in order; inside a layer the rules are
    repeated until no rule adds a new fact. The trace records every firing, so the result can be explained.
    """
    facts, trace = set(), []

    # layer 1: measurements -> facts
    for r in RULES:
        if r.fires(patient.get(r.feature)):
            facts.add(r.key)
            trace.append({"layer": 1, "rule": r.key, "because": [f"{r.feature} = {patient.get(r.feature):g}"],
                          "concludes": r.key, "text": r.label})

    # layers 2-5: facts -> new facts, until nothing new can be added in the layer
    for layer in (2, 3, 4, 5):
        changed = True
        while changed:
            changed = False
            for rule in (x for x in ALL_INFERENCE_RULES if x.layer == layer):
                if rule.concludes not in facts and rule.conditions_met(facts):
                    facts.add(rule.concludes)
                    trace.append({"layer": layer, "rule": rule.key, "because": rule.because(facts),
                                  "concludes": rule.concludes, "text": rule.label})
                    changed = True

    band = next(BANDS[b] for b in ("band_high", "band_medium", "band_low") if b in facts)
    action_fact = next(a for a in ("action_followup", "action_review", "action_routine") if a in facts)
    return Inference(facts=facts, trace=trace, band=band, action=ACTIONS[action_fact])


def infer_bands(df):
    """Risk band ("Low"/"Medium"/"High") for every row of df, by running the same engine row by row."""
    cols = [r.feature for r in RULES]
    return pd.Series([infer(dict(zip(cols, row))).band for row in df[cols].itertuples(index=False, name=None)],
                     index=df.index, name="rule_band")
