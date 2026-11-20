"""Validate the rule-based expert system. Run with:  python src/validate_rules.py

Writes outputs/results/rule_band_validation.csv  (disease rate per rule band, on A and on B)
       outputs/results/rule_model_agreement.csv  (rule band vs the model's flag, on A and on B)
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import preprocessing as pp  # noqa: E402
import rules as rl  # noqa: E402

RESULTS = ROOT / "outputs" / "results"
ORDER = ["Low", "Medium", "High"]


def main():
    info = json.loads((RESULTS / "model_info.json").read_text())
    model = joblib.load(ROOT / "models" / info["models"][info["best_model"]])
    cutoff = info["cutoffs"][info["best_model"]]["recall75"]
    band_rows, agree_rows = [], []
    for name, f in [("A", "dataset_A.csv"), ("B", "dataset_B.csv")]:
        df = pd.read_csv(ROOT / "data" / "processed" / f)
        y = df[pp.TARGET].values
        band = rl.infer_bands(df)
        score_rank = band.map({b: i for i, b in enumerate(ORDER)})
        for b in ORDER:
            m = (band == b).values
            band_rows.append({"dataset": name, "rule_band": b, "n": int(m.sum()), "share_pct": round(m.mean() * 100, 2),
                              "disease_rate_pct": round(y[m].mean() * 100, 2) if m.any() else None})
        band_rows.append({"dataset": name, "rule_band": "ROC-AUC of band as a score", "n": len(df), "share_pct": None,
                          "disease_rate_pct": round(roc_auc_score(y, score_rank), 3)})
        model_flag = model.predict_proba(pp.get_X(df, info["features"]))[:, 1] >= cutoff
        rule_alert = (band != "Low").values  # Medium or High
        for label, mask in [("both alert", rule_alert & model_flag), ("both quiet", ~rule_alert & ~model_flag),
                            ("rules alert, model quiet", rule_alert & ~model_flag),
                            ("model alert, rules quiet", ~rule_alert & model_flag)]:
            agree_rows.append({"dataset": name, "case": label, "n": int(mask.sum()), "share_pct": round(mask.mean() * 100, 2),
                               "disease_rate_pct": round(y[mask].mean() * 100, 2) if mask.any() else None})
    bands, agree = pd.DataFrame(band_rows), pd.DataFrame(agree_rows)
    bands.to_csv(RESULTS / "rule_band_validation.csv", index=False)
    agree.to_csv(RESULTS / "rule_model_agreement.csv", index=False)
    return bands, agree


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    bands, agree = main()
    print(bands.to_string(index=False))
    print()
    print(agree.to_string(index=False))
