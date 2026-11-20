# Cardiovascular Risk Screening: train on exam data, test on a long-term cohort

A student project that builds a screening model from **8 routine measurements** and asks a hard question:
if a model is trained on a large exam-based dataset, does it also rank people by who goes on to develop coronary heart disease (CHD) in an independent cohort?

**Not a medical device.** Do not use it for medical decisions.

## The two datasets

| | Dataset A (train) | Dataset B (test) |
|---|---|---|
| Source | Cardiovascular Disease dataset (Kaggle, user `sulianova`) | Framingham Heart Study (Kaggle copy) |
| Rows after cleaning | 68,515 | 4,238 (none removed) |
| Label | `cardio`: disease present at the exam (49.5% positive) | `TenYearCHD`: CHD within 10 years (15.2% positive) |

**Shared attributes:** `age`, `sex`, `sysbp`, `diabp`, `bmi`, `smoker`, `chol_level` (1-3), `gluc_level` (1-3).

**Known mismatches (the point of the project is to measure their effect):**
- The labels mean different things (disease now vs CHD in 10 years) and have different prevalence.
- Smokers are 8.8% of A and 49.4% of B.
- Cholesterol levels are 75/14/12% in A and 20/35/45% in B under clinical bands. Cardio's level definitions are not documented in mg/dl.
- Cardio's blood-pressure readings are heavily rounded (97% multiples of 10, against 12% in Framingham).
- B has missing BMI (19 rows), cholesterol (50) and glucose (388).

## Results (all produced by the notebooks; see `outputs/results/`)

| | Logistic Regression (baseline) | Random Forest | Gradient Boosting | Soft-voting ensemble |
|---|---|---|---|---|
| Within A, nested 5-fold CV, ROC-AUC | 0.791 | 0.800 | 0.800 | 0.799 |
| Within A, recall at 70% specificity | 0.747 | 0.757 | 0.755 | 0.757 |
| **Train A, test B, ROC-AUC** (95% bootstrap interval) | **0.680** (0.659-0.701) | 0.648 (0.628-0.669) | 0.661 (0.640-0.682) | 0.670 (0.650-0.691) |
| Train A, test B, PR-AUC (baseline 0.152) | 0.280 | 0.220 | 0.232 | 0.249 |
| Reference: trained inside Framingham, ROC-AUC | 0.723 | 0.712 | 0.705 | |

- The model selected on A alone is **Gradient Boosting** (all four, the ensemble included, are tied on the selection score, and the tie is broken by ROC-AUC).
- **Ensemble (Exp 4):** within A it matches the best single model. On B it loses less than either tree model (0.670 against 0.648 and 0.661), so averaging makes the transfer more stable.
- **Class imbalance (Exp 5, Framingham, 15.2% positive):** without treatment the models catch only 0-5% of CHD cases at the 0.5 cutoff. With `class_weight="balanced"` they catch 64-68%, and F1 rises from under 0.1 to 0.37-0.39. ROC-AUC hardly changes (the treatment moves the cutoff, not the ranking). Details in `outputs/results/exp5_imbalance_framingham.csv`.
- **Baseline vs improved** in one table and chart: `outputs/results/baseline_vs_improved.csv`, `outputs/figures/baseline_vs_improved.png`.
- On B, ROC-AUC falls to about 0.65 to 0.68. The models still beat chance, but less well than within A. The best model on A is not the best on B.
- Counting how many of the 7 rules fire gives ROC-AUC 0.66 on B, in the same range as the trained models.
- The forward-chaining expert system (below) gives a Low / Medium / High band. Disease rate rises with the band: A 32.5% / 69.1% / 82.0%, B 7.5% / 14.7% / 29.1% (`outputs/results/rule_band_validation.csv`).
- Removing `smoker` (Exp 2b) or using only complete B rows (Exp 2d) changes ROC-AUC by 0.01 or less. Re-binning B's levels to match A (Exp 2c) helps the tree models most.
- Cutoffs do not transfer: A's recall-75 cutoff flags 64% to 70% of B.

## Folder layout

```
data/raw/            cardio_train.csv, framingham.csv (original files)
data/processed/      dataset_A.csv, dataset_B.csv (written by src/make_dataset.py)
notebooks/           01_data_and_eda.ipynb, 02_modeling_and_evaluation.ipynb, 03_improvements.ipynb (executed, with outputs)
src/make_dataset.py  raw -> processed, with a rule-by-rule cleaning log
src/preprocessing.py cleaning steps, level binning, model pipeline
src/rules.py         expert system: 7 measurement rules + forward-chaining inference engine (layers 1-5)
src/fairness.py     fairness check by sex and age group -> outputs/results/fairness_by_group.csv, outputs/figures/fairness.png
src/validate_rules.py validates the rule bands and rule-vs-model agreement -> outputs/results/rule_*.csv
src/improvements.py  soft-voting ensemble, class-imbalance experiment, baseline-vs-improved summary (used by notebook 03)
models/              three trained models, the soft-voting ensemble and best_model.joblib (all trained on all of A)
outputs/figures/     EDA and modelling charts
outputs/results/     all result tables, cleaning_log.csv, model_info.json
app.py               Streamlit app
```

## How to run

```bash
pip install -r requirements.txt
python src/make_dataset.py          # writes data/processed/*.csv and outputs/results/cleaning_log.csv
# run notebooks 01, 02 and 03 in order (in Jupyter, from the notebooks/ folder); 02 and 03 take a few minutes each
# (notebook 03 can also be run as: python src/improvements.py)
python src/validate_rules.py        # expert-system validation tables
python src/fairness.py              # fairness check by sex and age group
streamlit run app.py
```

## Method in brief

- **Cleaning (A):** drop duplicates; convert age to years; compute BMI; remove impossible values one rule at a time (each rule is logged); recode `gender` (code 2 is the taller group, so 2 = male); drop attributes Framingham lacks.
- **EDA (notebook 01):** summary statistics (count, mean, SD, min, quartiles, max, missing) for A and B, means by class, category shares (`outputs/results/summary_*.csv`); then class balance, missing values, distributions, disease rate by category, correlations and an A-vs-B shift test.
- **Binning (B):** total cholesterol below 200 / 200-239 / 240 and above mg/dl, and glucose below 100 / 100-125 / 126 and above mg/dl, give levels 1/2/3.
- **Models:** Logistic Regression (baseline), Random Forest and Gradient Boosting (scikit-learn's histogram-based implementation), with small grids. Imputation (median, most frequent level) is fitted on training data only; scaling is applied for Logistic Regression only.
- **Ensemble (Exp 4):** soft voting (the average probability) of the three models. Tested with the same nested 5-fold CV on A: in each outer fold, each member uses the setting its inner search chose in that fold. Then trained on all of A and tested on B.
- **Selection:** nested 5-fold CV on A only; highest recall at 70% specificity, ties within 0.02 broken by ROC-AUC. The ensemble is included. B is never used to select, tune or impute.
- **Imbalance (Exp 5):** inside Framingham's 5-fold CV (the Exp 3 setting), each model is trained with no treatment, `class_weight="balanced"`, or SMOTE (in an `imblearn` pipeline, so only training folds are resampled).
- **Experiments:** 2 (train A, test B), 2b (no smoker), 2c (B's levels re-binned to A's shares), 2d (B complete cases), 3 (reference: train and test inside Framingham, not used for selection).
- **App cutoff:** the score that gives 75% recall on A's out-of-fold predictions. The Youden cutoff is stored in `model_info.json` for reference.

## Class imbalance

Dataset A is balanced (49.5% positive, 33,901 of 68,515), so no resampling or class weights are needed for training. Dataset B (Framingham) is imbalanced
(15.2% positive, 644 of 4,238), but it is only used for testing. On B the models are therefore judged with ROC-AUC, PR-AUC against the 0.152
baseline, and recall at a fixed specificity, not with accuracy. The counts are in `outputs/results/class_balance.csv` and in the EDA notebook.

**How the imbalance was addressed (Exp 5, notebook 03).** When a model is trained on Framingham itself, the imbalance does matter, so it was tested there
with 5-fold CV and three treatments:

| Logistic Regression on Framingham, 0.5 cutoff | Recall | F1 | ROC-AUC |
|---|---|---|---|
| No treatment | 0.05 | 0.09 | 0.723 |
| `class_weight="balanced"` | 0.68 | 0.39 | 0.724 |
| SMOTE (training folds only) | 0.68 | 0.39 | 0.722 |

Class weighting is the better choice overall: it works for all three models (recall 0.64-0.68), while SMOTE helps the tree models much less
(recall 0.14-0.19). The app handles the same problem a different way, by choosing its flag cutoff for 75% recall instead of using 0.5.

## Data privacy and ethics

- **Privacy:** both datasets are public research datasets with no names, contact details or other direct identifiers. The `id` column of Cardio is dropped,
  and no personal data is collected or stored by the app (values typed into the app are used for one prediction and not saved).
- **Transparency:** permutation importance shows what each model relies on, and the expert system shows every rule that fired and the facts it used.
- **Accountability:** the app and README state that this is a student project and not a medical device. Disagreements between the model and the rules are
  marked "refer for review", so a person makes the final decision.
- **Fairness:** see the next section.

## Fairness check (`src/fairness.py`)

Error rates by sex and by age group at the app's flag cutoff (`outputs/results/fairness_by_group.csv`, `outputs/figures/fairness.png`).
On A the predictions are out-of-fold; on B the model is trained on A.

| Gap (max - min across groups) | Recall | False-positive rate |
|---|---|---|
| Sex, A | 0.0 points | 1.5 points |
| Sex, B | 0.4 points | 1.7 points |
| Age group, A | 23.2 points | 45.6 points |
| Age group, B | 41.9 points | 61.1 points |

- **Sex:** the model treats women and men almost the same on both datasets. Note that Cardio's gender code was mapped to sex by assumption (code 2 = male, the taller group), which is stated in the cleaning log.
- **Age:** the gaps are large. Disease becomes much more common with age, so the model flags most older people (on B, 89% of ages 55-64 and 97% of ages 65 and over) and has a high false-positive rate in them, while it misses many younger cases (recall 60% on A and 57% on B under age 45). The cutoff is not equally good for every age group.
- **What to do about it:** use age-specific cutoffs or calibrate by age group, and do not use one national cutoff on a population with a different age mix. This is not done here and is listed as a limitation.

## Expert system (rule base with forward chaining)

`src/rules.py` is a small rule-based expert system. It has the four standard parts: a **knowledge base** (`RULES`, `INFERENCE_RULES`),
a **working memory** (the set of facts about one person), an **inference engine** (`infer()`, forward chaining) and an
**explanation facility** (the trace of every rule that fired, shown in the app).

| Layer | What happens | Example |
|---|---|---|
| 1 | Measurements become facts (7 rules) | systolic BP 150 gives `high_sysbp` |
| 2 | Facts combine into new facts (R8-R10) | `high_sysbp` and `high_chol` give `cardiometabolic_load` |
| 3 | Facts give a risk band (R11-R14) | 4 or more risk factors, or 2 combined patterns, give High; 2 or more risk factors, or 1 combined pattern, give Medium |
| 4 | Default band (R17) | no other band reached gives Low |
| 5 | Band gives the recommended action (R15, R16, R18) | High gives "follow-up with a clinician" |

A rule fires only when all its conditions are already facts, so each layer needs the one before (forward chaining). A rule never fires on a missing value.
The app shows the rule band next to the model's flag and marks disagreements as "refer for review" (the rules alert on Medium or High).
Run `python src/validate_rules.py` to refresh `rule_band_validation.csv` and `rule_model_agreement.csv`.

## Sources for the rule thresholds

| Rule / band | Cut-off | Source |
|---|---|---|
| `high_sysbp`, `high_diabp` | 140 / 90 mmHg | Williams B, et al. 2018 ESC/ESH Guidelines for the management of arterial hypertension. *Eur Heart J* 2018;39:3021-3104 |
| `high_chol` and the cholesterol levels | 200 and 240 mg/dl | NCEP Adult Treatment Panel III, Executive Summary. *JAMA* 2001;285:2486-2497 |
| `high_gluc` and the glucose levels | 100 and 126 mg/dl | American Diabetes Association, Standards of Care in Diabetes, Section 2: Diagnosis and Classification of Diabetes, *Diabetes Care* (2026 edition, as accessed) |
| `obese` | BMI 30 kg/m2 | World Health Organization, fact sheet "Obesity and overweight" |
| `smoker`, and age as a risk factor | n/a | D'Agostino RB Sr, et al. General cardiovascular risk profile for use in primary care: the Framingham Heart Study. *Circulation* 2008;117:743-753 |
| `older` | 60 years | Project choice. No guideline sets 60 as a cardiovascular cut-off. Supported by this project's data (disease rate is higher when the rule fires, on A and on B) |
| Combined rules R8-R10 | n/a | Project design, based on risk factors the Framingham model uses together. Not guideline thresholds |
| Band rules R11-R14 | 2 and 4 risk factors; 1 and 2 combined patterns | Project design, from `rule_count_vs_disease.csv` and checked in `rule_band_validation.csv` |
| Action wording R15, R16, R18 | n/a | Project design. Illustrative, not clinical advice |

Two limits of the thresholds:
the ADA glucose cut-offs are for fasting glucose, and the datasets do not say whether the glucose values were fasting; and the `smoker` rule
does not behave as expected on Cardio (disease rate is slightly lower for smokers).

## Limitations

- The score means "cardiovascular disease present at the exam", as defined in Cardio. It is **not** a 10-year risk and is **not calibrated** for lower-risk groups. A's cutoff comes from a roughly 50%-positive dataset, so in a lower-risk group most people flagged will be false alarms.
- Cardio is an exam dataset with a known-noisy blood-pressure field, and its cholesterol and glucose levels are not defined in mg/dl. The app converts mg/dl with clinical bands, which only approximately match what the model learned.
- B's ages reach 70, while A's stop at 64 (4% of B is older than 64).
- Whether rounded blood-pressure readings in A explain why the tree models transfer worse was not tested.
- The flag cutoff is not equally good across age groups (see the fairness check); no age-specific cutoffs were built.
- Gradient Boosting here is scikit-learn's `HistGradientBoostingClassifier`.
- SMOTE (Exp 5) runs after one-hot encoding, so synthetic people can have in-between level values. Class weighting, which has no such issue, worked as well or better.

## Credits

- **Dataset A:** Ulianova, S. *Cardiovascular Disease dataset.* Kaggle. https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset (licence listed as "Unknown" on Kaggle).
- **Dataset B:** Dileep. *Logistic regression To predict heart disease* (Framingham Heart Study data). Kaggle. https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression (licence listed as "Unknown" on Kaggle).
- Both datasets are public research data with no personal identifiers. They are used here for non-commercial, educational purposes only, and all credit for the data belongs to their original authors.
