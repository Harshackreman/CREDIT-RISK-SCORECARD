# Credit Risk Scorecard & Model Validation

End-to-end probability-of-default scorecard with a model-validation layer, built on **synthetic** retail-loan data.

## What it does
1. Generates a 24-month synthetic loan book (deliberate drift after month 18).
2. Bins features and computes Weight of Evidence (WoE) and Information Value (IV); drops weak predictors.
3. Fits a logistic-regression scorecard and converts PD to points (base 600 at 50:1 odds, 20 PDO).
4. Validates on a chronological train / validation / out-of-time split: AUC, Gini, KS, decile calibration.
5. Checks stability with Population Stability Index (score and per-feature).
6. Runs a SQL monitoring query (default rate vs predicted PD by score band).

## Run
    pip install -r requirements.txt
    python run_pipeline.py

Output is printed and saved to `outputs/validation_report.md`.

## Notes
- Data is synthetic; results show the methodology, not real-world performance.
- Extend: monotonic binning, challenger model (gradient boosting) vs scorecard, SHAP reason codes.
