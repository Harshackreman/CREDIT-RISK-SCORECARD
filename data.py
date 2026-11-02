"""Synthetic retail-loan data generator (no real customer data is used)."""
import numpy as np
import pandas as pd


def generate(n=30000, months=24, seed=42):
    rng = np.random.default_rng(seed)
    month = rng.integers(1, months + 1, n)
    # Mild population drift in later months so the PSI check has something to detect
    drift = np.where(month > 18, 1.0, 0.0)
    age = rng.normal(38, 10, n).clip(21, 70)
    income = rng.lognormal(10.9 - 0.15 * drift, 0.5, n)
    utilization = rng.beta(2, 5 + 2 * (1 - drift), n)
    delinq = rng.poisson(0.4 + 0.3 * drift, n)
    loan_amount = rng.lognormal(12.2, 0.6, n)
    tenure = rng.choice([12, 24, 36, 48, 60], n)
    emp_years = rng.gamma(2.0, 3.0, n).clip(0, 35)
    dti = (loan_amount / tenure) / (income / 12) * 0.5
    logit = (-3.6 + 2.2 * utilization + 0.55 * delinq + 1.4 * dti
             - 0.02 * (age - 38) - 0.04 * emp_years)
    p = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, p)
    return pd.DataFrame(dict(month=month, age=age, income=income, utilization=utilization,
                             delinq_12m=delinq, loan_amount=loan_amount, tenure_months=tenure,
                             emp_years=emp_years, dti=dti, default=default))
