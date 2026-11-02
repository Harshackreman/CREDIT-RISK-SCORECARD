"""Model-validation metrics: discrimination, calibration, stability."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def auc_gini_ks(y, p):
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    return dict(auc=auc, gini=2 * auc - 1, ks=float(np.max(tpr - fpr)))


def calibration_table(y, p, q=10):
    df = pd.DataFrame(dict(y=np.asarray(y), p=p))
    df["decile"] = pd.qcut(df.p, q, labels=False, duplicates="drop")
    return df.groupby("decile").agg(n=("y", "size"), predicted_pd=("p", "mean"),
                                    observed_dr=("y", "mean")).round(4)


def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))
