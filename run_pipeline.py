"""End-to-end: generate data -> WoE/IV -> scorecard -> out-of-time validation -> monitoring SQL."""
import sqlite3
import pandas as pd
from src.data import generate
from src.woe import WoEBinner
from src.scorecard import Scorecard
from src.validate import auc_gini_ks, calibration_table, psi

FEATURES = ["age", "income", "utilization", "delinq_12m", "loan_amount",
            "tenure_months", "emp_years", "dti"]

df = generate()
# Time-based split (no random shuffling): train / validation / out-of-time
tr, va, oot = df[df.month <= 16], df[(df.month > 16) & (df.month <= 20)], df[df.month > 20]

binner = WoEBinner().fit(tr[FEATURES], tr.default)
iv = pd.Series(binner.iv).sort_values(ascending=False)
selected = list(iv[iv > 0.02].index)  # drop weak predictors (IV < 0.02)
W = {k: binner.transform(d[FEATURES], selected) for k, d in dict(tr=tr, va=va, oot=oot).items()}

sc = Scorecard().fit(W["tr"], tr.default)

lines = ["# Validation report (synthetic data)\n", "## Information Value\n", iv.round(3).to_string(), ""]
for name, key, d in [("train", "tr", tr), ("validation", "va", va), ("out_of_time", "oot", oot)]:
    m = auc_gini_ks(d.default, sc.pd_hat(W[key]))
    lines.append(f"{name:12s} AUC={m['auc']:.3f}  Gini={m['gini']:.3f}  KS={m['ks']:.3f}")
lines += ["", "## Calibration (out-of-time)\n", calibration_table(oot.default, sc.pd_hat(W["oot"])).to_string()]
s_tr, s_oot = sc.score(W["tr"]), sc.score(W["oot"])
lines += ["", f"## Score PSI (train vs out-of-time): {psi(s_tr, s_oot):.3f}  (<0.10 stable, 0.10-0.25 monitor, >0.25 shift)"]
lines += ["", "## Feature PSI (train vs out-of-time)\n",
          pd.Series({c: psi(tr[c], oot[c]) for c in selected}).round(3).sort_values(ascending=False).to_string()]

scored = oot.assign(sample="out_of_time", pd_hat=sc.pd_hat(W["oot"]), score=s_oot)
scored["score_band"] = pd.cut(scored.score, [-1e9, 500, 550, 600, 650, 1e9],
                              labels=["1_<500", "2_500-549", "3_550-599", "4_600-649", "5_650+"]).astype(str)
con = sqlite3.connect(":memory:")
scored[["default", "pd_hat", "sample", "score_band"]].rename(columns={"default": "is_default"}).to_sql("scored", con, index=False)
q = pd.read_sql(open("sql/monitoring.sql").read(), con)
lines += ["", "## Score-band monitoring (SQL)\n", q.to_string(index=False)]

report = "\n".join(lines)
open("outputs/validation_report.md", "w").write(report)
print(report)
