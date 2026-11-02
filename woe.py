"""Weight-of-Evidence binning and Information Value."""
import numpy as np
import pandas as pd


class WoEBinner:
    def __init__(self, n_bins=8, eps=0.5):
        self.n_bins, self.eps = n_bins, eps
        self.edges, self.woe, self.iv = {}, {}, {}

    def fit(self, X, y):
        for c in X.columns:
            q = np.unique(np.quantile(X[c], np.linspace(0, 1, self.n_bins + 1)))
            q[0], q[-1] = -np.inf, np.inf
            b = pd.cut(X[c], q, labels=False, include_lowest=True)
            t = pd.crosstab(b, y).reindex(columns=[0, 1], fill_value=0)
            good = (t[0] + self.eps) / (t[0].sum() + self.eps * len(t))
            bad = (t[1] + self.eps) / (t[1].sum() + self.eps * len(t))
            w = np.log(good / bad)
            self.edges[c], self.woe[c] = q, w
            self.iv[c] = float(((good - bad) * w).sum())
        return self

    def transform(self, X, cols=None):
        out = {}
        for c in (cols or X.columns):
            b = pd.cut(X[c], self.edges[c], labels=False, include_lowest=True)
            out[c] = b.map(self.woe[c]).fillna(0.0).values
        return pd.DataFrame(out, index=X.index)
