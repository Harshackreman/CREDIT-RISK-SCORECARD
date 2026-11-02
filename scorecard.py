"""Logistic-regression scorecard on WoE features, converted to points."""
import numpy as np
from sklearn.linear_model import LogisticRegression


class Scorecard:
    def __init__(self, base_score=600, base_odds=50, pdo=20, C=1.0):
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
        self.model = LogisticRegression(C=C, max_iter=1000)

    def fit(self, W, y):
        self.cols = list(W.columns)
        self.model.fit(W, y)
        return self

    def pd_hat(self, W):
        return self.model.predict_proba(W[self.cols])[:, 1]

    def score(self, W):
        p = np.clip(self.pd_hat(W), 1e-6, 1 - 1e-6)
        return self.offset + self.factor * np.log((1 - p) / p)
