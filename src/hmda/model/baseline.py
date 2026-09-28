"""The base-rate baseline model: always predicts the training set's denial rate.

The baseline model is required, not optional: the M7 percent-better number
has no meaning without it.

What this model is, and why its ranking quality is exactly 0.5
--------------------------------------------------------------
It reads no feature. Every applicant receives the same score: the denial rate
observed in the training data. Because every score is identical, every
denied/approved pair is a tie, and a tie counts half, so the model orders
exactly 50 of every 100 such pairs correctly. That is the floor any
challenger must beat, and it is the anchor the M7 sentence names.

This is a deliberate, documented property rather than a measured one:
``sklearn.metrics.roc_auc_score`` returns 0.5 for a constant score vector by
construction. ``tests/test_model.py::test_base_rate_baseline_ranks_at_chance``
checks it holds in this implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaseRateBaseline:
    """A trivial classifier that always predicts the training denial rate.

    base_rate: the training-set denial rate, set by :meth:`fit`.
    """

    base_rate: float | None = None

    def fit(self, y_train) -> "BaseRateBaseline":
        """Set :attr:`base_rate` to the mean of ``y_train`` and return self."""
        values = np.asarray(y_train, dtype="float64")
        if values.size == 0:
            raise ValueError("cannot fit a base rate on an empty training set")
        self.base_rate = float(values.mean())
        return self

    def predict_proba(self, X):
        """Return :attr:`base_rate` repeated for every row of ``X``, ignoring ``X``'s values."""
        if self.base_rate is None:
            raise ValueError("BaseRateBaseline.fit must be called before predict_proba")
        n_rows = len(X)
        return np.full(n_rows, self.base_rate, dtype="float64")
