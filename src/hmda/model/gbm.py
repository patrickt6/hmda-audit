"""The challenger models: logistic regression and a gradient-boosted model (lightgbm).

Both are trained on ``model.features.FEATURE_COLUMNS`` only.
Neither ever sees a protected-class column: they are handed the matrix that
:func:`hmda.model.features.build_feature_matrix` returns, and that function
assembles from an allowlist.

Both fits are seeded (``random_state=SEED``) so the reproducibility check
does not fail on model noise.

Versions in the project venv, recorded 2026-09-11 from
``.venv/bin/python -c "import sklearn, lightgbm; print(...)"``:
scikit-learn 1.9.1, lightgbm 4.7.0.
"""

from __future__ import annotations

from typing import Any

#: Fixed seed. Every stochastic step in this module uses it.
SEED = 20260911


def fit_logistic_regression(X_train, y_train) -> Any:
    """Fit a scikit-learn ``LogisticRegression`` on ``X_train``/``y_train`` and return it.

    Wrapped in a ``Pipeline`` with median imputation, a missing-value
    indicator, and standardisation, because a linear model cannot take the
    NaNs that the HMDA sentinels legitimately leave behind (the sentinels
    are missing values, not numbers). The imputation is part of
    the model, so it is fitted on the training fold only and applied to the
    test fold; it cannot leak test information backwards.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def fit_gbm(X_train, y_train) -> Any:
    """Fit a ``lightgbm.LGBMClassifier`` on ``X_train``/``y_train`` and return it.

    LightGBM takes NaN natively, so the missing values are left as missing
    rather than imputed: "missing" is itself informative in HMDA (an exempt
    filer, a not-provided field) and imputing it would erase that.
    """
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=SEED,
        n_jobs=1,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
    )
    model.fit(X_train, y_train)
    return model
