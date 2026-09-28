"""Exact ranking AUC and Gini of the two denial models on the national sample.

    .venv/bin/python scripts/model_auc.py [--seeds 0 1 2] [--sample-n 1500000]

``hmda model --eval`` prints ranking quality as "percent better than a
base-rate guess" and never prints a raw AUC (tests/test_translate.py). That
percent is (AUC - 0.5) / 0.5 (src/hmda/model/translate.py), which is the
Gini coefficient, 2 x AUC - 1. This script runs the same evaluation
(``hmda.model.evaluate.run_eval`` on ``Source.sample``, the draw the CLI
uses) and records the AUC, the Gini and the split sizes per seed, so the
README and the portfolio page can state "AUC (Gini)" from a committed file.

Writes ``results/model_auc.json``. Several minutes and about 6 GB per seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hmda.clean.source import NATIONAL, open_source
from hmda.model.evaluate import run_eval

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sample-n", type=int, default=1500000)
    args = ap.parse_args(argv)

    src = open_source(NATIONAL)
    print(src.label, flush=True)
    by_seed = []
    for seed in args.seeds:
        frame = src.sample(args.sample_n, seed=seed)
        r = run_eval(frame=frame, source_label=f"national sample, seed {seed}", sample_seed=seed)
        lr, gbm = r["logistic_result"], r["gbm_result"]
        rec = {
            "seed": seed,
            "rows_sample": r["rows_raw"],
            "rows_analysis": r["rows_analysis"],
            "rows_train": r["rows_train"],
            "rows_test": r["rows_test"],
            "test_denial_rate": round(r["test_denial_rate"], 4),
            "baseline_auc": round(lr.baseline_auc, 4),
            "logistic_regression_auc": round(lr.challenger_auc, 4),
            "gradient_boosted_trees_auc": round(gbm.challenger_auc, 4),
            "logistic_regression_gini": round(2 * lr.challenger_auc - 1, 4),
            "gradient_boosted_trees_gini": round(2 * gbm.challenger_auc - 1, 4),
        }
        print(json.dumps(rec), flush=True)
        by_seed.append(rec)

    def mean(key):
        return round(sum(s[key] for s in by_seed) / len(by_seed), 4)

    out = {
        "command": ".venv/bin/python scripts/model_auc.py",
        "source": src.label.replace(str(ROOT) + "/", ""),
        "status": "SAMPLE_BASED",
        "sample_n": args.sample_n,
        "seeds": args.seeds,
        "train_years": [2023, 2024],
        "test_year": 2025,
        "analysis_rows_note": "action_taken 1, 2 and 3 only; purchased loans and non-decisions are dropped before the split",
        "baseline": "a constant score (the training denial rate) cannot order any pair, so its AUC is 0.5",
        "gini_note": "Gini = 2 x AUC - 1, the same number that hmda model --eval prints as percent better than a base-rate guess",
        "mean": {
            "logistic_regression_auc": mean("logistic_regression_auc"),
            "gradient_boosted_trees_auc": mean("gradient_boosted_trees_auc"),
            "logistic_regression_gini": mean("logistic_regression_gini"),
            "gradient_boosted_trees_gini": mean("gradient_boosted_trees_gini"),
            "rows_test": round(sum(s["rows_test"] for s in by_seed) / len(by_seed)),
        },
        "by_seed": by_seed,
    }
    path = ROOT / "results" / "model_auc.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
