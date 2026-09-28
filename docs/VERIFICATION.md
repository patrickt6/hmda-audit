# Verification

This page lists every check in the repo that tests whether the audit's
numbers are right. For each one it says what the check shows, what it does
not show, how to rerun it, and where the result is written. The numbers
quoted here come from those result files. If a number here disagrees with a
result file, the file is correct.

A summary first. The main four-fifths numbers (32,620,789 applications,
6,166,654 denials, 31,793 lender-by-race rows) have now been computed three
ways: by the repo's SQL in DuckDB, by the same SQL in Spark, and by separate
pyarrow code written from the definition. All three agree on every count.
The flagged-lender count is 614. It was 615 until 2026-09-28, when a rule
in `air.py` that flagged a 0/0 ratio was fixed (section 3). The repo's
screen and the pyarrow version now flag the same 614 lenders.

## 1. Claims CI (claimtrail)

**What it shows.** Every number in the README results table links to a
record in `.claimtrail/`, and each record is built from a committed file
under `results/`. On every push, CI checks that the results files still
match their records, that each stated assertion still holds (for example
`lenders_flagged == 614`), that every number in the README results table
is supported, and that the verification log's hash chain is intact.

**What it does not show.** CI cannot rerun the national audit (the 36.7M
rows are not in CI). It checks that the README agrees with the committed
results, not that the results are correct. The test count in the README is
a reported number, checked by hand with `claimtrail verify`. When a fix
moves a number, the steps in [`docs/CLAIMS.md`](CLAIMS.md) ("Changing a
number after review") record it.

**Rerun.**

```
.venv/bin/python scripts/record_claims.py
claimtrail check --paper hmda-audit
claimtrail audit-report README.md
claimtrail verifications --check-chain
```

**Result.** `.claimtrail/`, and the `claims` job in `.github/workflows/claims.yml`.
More detail is in [`docs/CLAIMS.md`](CLAIMS.md).

## 2. Engine parity: DuckDB against Spark

**What it shows.** `sql/air_by_lei_group.sql` gives the same result on two
engines. DuckDB reads the local parquet files and Spark SQL on Databricks
reads a Delta table loaded from the same files. The check compares every
(lender, race) row. Both engines return 31,793 rows with 32,620,789
applications and 6,166,654 denials, and no row differs.

**What it does not show.** Both engines run the **same SQL text**. If that
SQL had a logic error, such as the wrong filter or the wrong denial code,
both engines would agree on the wrong answer. This check finds engine and
loading bugs, not logic bugs. Section 3 is the check for logic bugs.

**Rerun.** You need a logged-in Databricks CLI profile and a SQL warehouse:

```
.venv/bin/python databricks/compare_national.py <warehouse_id>
```

**Result.** `results/spark_parity.json`. Details are in [`databricks/README.md`](../databricks/README.md).

## 3. Independent reimplementation

**What it shows.** `scripts/air_independent.py` computes the per-lender
four-fifths screen a second way. It uses pyarrow instead of SQL and works
from the written definition: drop purchased loans (action_taken 6), count
action_taken 3 as a denial, approval rate = 1 - denials / applications,
reference group = the highest approval rate among the groups that clear the
floor, suppress groups with fewer than 100 applications, and flag a ratio
below 0.8. It reads only the columns it needs from the 168 parquet files.
Then it compares its output with the repo's SQL path, row by row.

Measured on 2026-09-28, after the 0/0 fix:

| check | result |
|---|---|
| (lender, race) rows compared | 31,793 (none missing on either side) |
| mismatched applications, denials, approval rates | 0, 0, 0 (largest difference 0.0) |
| mismatched status (ratio, suppressed or undefined) or ratio | 0, 0 |
| flagged lenders, race only | SQL 444, independent 444, both 444 (symmetric difference 0) |
| flagged lenders, all three columns | SQL 614, independent 614, both 614 (symmetric difference 0) |
| per-state totals (file names and `state_code`) add up to national | pass, pass |
| per-year totals (file names and `activity_year`) add up to national | pass, pass |
| rows whose file name disagrees with their state or year column | 0, 0 |

Before the fix, the two flagged sets differed by exactly one lender (SQL
615 and 445, independent 614 and 444). That lender took 634 applications,
all with race, ethnicity and sex not available, and denied every one of
them. So its only group is the reference group, and that group's approval
rate is 0. `air.py` set the ratio to 0.0 when the reference rate was 0 and
flagged the lender, even though the only comparison was the reference
group against itself. The independent script treated 0/0 as undefined and
did not flag it. That choice was made before the first run and not changed
afterwards to make the two sides match.

`air.py` now does the same: when the reference approval rate is 0, every
eligible group gets ratio `"UNDEFINED"` and is not flagged, in the same way
a group under the floor gets `"SUPPRESSED"` (`src/hmda/fairness/air.py`).
The script now compares that status row by row, and it fails unless both
flagged sets match exactly. The national count moved from 615 to 614
(`results/four_fifths.json` keys `lenders_flagged` and `superseded`). The strict `xfail` test that recorded the old behaviour is now
a passing property, and `tests/test_air.py` has regression tests for this
case.

**What it does not show.** The code is independent (it reads no file in
`sql/` and calls nothing in `rates.py` or `air.py`), but the author is not.
The same person read the SQL and `air.py` before writing this script, so a
misreading shared by both would not be caught. The script checks the
counting and the screen. It does not check the model, the controlled gaps
or the watch list. The two AS and MP files contain no applications, so
there are 56 file states but 54 `state_code` values. This is expected and
the totals still match.

**Rerun.** You need the national parquet files in `data/parquet/`. It takes
about 8 seconds:

```
.venv/bin/python scripts/air_independent.py
```

**Result.** `results/independent_check.json` (aggregate counts only, no
lender identities).

## 4. Property-based tests (Hypothesis)

**What it shows.** `tests/test_properties.py` runs 15 properties on
generated inputs against the real functions in `air.py`, `rates.py` and
`shrink.py`. Examples:

- Row order does not change any ratio, flag or score.
- Groups under the floor are kept and marked `SUPPRESSED`, not dropped.
- Every ratio is in [0, 1], and a group is flagged exactly when its ratio is below 0.8.
- The reference group's ratio is 1 when its approval rate is above 0, and
  `"UNDEFINED"` (never flagged) when it is 0.
- Two identical groups both get ratio 1.
- Purchased loans never change a denial rate.
- The watch-list weights and probabilities are in [0, 1].
- Each posterior mean lies between the lender's raw gap and mu.
- Lenders under `min_each` are never scored.
- An invalid `level` raises an error.

Most properties run 300 examples. The ones that open a DuckDB connection for
each example run 40. The whole file takes about 8 seconds. Every test uses
`derandomize=True` and no example database, so CI sees the same examples
on every run.

The properties found two real problems:

1. `shrink_lender_gaps` crashed with `ZeroDivisionError` when both pooled
   denial rates were 0 or 1, because every variance was then 0. It now
   raises a `ValueError` that explains the cause, and a property tests this.
2. The 0/0 rule in `air.py` described in section 3. Smallest counterexample:
   one group, 1 application, 1 denial, `min_count=1`. The reference group
   got ratio 0.0 and was flagged. This was a strict `xfail` until
   2026-09-28. It is now fixed: the ratio is `"UNDEFINED"` and not flagged,
   and the property passes.

**What it does not show.** A property only tests what it states. These
properties cover invariants and bounds, not exact values. The exact-value
checks are in the ordinary tests and in section 5. The generated inputs are
small (up to 12 lenders and 60 applications), so behaviour that only
appears at national scale is not covered.

**Rerun.**

```
.venv/bin/python -m pytest -q tests/test_properties.py --hypothesis-show-statistics
```

**Result.** The test run itself. The example counts come from `--hypothesis-show-statistics`.

## 5. Mutation testing (mutmut)

**What it shows.** mutmut makes hundreds of small changes to the source,
such as flipping `<` to `<=`, changing a constant or deleting an argument.
It then runs the tests against each changed version. A change that no
test catches points to behaviour that no test checks. On `air.py`,
`rates.py`, `shrink.py` and `floors.py`, measured before the 0/0 fix
described below (474 mutants):

| tests | killed | survived | score |
|---|---|---|---|
| existing tests | 383 | 91 | 80.8% |
| plus property tests | 409 | 65 | 86.3% |
| plus `tests/test_contract_details.py` | 463 | 11 | 97.7% |

`tests/test_contract_details.py` was written to catch the survivors that
mattered. It covers:

- the 0.8 boundary itself
- the result fields
- the `NONE_ELIGIBLE` label
- the lender filter
- a scan in `flagged_lenders` that must not stop early
- the error messages
- the DerSimonian-Laird arithmetic, checked against a second, hand-written copy of the formulas

The 0/0 `UNDEFINED` fix (section 3) changed `air.py` and added tests, so
the run above no longer describes the current code. It was re-run on the
current code with the same four files and the same test selection. `air.py` grew from 137 to 150 mutants (the new `UNDEFINED`
branch adds mutable code), for 487 mutants total:

| tests | killed | survived | score |
|---|---|---|---|
| same selection, after the 0/0 fix (first pass) | 474 | 13 | 97.3% |
| plus one assertion added to the existing `UNDEFINED`-branch test (current code) | 476 | 11 | 97.7% |

The first pass found two new survivors: setting `lei` or `group_column` to
`None` inside the `UNDEFINED` branch of `_air_rows_from_raw` still passed
every test. `tests/test_air.py::test_lender_that_denied_every_application_is_not_flagged`
(added with the 0/0 fix) checked that branch's `ratio` and `flagged`
fields but not `lei` or `group_column`, even though
`tests/test_contract_details.py::test_air_rows_carry_their_scope_fields`
already pins those same fields for the `SUPPRESSED` and normal-ratio
branches. The second pass added the missing assertions to the existing
test; no new test function was needed and the full-suite count stayed at
326 passed, 6 skipped.

The 11 survivors left on the current code were each inspected and are
the same ones found in the first run above (`rates.py` and `shrink.py`
have not changed since the first run, and the `air.py` survivors were
re-checked against the current source). They fall into three groups:

- **Equivalent.** Doubling every weight leaves a weighted mean unchanged, `tau2 > 0` against `tau2 >= 0` gives the same weight when tau2 is 0, and `flagged_lenders` passing `None` in place of `lei` or `group_column` to `_air_rows_from_raw` does not change any row's `.flagged` value, which is the only field `flagged_lenders` reads.
- **Unreachable.** A SQL group always has at least one application, and the second identifier guard sits behind the allowlist.
- **Dependent on the file system.** Upper-casing a `.sql` file name survives only on macOS, which ignores case in file names.

Each one is listed with its file and line in `results/mutation.json`, under
`rerun_2026_09_28` for the current code.

`controlled.py` was also run, as a separate pass: 358 mutants, 260 killed,
98 survived, a score of 72.6%. One survivor inverted the denial indicator
(`action_taken != 3`) and no test caught it. A raw-gap sign test now
catches it. The other 98 survivors are listed but have not been reviewed.

**What it does not show.** A high mutation score means the tests notice
small changes to the code. It does not mean the code is correct, and it
does not show that the tests would catch larger, less mechanical mistakes.
The score also depends on which tests were
selected. Only the tests listed under `[tool.mutmut]` in `pyproject.toml`
were run.

**Rerun.** mutmut rewrites source files, so run it in a throwaway worktree
with its own venv:

```
git worktree add ../hmda-audit-mutmut HEAD
cd ../hmda-audit-mutmut
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev,mutation]"
.venv/bin/mutmut run --max-children 10
.venv/bin/mutmut results
cd - && git worktree remove --force ../hmda-audit-mutmut
```

For `controlled.py`, set `only_mutate` to that file and the tests to
`tests/test_controlled.py` and `tests/test_contract_details.py`. On macOS,
run with `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES OMP_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1`, or the forked workers crash inside scikit-learn.
A pass on the four files takes about 2 minutes on a 12-core Mac.

**Result.** `results/mutation.json`.

## 6. Experiment controls

**What it shows.** `experiments/nn_probe.py` asks whether race can be read
from a denial model that never sees race. Its controls set a floor and a
ceiling for the result:

- A probe on an untrained network of the same shape shows what random features of the inputs already give.
- A probe with the race labels shuffled should score about 0.5, and it does (0.499 ± 0.003 linear, 0.502 ± 0.004 MLP).
- The network's denial score is also tested alone as a race score.

Five seeds, paired by seed, give a spread for each number.

**What it does not show.** A probe measures what information is present,
not whether a decision uses it. It also does not show intent or
discrimination. These are sample numbers (500,000 rows per seed), not
national ones. The MPS backend is not bit-for-bit deterministic, so a rerun
can differ in the last digits.

**Rerun.**

```
uv pip install -e ".[nn]"
.venv/bin/python experiments/nn_probe.py --n 500000 --seeds 0 1 2 3 4
```

**Result.** `results/nn_probe.json`. Details are in [`experiments/README.md`](../experiments/README.md).

## 7. Recorded model runs

The model and mitigation stages train models, so they run on a stratified
sample of the national file, not on all of it. The runs below take several
minutes and about 6 GB of memory each, so they are recorded here instead of
rerun on every commit. `hmda report --model-card` reads its sample-based
numbers from this section (`src/hmda/governance/model_card.py`), and
`hmda verify --card` fails if this section changes shape.

**Ranking quality.** Command, run at seeds 0, 1 and 2:

```
hmda model --eval --source national --sample-n 1500000 --sample-seed S
```

Status: SAMPLE_BASED (n=1,500,000; seeds 0, 1, 2). The test set is 2025 and
the model trains on 2023 and 2024.

| seed | analysis rows | logistic regression | gradient-boosted trees |
|---|---|---|---|
| 0 | 1,058,400 | 57% (78 of 100 pairs) | 72% (86 of 100 pairs) |
| 1 | 1,060,642 | 57% (78 of 100 pairs) | 72% (86 of 100 pairs) |
| 2 | 1,060,029 | 57% (78 of 100 pairs) | 72% (86 of 100 pairs) |

"Better than a base-rate guess" means: take one denied and one approved
application at random. A base-rate guess orders the pair correctly 50 times
in 100. On a 1.5M-row national sample, logistic regression
ranks denials **57%** better than the base-rate baseline (78 of 100 pairs), and
gradient-boosted trees **72%** better (86 of 100 pairs). The CLI prints
whole percentages, so the three seeds agree to 1 point. The same numbers
are in `results/model_eval.json`. On the 50,000-row test fixture the
figures were 61% and 75%, so the small fixture flattered the model.

**Mitigation.** Command, run by hand at seeds 0, 1 and 2:

```
hmda mitigate --compare --profit --source national --sample-n 1500000 --sample-seed S
```

Only the result figures below were written down from these runs, not the
full console output, so they have not been checked a second time.

| technique | gap cut, seed 0 / 1 / 2 | accuracy cost (pts) |
|---|---|---|
| `per_group_threshold` | 60.1% / 61.4% / 55.7% | 0.59 / 0.57 / 0.56 |
| `fair_constrained_gbm` | 10.7% / 11.6% / 11.3% | not recorded |
| `reweighing` | 3.9% / 0.2% / 2.3% | not recorded |

Per-group thresholding cuts the approval-rate gap **55-61%** at a cost of
about 0.6 accuracy points, on a 1.5M-row national sample. The range is the
result; no single seed is. At 500,000 rows the same technique gave 86.3%,
65.2% and 45.9% across the three seeds, a spread of 40.4 points, which is why
these runs use 1,500,000 rows. The pair of groups that defines the widest
gap changed between seeds, so the size of the gap is reported and the two
groups are not named. Per-group thresholding also applies a different
cut-off by race, which is disparate treatment on its face under US law. It
is measured to complete the trade-off study, not offered as a practice.

## Sources for the methods

These describe the methods used here. Only the abstracts were read, on
2026-09-28:

- Semantic triangulation, checking a result against a solution to a different formulation of the same problem: arXiv 2511.12288.
- Property templates, including aggregation decomposition (a global total equals the sum of per-partition totals): arXiv 2607.09072.
- Mutation-guided test generation at Meta: arXiv 2501.12862.
