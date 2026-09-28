# The recourse / effort study (M13)

**Verdict: M13 is UNMEASURABLE, with the reason below.** The method ran, it
is correct, and its answer is not stable enough to quote. That result is
recorded as the result. The metric is not silently dropped, and no proxy is
reported in its place. No figure in this document should be quoted as a
money figure. The fixture study (sections 1 to 6) found the instability,
and a national check on a 1,500,000-row sample (section 8) confirmed it.

Code: `src/hmda/fairness/recourse.py`. Tests: `tests/test_recourse.py`.
Fixture study measured 2026-09-11.

---

## 0. What this is, and what it is not

The study asks: taking a model fitted to historical mortgage application
records, for the applicants that model scores on the denial side of a
decision line, how much extra annual income would it take to move each of
them to the other side, and does that amount differ by protected group?

**This is a property of a model. It is not a statement about what any real
lender requires of any real applicant.** The model makes no credit decision,
influences no credit decision, and has never been deployed. The
decision line is a modelling choice made in this repo (§3 below), not any
lender's rule; no lender's threshold is known here and none is claimed. No
individual lender is named anywhere in this study.

**Every number in sections 2 to 6 is a FIXTURE number.** The source is
`tests/fixtures/hmda_50k.parquet`, which holds DC, WY and VT rows only. It
is **not** a national sample. Section 8 covers the national check. No group
may be characterised nationally from any figure on this page.

---

## 1. The method

A **one-dimensional grid search over income**, holding every other input
fixed.

For each applicant in scope, the code adds a candidate amount to the
`income_dollars` feature, recomputes the one engineered feature that depends
on it (`loan_to_income_ratio`), re-scores the row with the real model, and
takes the smallest candidate whose predicted denial probability falls below
the decision line. Candidates run at $1,000 resolution to $100,000, then on a
coarser geometric tail to a $1,000,000 cap. An applicant no candidate helps
is recorded as **unreachable**, never as a number.

Why a grid and not a bisection or an optimiser: the challenger is a
gradient-boosted tree model, which is a step function and is not guaranteed
monotone in income. Bisection assumes a single crossing and can return a root
that does not exist. A grid evaluates the actual model at every candidate, so
it can only report a crossing the model really has. Its cost is resolution,
which is a stated parameter (`GRID_STEP_USD`), not a hidden one.

**Actionable inputs only.** `ACTIONABLE_FEATURES` in the module is
`("income", "loan_amount", "loan_to_value_ratio")`. An applicant cannot
change their race, so a recourse number computed over a protected feature
would be meaningless. Beyond the explicit list, the feature matrix the search
perturbs comes from `model/features.py`'s **allowlist**, which contains no
protected column at all, so no protected field can enter the search even by
accident.

**Declared but not searched:** `loan_amount` and `loan_to_value_ratio`.
M13's required output shape is a money figure in additional income; a
loan-amount or LTV change is a different claim in different units, and
folding all three into one "minimum cost" needs a rate of exchange between a
dollar of income and a dollar of loan that this repo has no evidence for and
will not invent. This is a stated limit, recorded in the code as
`UNSEARCHED_ACTIONABLE`.

**Grouping is not actionability.** The comparison is *by* protected group, so
a protected column is needed to split rows for reporting. The module never
spells one; it iterates `hmda.fairness.rates.ALLOWED_GROUP_COLUMNS`. Those
columns are used only to partition rows, never as an input the search may
move. `grep -rn "derived_race\|derived_sex\|derived_ethnicity"
src/hmda/fairness/recourse.py` returns nothing, and
`tests/test_recourse.py::test_no_protected_field_name_appears_in_the_module_source`
asserts it so it cannot silently regress.

---

## 2. Units: the 1000x trap

HMDA's raw `income` column is denominated in **thousands of dollars**. The study reads income only through
`clean/sentinels.py::income_dollars`, via
`model/features.py::build_feature_matrix`, which multiplies by 1,000.

Measured on the fixture, 2026-09-11: **median income $114,000**
(`test_median_income_on_the_fixture_is_in_a_human_range` asserts it lands
between $30,000 and $300,000). Read raw, the median would be $114 and every
figure in this study would be wrong by three orders of magnitude.

---

## 3. Scope and the decision line

- Model: gradient-boosted trees (`model/gbm.py::fit_gbm`, seed 20260911),
  fitted on the **30,232** fixture rows before `activity_year` 2025.
- Study fold: the **4,863** held-out rows from 2025 onward. Same time split
  as `model/evaluate.py`; never a random shuffle.
- Decision line: the **base-rate cut**, the quantile of the *training*
  scores that puts the same share of rows on the denial side as were actually
  denied in training (training denial rate 0.2184, giving a threshold of
  **0.279**). Fitted on the training fold only.
- In scope: applicants the data records as denied **and** whom the model also
  scores at or above the line. A row the model already places on the approval
  side has no line to cross and would contribute a meaningless zero.
- Reporting floor: **30** denied applicants, or the group is reported as
  NOT REPORTED rather than given a number, because a median over a handful
  of people swings with one row. Groups below the floor are flagged, never
  dropped.

---

## 4. What the run produces

`.venv/bin/python -m hmda.fairness.recourse` (exit 0). Grouped by race, the
fixture run gives:

| Group (fixture only) | Median extra annual income | Reachable / denied | Unreachable |
|---|---|---|---|
| Black or African American | $7,500 | 22 / 303 | 92.7% |
| Race Not Available | $30,000 | 20 / 242 | 91.7% |
| White | $40,500 | 8 / 173 | 95.4% |
| Asian | NOT REPORTED | n/a / 19 | below the floor of 30 |
| Joint | NOT REPORTED | n/a / 12 | below the floor of 30 |
| 2 or more minority races | NOT REPORTED | n/a / 5 | below the floor of 30 |
| Native Hawaiian or Other Pacific Islander | NOT REPORTED | n/a / 4 | below the floor of 30 |

**Do not quote the median column.** Section 5 is why.

---

## 5. Why M13 is UNMEASURABLE

### 5.1 The medians are not stable

Across **3 model seeds x 3 decision lines** (20260911 / 7 / 99, and thresholds
base-rate / 0.5 / 0.3), the median for the White group moved across
**$38,000, $40,500, $56,000, $62,000, $77,000, $88,000, $205,007, $507,041**,
and the Black-or-African-American median across **$7,000 to $57,500**. The
*ordering between groups reversed* between configurations: at the base-rate
line with seed 20260911 the Black group's median is the smallest of the
three; at seed 7 with a 0.5 line it is the largest.

Across **10 bootstrap resamples** of the study fold, with the model and the
line held fixed (sampling variability alone), the White median ran
$20,500 / $52,000 / $79,000 / $26,000 / $40,500 / $52,000 / $20,500 /
$52,000 / $29,000 / $52,000, and the Black median $2,000 to $18,000. A
four-fold to nine-fold swing from resampling alone.

A number whose value moves by an order of magnitude and whose between-group
ordering reverses under a seed change is not a measurement. Reported as M13 it
would be a fabrication with a command attached.

### 5.2 The root cause, which is itself a stable finding

For roughly **93%** of denied applicants, **no** income increase up to
$1,000,000 crosses the line at all. The model's denial decision on those rows
is dominated by inputs the search holds fixed: debt-to-income band,
loan-to-value, property value. Income cannot move them at any amount.

Each median is therefore taken over the small self-selected remainder that
income *can* move: 4 to 29 people per group in the fixture run. A median over
8 people is not a group statistic, and resampling 8 people is exactly why
§5.1 swings.

### 5.3 The smoother model does not rescue it

The logistic challenger (`fit_logistic_regression`) was tried as the obvious
smooth, monotone alternative. It is **worse**: at the base-rate line it leaves
**0 to 2** reachable applicants per group (Black or African American: 0 of
261; White: 1 of 147, at $322,408). There is no stable number there either,
and a median over one person is not a number at all.

### 5.4 What is NOT being reported instead

The unreachable share **is** stable (0.87 to 0.97 per group across all ten
bootstraps, and 0.89 to 0.93 across the seed/threshold sweep). It is a
genuine, defensible diagnostic about the model.

**It is not M13 and it is not being substituted for M13.** M13's required
shape is a money figure ("need $X more income"); a share is a percentage of a
different thing entirely. Reporting the share under M13's name would be the
proxy substitution this study rules out. It is recorded here as a
diagnostic, and downstream consumers should treat M13 as `UNMEASURABLE`.

---

## 6. Reproducing every number on this page

```bash
.venv/bin/python -m hmda.fairness.recourse          # §4 table, exit 0
.venv/bin/pytest tests/test_recourse.py -q          # exit 0
grep -rn "derived_race\|derived_sex\|derived_ethnicity" src/hmda/fairness/recourse.py   # no output
```

The §5.1 bootstrap numbers come from the shipped
`recourse.stability_check(group_column, n_bootstrap=10)`, a function in
the module rather than a scratch script, so the verdict can be re-derived rather than
taken on trust. The seed/threshold sweep in §5.1 and the logistic result in
§5.3 were run in throwaway scripts on 2026-09-11 and are **not** shipped as
code; they can be reconstructed by passing a differently-seeded
`LGBMClassifier` or `fit_logistic_regression` to `recourse_by_group`, but
treat the exact figures in §5.1's seed sweep and §5.3 as recorded
observations from that session rather than as a command you can re-run
verbatim today.

---

## 7. Limits

1. **Fixture, not nation.** Sections 2 to 6 are DC/WY/VT only. The national
   check in section 8 confirms the verdict and gives no quotable figure.
   Nothing here supports a national claim about any group.
2. **Model, not lender.** See §0. No lender is named; no lender's threshold
   is known or claimed.
3. **Income only.** The other two actionable inputs are declared and not
   searched (§1). A study that searched loan amount might reach more
   applicants; it would answer a different question in different units.
4. **The decision line is a choice.** §5.1 shows the result depends on it,
   which is itself part of why the metric fails.
5. **One feature at a time.** A real applicant might change income and loan
   amount together. This search does not, so where it reports "unreachable"
   the honest reading is "unreachable by income alone", not "no recourse
   exists".
6. **`derived_race` is a coarse derived field**, not self-identification
   and "Race Not Available" is a large category in this
   data; it appears in §4 as its own row rather than being dropped.
7. **Not verified:** whether a more expressive search (multi-feature, or a
   different model class than the two tried) would produce a stable number.
   The two tried did not, and the study stopped there rather than tuning
   until the gap looked interesting.

---

## 8. National check (1,500,000-row sample, three seeds)

```bash
hmda recourse --by-group --source national --sample-n 1500000 --sample-seed S   # S = 0, 1, 2
```

Each run trains the gradient-boosted model on the 2023 and 2024 rows of the
sample (678,035 rows at seed 0) and runs the study on the 380,365 held-out
2025 rows, with the base-rate decision line and the same income-only
search. All three runs exited 0. The command prints the verdict
`M13 VERDICT: UNMEASURABLE.` after the per-group lines.

What the three seeds show:

- **The ordering between groups still reverses.** Native Hawaiian or Other
  Pacific Islander has the highest median of any race group at seed 0
  ($81,000, over 19 of 244 denied applicants) and the lowest at seed 2
  ($18,000, over 9 of 206). A figure that moves a group from worst to best
  on a change of seed cannot be published about that group.
- **The instability is worst in thin groups.** Black or African American,
  White and Race Not Available, with about 450 to 2,200 reachable
  applicants each, moved by $2,000 across the seeds. Native Hawaiian or
  Other Pacific Islander, American Indian or Alaska Native and 2 or more
  minority races, with 9 to 49 reachable applicants, moved by $7,000 to
  $63,000. Free Form Text Only had 0 or 1 reachable applicant. The pattern
  is not strict: Joint, with 45 to 55 reachable applicants, moved by only
  $1,000.
- **The unreachable share is stable.** Apart from Free Form Text Only
  (97.0% to 100%), every race group at every seed fell between 90.7% and
  95.6%. This is the same diagnostic as section 5.4.

This check varies only the sample draw. It does not repeat the fixture
study's sweep over model seeds, decision lines and bootstrap resamples, so
it confirms the verdict and does not replace section 5. The CLI's verdict
text still says each median is taken over "4 to 29 people per group". That
is the fixture count. In these national runs the reachable remainder was 0
to 2,234 people per race group.

These figures were copied from the console output of the three runs. That
output is not committed. Each run took between 12 and 15 minutes, so they are
not re-run in CI.
