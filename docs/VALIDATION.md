# VALIDATION

An independent-validation write-up in the SR 11-7 sense. Its four headings are
the core elements SR 11-7 names for model validation, carried out under what
that guidance calls *effective challenge*: **conceptual soundness**,
**outcomes analysis**, **benchmarking**, and **ongoing monitoring**.
The Canadian counterpart, **OSFI Guideline E-23, "Model Risk Management"**,
places the same activity in its model lifecycle as independent review and
validation before approval and use.

This document records what was validated, **by what evidence**, what remains
**not validated**, and what the known limitations are. A validation report
that records only successes is worthless, so the two failures this exercise
found (an unstable mitigation magnitude and an unstable group pair) are
stated first-class in §5, not buried.

---

## 0. Sourcing and independence: read this before the findings

Three caveats bound everything below.

1. **Framework citations are by named principle, not by clause.** Neither
   SR 11-7 nor OSFI E-23 was opened when this document was written, so no
   section or clause number is quoted. The named validation elements above
   are standard terminology and are used as headings; treat every other
   framework reference here as **approximate** until checked against the
   published guidance. Same rule and same source list as
   [`docs/CONTROLS.md`](CONTROLS.md#sources-and-their-limits).
2. **This is not independent in the organisational sense.** SR 11-7's
   independence expectation is about a validation function separate from the
   model's developers with the standing to challenge them. Here the same
   author built and validated the model. What *is* genuinely independent is
   the evidence: every claim in §1-§4 is a test or a command that a reader can
   re-run, and the controls behind them are enumerated and machine-checked in
   `docs/CONTROLS.md` by `hmda verify --governance`.
3. **This model makes no decisions.** It is an offline counterfactual study
   over historical, already-decided mortgage applications. Nothing in this
   repository is deployed, live, or in production; the mitigation output
   header says so on every run (`src/hmda/fairness/mitigate.py:672`, held by
   `tests/test_mitigate.py::test_reporting_never_claims_deployment`). Validation
   findings here therefore bound what may be *claimed*, not what may be
   *decided*.

---

## 1. Conceptual soundness

*Is the model's design defensible for its stated purpose, and are its inputs,
assumptions and limitations documented?*

**Validated, with the evidence.**

| Question | Evidence |
|---|---|
| Are protected attributes kept out of the model? | Inputs come from an explicit allowlist, `src/hmda/model/features.py:151`, not a blocklist of things someone remembered to remove. `tests/test_model.py::test_no_protected_column_reaches_the_built_matrix` asserts it against the built matrix, not against the intent. |
| Are proxies for race excluded, not just race? | `tract_minority_population_percent` is excluded at `src/hmda/model/features.py:143`; `tests/test_model.py::test_the_tract_minority_share_proxy_is_excluded`. |
| Can the model learn the answer from the outcome? | Post-decision pricing fields are excluded (`src/hmda/model/features.py:104`); `tests/test_model.py::test_post_decision_pricing_columns_are_never_features`. The allowlist and blocklist are also asserted disjoint (`tests/test_model.py::test_the_allowlist_and_the_blocklist_do_not_intersect`). |
| Is the evaluation split honest? | Time-ordered, never random (`src/hmda/model/features.py:370`); `tests/test_model.py::test_no_training_row_is_dated_after_any_test_row` and `::test_time_split_is_not_random`. |
| Are the data's sentinel codes handled? | Sentinels become null rather than being cast to a number; `tests/test_sentinels.py::test_income_dollars_is_never_a_silent_float_cast_of_the_sentinel` is the one that matters, because the naive cast turns a sentinel income into a plausible seven-figure number rather than an obvious error. |
| Are exclusions visible? | Every filter is its own row in the waterfall, naming the column and the rule (`src/hmda/clean/waterfall.py:69`); `tests/test_waterfall.py::test_every_filter_row_names_its_column_and_rule`. |
| Are the economic assumptions falsifiable? | They live in `config/economics.yaml`, are printed in the output header under an ILLUSTRATIVE banner (`src/hmda/model/economics.py:92`), and a null or out-of-range value is refused rather than defaulted (`tests/test_economics.py::test_a_null_assumption_is_refused_rather_than_defaulted`). `tests/test_economics.py::test_changing_the_yaml_changes_the_number` proves the printed assumptions are the ones actually used. |

**Weaknesses found, and kept.**

- The economic assumptions are **illustrative, not measured**. HMDA carries no
  margin and no default data, so the margin and loss-given-default rates have
  no external source. The module says so in its own output
  (`src/hmda/model/economics.py:92`: "These are ILLUSTRATIVE assumptions, not
  measurements"). Any dollar figure derived from them is a sensitivity
  statement, never a profit measurement.
- Denial is modelled from `action_taken`, which records the lender's recorded
  disposition, not the underwriting reason. Withdrawn and incomplete
  applications are a known source of noise in that label.
- The dataset has no credit score and no automated-underwriting result. The
  "controlled" disparity is therefore controlled for the credit factors HMDA
  publishes, not for the ones a lender actually underwrites on. This is the
  single largest conceptual limit on the whole audit and is why every output
  says a flag warrants review rather than proves discrimination.

---

## 2. Outcomes analysis

*Does the model perform as expected on data it did not see, and are the
results reported completely?*

**Validated, with the evidence.**

- Both models are always scored on the same held-out time split and returned
  together; there is no public function that returns a challenger score alone
  (`src/hmda/model/evaluate.py:116`,
  `tests/test_model.py::test_no_module_function_returns_a_challenger_score_alone`).
- The challengers beat the base-rate baseline and the gradient-boosted model
  is at least as good as the logistic regression
  (`tests/test_model.py::test_both_challengers_beat_the_base_rate_baseline`,
  `::test_the_gbm_is_at_least_as_good_as_the_logistic_regression`).
- Raw and controlled disparities always ship together
  (`tests/test_controlled.py::test_raw_and_controlled_always_ship_together`,
  `::test_no_public_function_returns_the_raw_gap_alone`). Publishing the raw
  gap alone overstates; publishing the controlled gap alone understates.
- Every mitigation line carries the fairness gain, the accuracy cost and the
  margin cost on the same line
  (`tests/test_mitigate.py::test_reporting_every_line_carries_all_three_numbers`).
  A fairness result with the cost stripped off is the fair-lending equivalent
  of a backtest with no drawdown.
- Groups below the count floor are suppressed as not-a-number, never dropped
  and never shown as zero disparity
  (`tests/test_air.py::test_air_group_below_floor_is_suppressed_not_dropped_not_a_number`,
  `tests/test_controlled.py::test_suppressed_group_never_silently_dropped`).
- The "Race Not Available" share prints beside every race disparity
  (`tests/test_controlled.py::test_race_not_available_share_reported_alongside_race_comparisons`).
  On the fixture this share is large enough that a disparity quoted without it
  is misleading on its own.

**The reported translation of model quality is itself controlled.** No raw AUC
string reaches a user-facing sentence
(`tests/test_translate.py::test_raw_auc_string_never_appears_in_output`), a
statistic with no approved translation is refused rather than guessed at
(`::test_an_unknown_statistic_is_refused_not_guessed`), and a model that fails
to beat the baseline is said to have failed
(`::test_a_model_that_does_not_beat_the_baseline_is_said_so`).

---

## 3. Benchmarking

*Is there an alternative to compare against, and is the comparison fair?*

- The benchmark is a **base-rate baseline**: always predict the training
  denial rate. It ranks at chance by construction
  (`tests/test_model.py::test_base_rate_baseline_ranks_at_chance`), which is
  the point: it is the honest floor a challenger must clear, and every
  reported improvement is stated as a percentage against it
  (`src/hmda/model/translate.py:97`,
  `tests/test_translate.py::test_sentence_names_its_baseline`).
- All three mitigation techniques are measured against the **same** unmitigated
  baseline, so the three are comparable to each other
  (`tests/test_mitigate.py::test_every_technique_is_measured_against_the_same_unmitigated_baseline`).
  The unmitigated gap is reported even though it is not itself a result
  (`::test_the_unmitigated_gap_is_reported_even_though_it_is_not_a_result`).
- The technique set is capped at three, one per pipeline stage, and a fourth
  turns a test red
  (`tests/test_mitigate.py::test_a_fourth_technique_makes_the_cap_test_red`).
  This is a scope control, not a finding.

**Benchmarking gap.** The baseline is a base-rate model, not a credible
underwriting model. A bank validation function would also want a
regulator-recognised or vendor challenger. None exists here, and no claim in
this repository should be read as "better than an industry model".

---

## 4. Ongoing monitoring

*Would a shift in the population be detected?*

- Year-over-year population stability is checked on the model-input allowlist
  (`src/hmda/governance/drift.py:237`,
  `tests/test_drift.py::test_check_drift_defaults_to_the_model_input_allowlist`).
- It is reported to users as a **count of shifted columns**, never as a raw
  score; neither a PSI value nor the word "PSI" may appear in the user-facing
  sentence (`src/hmda/governance/drift.py:124`,
  `tests/test_drift.py::test_sentence_never_contains_a_raw_psi_number_or_the_word_psi`,
  `::test_cli_module_output_never_leaks_psi_or_a_raw_score`).
- The shift threshold is a named constant set by policy, not a magic number at
  a call site (`src/hmda/governance/drift.py:77`,
  `tests/test_drift.py::test_threshold_is_a_named_constant_not_a_magic_number`).
- The input schema is asserted stable across activity years
  (`tests/test_ingest.py::test_measured_header_stable_across_2023_2024_2025`),
  and every ingested file carries a deterministic SHA-256 record
  (`tests/test_ingest.py::test_sha256_file_is_deterministic`).

**Monitoring gap.** There is no *performance* monitoring, only *population*
monitoring. Drift on the inputs is detected; degradation of the model's
ranking quality over time is not tracked, because the model is not in use and
there is no production outcome feed. There is also no alerting: the drift
check runs when a human runs it.

---

## 5. Findings: what came back UNSTABLE

This is the part a validation report usually omits.

### 5.1 The mitigation magnitude was UNSTABLE at a 500,000-row sample

At a **500,000**-row national sample, per-group thresholding cut the
approval-rate gap by 86.3%, 65.2% and 45.9% across seeds 0, 1 and 2. At
**1,500,000** rows the same technique gave 60.1%, 61.4% and 55.7%, so the
result is reported as the range 55% to 61%, not as one seed's figure.
Both sets of figures, and the command for the 1,500,000-row runs, are in
[`docs/VERIFICATION.md`](VERIFICATION.md) section 7, "Recorded model runs".

**The limit on this evidence.** Only the result figures of these runs were
written down. Their console output is not in the repository, so the figures
have not been checked a second time. `results/mitigation_conflict.json`
records this gap and leaves it open. Before any mitigation figure is quoted
outside this repository, the command should be re-run with its output
captured.

**Why it matters.** A spread that wide across seeds at half a million rows
means a mitigation figure taken from one seed at that size is a draw, not a
measurement. The single-seed dollar margin figure in
`results/mitigation.json` was measured at 500,000 rows, and that file marks
its uncertainty as unquantified (`uncertainty_note`).

### 5.2 The pair of groups defining the gap is UNSTABLE across seeds

In the mitigation runs, the two groups whose difference defines the widest
approval-rate gap **changed from seed to seed** (`docs/VERIFICATION.md`
section 7). This is the more damaging of the two findings. A number that is
only noisy is still about the same thing, but a number whose *subject*
changes is not one quantity at all. Any sentence of the form "the gap
between group A and group B is X" is therefore unsupported at present
sample sizes. Only the existence and rough size of *a* gap survives, which
is why the groups are not named.

The recourse metric fails in the same way.
[`docs/RECOURSE.md`](RECOURSE.md) section 5.1 records that across 3 model
seeds and 3 decision lines the median for one group moved from $38,000 to
$507,041 and the ordering between groups reversed, with a bootstrap swing
of $20,500 to $79,000 at a fixed model and line.

### 5.3 A legally contested technique is included, and labelled

`per_group_threshold` applies a different approval cut-off by race. In the US
that is disparate treatment on its face. It is included because a trade-off
study that omits the technique with the largest measured effect is not a
trade-off study, and it carries a LEGALLY CONTESTED warning wherever it is
reported (`src/hmda/fairness/mitigate.py:619`,
`tests/test_mitigate.py::test_reporting_flags_the_per_group_threshold_as_legally_contested`).
**It is measured for completeness and is not presented as approved practice.**

### 5.4 One control has no test

`hmda verify --governance` reports 1 of 38 controls as `NOT IMPLEMENTED`:
**C-38**, the per-figure lineage record, which is still a stub
(`results/governance.json` keys `total_controls`, `backed_by_test`,
`not_implemented_ids`; the reason is in [`docs/CONTROLS.md`](CONTROLS.md)).
The other 37 each name a test that exists. C-35 (the "warrants review,
never a finding of discrimination" disclaimer at `src/hmda/cli.py:266`) and
C-36 (lender-identity suppression at `src/hmda/fairness/air.py:116`) are
tested by driving the real CLI and reading its output.

---

## 6. Not validated

Stated plainly, because a limitation omitted is a limitation asserted not to
exist.

1. **Most numbers come from samples, not the full file.** The four-fifths
   screen and the denial rates run on the full national file
   (`results/four_fifths.json`, `results/denial_rates.json`). The
   controlled gap uses a 200,000-row sample (`results/controlled.json`),
   the models and mitigation a 1,500,000-row sample
   ([`docs/VERIFICATION.md`](VERIFICATION.md) section 7), and SHAP 2,000
   held-out rows (`results/metrics_ledger.json`, M14). The tests run on a
   50,000-row DC/WY/VT fixture. The source label prints on every command so
   the scale cannot be forgotten
   (`tests/test_source.py::test_fixture_source_reports_its_row_count_and_label`),
   and four separate guards refuse an unlabelled frame (controls C-06 to
   C-09 in `docs/CONTROLS.md`).
2. **The instability in §5.1 and §5.2** is not resolved, and the §5.1
   figures were recorded by hand without their console output.
3. **No causal claim is validated, and none is made.** SHAP drivers are
   descriptive of the sample (`src/hmda/governance/explain.py:465`,
   `tests/test_explain.py::test_rendered_doc_states_drivers_are_descriptive_not_causal`).
   Disparity is not discrimination; the missing credit-score variable in §1 is
   why.
4. **No independent validation function.** See §0 caveat 2.
5. **No production or post-implementation review**, because there is no
   implementation. See §0 caveat 3.
6. **Report lineage is unbuilt** (C-38, §5.4), so the documentation set a
   validation function would require is incomplete. The model card is
   generated (`hmda report --model-card`) and checked (`hmda verify --card`).
7. **The framework citations throughout are approximate** (§0 caveat 1).

---

## 7. Conclusion

The **controls** are in good shape: 38 documented, 37 machine-checked
against a named test (`results/governance.json`), and the check itself goes
red when a control loses its test. The **pipeline** discipline is strong:
protected attributes and proxies excluded by allowlist, time-ordered
splits, sentinels handled, exclusions visible, raw and controlled results
shipped together, and fairness gains never reported without their cost.

The **numbers** carry different levels of support. The full-file
four-fifths counts were computed three ways and agree
([`docs/VERIFICATION.md`](VERIFICATION.md) sections 2 and 3). The ranking
figures (57% and 72% better than a base-rate guess) are sample-based and
agreed across three seeds. The mitigation magnitude is unstable at 500,000
rows, is reported only as a range at 1,500,000, and was recorded by hand.
The group pair that defines the gap moves between seeds, and recourse is
UNMEASURABLE. Under SR 11-7's effective-challenge standard the verdict is:
**the controls and the process pass; the sample-based model figures may be
quoted with their sample size; the mitigation results are provisional
until the instability in §5 is re-measured with a named, re-runnable
command whose output is kept.**
