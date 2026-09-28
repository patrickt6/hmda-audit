# CONTROLS

The control-to-standard mapping for this repository (metric M10).

Each row names one control, the supervisory expectation it serves, the
`file:line` in this repository that implements it, and the **test** that
proves it. A control with no test is listed as `NOT IMPLEMENTED` and is
never omitted, because omitting it would quietly raise M10.

`hmda verify --governance` parses this table (`src/hmda/governance/controls.py`)
and exits non-zero if any row names a test that does not exist, or an
`implementing_file` locator that does not resolve to a real line of a real
file. This document is the source of truth; the code only checks it.

**Read the sourcing caveat at the bottom before quoting any framework
citation in this table.** Neither framework document was opened when this
file was written, so the citations name principles in words and carry no
clause or section numbers on purpose.

---

## The two frameworks, in one paragraph each

**US Federal Reserve SR 11-7 / OCC Bulletin 2011-12, "Supervisory Guidance on
Model Risk Management" (2011).** Model risk is the risk of adverse
consequences from decisions based on incorrect or misused model output. The
guidance is built on three pillars: (1) robust model development,
implementation and use, including sound data and a clear statement of
intended use; (2) independent model validation, whose named core elements are
evaluation of conceptual soundness, ongoing monitoring, and outcomes analysis
including benchmarking, all carried out with "effective challenge"; and (3)
governance, policies and controls, covering documentation, a model inventory,
roles and responsibilities, and internal audit.

**OSFI Guideline E-23, "Model Risk Management".** The Canadian prudential
expectation. It frames model risk management around an enterprise-wide
framework and a model lifecycle: rationale and design, data, development,
independent review and validation, approval before use, deployment, ongoing
monitoring and performance review, and decommission. It expects models to be
inventoried and risk-rated so that the depth of review is proportionate to
the model's risk, and it expects the limitations of a model to be documented
and communicated to its users.

---

## Control table

| control_id | assertion | implementing_file | test | SR 11-7 clause | OSFI E-23 clause |
|---|---|---|---|---|---|
| C-01 | Protected-class fields are excluded from model inputs by an explicit allowlist, not by hoping no one adds them. | src/hmda/model/features.py:151 | tests/test_model.py::test_no_protected_column_reaches_the_built_matrix | Model development and implementation: model inputs are deliberate and documented (principle named in words, no clause cited) | Model design and development: inputs are justified and documented (principle named in words, no clause cited) |
| C-02 | A neighbourhood racial-composition field is excluded as a proxy for race, not merely the race field itself. | src/hmda/model/features.py:143 | tests/test_model.py::test_the_tract_minority_share_proxy_is_excluded | Model development: known input weaknesses and proxy effects are identified rather than assumed away | Model design and development: proxy and unintended-bias risk in inputs is addressed |
| C-03 | Post-decision pricing fields cannot become features, so the model cannot learn the answer from the outcome. | src/hmda/model/features.py:104 | tests/test_model.py::test_post_decision_pricing_columns_are_never_features | Model development: data quality and appropriateness for the stated purpose | Model design and development: data appropriate to the model's stated use |
| C-04 | Train and test are split by time, never randomly, so no future row informs a past prediction. | src/hmda/model/features.py:370 | tests/test_model.py::test_no_training_row_is_dated_after_any_test_row | Model validation, outcomes analysis: performance is measured out-of-sample in a way that reflects real use | Independent review and validation: performance evidence reflects the intended use |
| C-05 | Every challenger score ships beside a benchmark baseline score; no public function returns the challenger alone. | src/hmda/model/evaluate.py:116 | tests/test_model.py::test_no_module_function_returns_a_challenger_score_alone | Model validation: benchmarking against an alternative is a named core element of outcomes analysis | Independent review and validation: challenge against alternatives |
| C-06 | `run_eval` refuses an in-memory frame with no source label, so a sample number cannot print under a national or fixture label it did not come from. | src/hmda/model/evaluate.py:162 | tests/test_model.py::test_run_eval_rejects_a_frame_with_no_source_label | Governance, policies and controls: results are documented with the data they came from | Model lifecycle governance: outputs are traceable to the data used |
| C-07 | `mitigate.prepare` refuses an in-memory frame with no source label, same reason. | src/hmda/fairness/mitigate.py:276 | tests/test_mitigate.py::test_prepare_rejects_a_frame_with_no_source_label | Governance, policies and controls: results are documented with the data they came from | Model lifecycle governance: outputs are traceable to the data used |
| C-08 | `recourse.run_study` refuses an in-memory frame with no source label, same reason. | src/hmda/fairness/recourse.py:371 | tests/test_recourse.py::test_run_study_rejects_a_frame_with_no_source_label | Governance, policies and controls: results are documented with the data they came from | Model lifecycle governance: outputs are traceable to the data used |
| C-09 | `explain.run_explain` refuses an in-memory frame with no source label, same reason. | src/hmda/governance/explain.py:373 | tests/test_explain.py::test_run_explain_rejects_a_frame_with_no_source_label | Governance, policies and controls: results are documented with the data they came from | Model lifecycle governance: outputs are traceable to the data used |
| C-10 | Every data source states its own scope and row count in a label that every command prints. | src/hmda/clean/source.py:244 | tests/test_source.py::test_fixture_source_reports_its_row_count_and_label | Governance and documentation: the data behind a result is stated, not implied | Data: the data used by a model is identified and documented |
| C-11 | Sampling is seeded and reproducible, so a quoted number can be regenerated. | src/hmda/clean/source.py:130 | tests/test_source.py::test_sample_is_reproducible | Governance, policies and controls: documentation sufficient for an independent party to reproduce the work | Independent review and validation: a reviewer can reproduce the result |
| C-12 | A group below the minimum-count floor is suppressed and explicitly not-a-number, never silently dropped and never reported as zero disparity. | src/hmda/fairness/floors.py:13 | tests/test_air.py::test_air_group_below_floor_is_suppressed_not_dropped_not_a_number | Model validation: limitations of the result are identified rather than hidden | Model limitations are documented and communicated to users |
| C-13 | The raw disparity and the controlled disparity are always returned together; no public function returns the raw gap alone. | src/hmda/fairness/controlled.py:286 | tests/test_controlled.py::test_raw_and_controlled_always_ship_together | Model validation, conceptual soundness: a result is not presented without the adjustment that qualifies it | Independent review and validation: conclusions carry their qualifications |
| C-14 | The "Race Not Available" share is reported beside every race disparity, so the reader sees the coverage behind the number. | src/hmda/fairness/controlled.py:317 | tests/test_controlled.py::test_race_not_available_share_reported_alongside_race_comparisons | Model development: data quality and coverage limitations are surfaced with the result | Data: known data-quality limitations are disclosed |
| C-15 | Sentinel codes are never silently cast to a number; a sentinel becomes null, not a 1,111,000-dollar income. | src/hmda/clean/sentinels.py:123 | tests/test_sentinels.py::test_income_dollars_is_never_a_silent_float_cast_of_the_sentinel | Model development: data quality controls on model inputs | Data: data quality is assessed before use |
| C-16 | Every exclusion appears as its own row in the row-count waterfall, naming the column and the rule that dropped the rows. | src/hmda/clean/waterfall.py:69 | tests/test_waterfall.py::test_every_filter_row_names_its_column_and_rule | Governance and documentation: the transformation from raw data to analysis set is reviewable | Data: data preparation and exclusions are documented |
| C-17 | The input schema is asserted stable across activity years; a header change is a hard failure, not a silent shift. | src/hmda/ingest/schema.py:66 | tests/test_ingest.py::test_measured_header_stable_across_2023_2024_2025 | Ongoing monitoring: changes in inputs or environment are detected | Ongoing monitoring: input stability is monitored over time |
| C-18 | Every ingested file carries a deterministic SHA-256 record, so the data behind a number is identifiable after the fact. | src/hmda/ingest/checksum.py:37 | tests/test_ingest.py::test_sha256_file_is_deterministic | Governance, policies and controls: documentation and auditability of model data | Model lifecycle governance: data lineage is retained |
| C-19 | Year-over-year population stability is monitored on the model-input allowlist and reported as a count of shifted columns. | src/hmda/governance/drift.py:237 | tests/test_drift.py::test_check_drift_defaults_to_the_model_input_allowlist | Ongoing monitoring, a named core element of validation | Ongoing monitoring and performance review |
| C-20 | The drift threshold is a named constant, not a magic number chosen at a call site. | src/hmda/governance/drift.py:77 | tests/test_drift.py::test_threshold_is_a_named_constant_not_a_magic_number | Governance, policies and controls: thresholds are set by policy and documented | Model risk framework: thresholds and tolerances are defined, not ad hoc |
| C-21 | Drift is reported to users as a shifted-column count; no raw PSI value and not even the word "PSI" reaches the user-facing sentence. | src/hmda/governance/drift.py:124 | tests/test_drift.py::test_sentence_never_contains_a_raw_psi_number_or_the_word_psi | Governance: model output is communicated in terms its audience can act on | Model limitations and results are communicated to users in usable form |
| C-22 | A statistic with no approved plain-English translation is refused, never guessed at. | src/hmda/model/translate.py:171 | tests/test_translate.py::test_an_unknown_statistic_is_refused_not_guessed | Governance: reported model results are the ones policy allows, not improvised | Model risk framework: reporting follows a defined standard |
| C-23 | The mitigation technique registry is capped at exactly three techniques, one per pipeline stage; a fourth turns a test red. | src/hmda/fairness/mitigate.py:211 | tests/test_mitigate.py::test_the_registry_has_exactly_three_techniques | Model development: scope of the exercise is fixed and documented | Model design: the model's scope and boundaries are defined |
| C-24 | Every mitigation line reports the fairness gain, the accuracy cost and the margin cost together; a gain never ships without its cost. | src/hmda/fairness/mitigate.py:689 | tests/test_mitigate.py::test_reporting_every_line_carries_all_three_numbers | Model validation, outcomes analysis: performance is reported with its trade-offs, not selectively | Independent review and validation: results are reported completely, including adverse findings |
| C-25 | The per-group threshold technique carries a LEGALLY CONTESTED warning naming it as disparate treatment on its face in the US, measured for completeness and not presented as approved practice. | src/hmda/fairness/mitigate.py:619 | tests/test_mitigate.py::test_reporting_flags_the_per_group_threshold_as_legally_contested | Model use: known constraints on the acceptable use of a result are stated with it | Model limitations, including legal and regulatory constraints, are documented |
| C-26 | No mitigation output may be described as deployed, live, or in production; every report header states it is an offline counterfactual study over historical applications. | src/hmda/fairness/mitigate.py:672 | tests/test_mitigate.py::test_reporting_never_claims_deployment | Model use: the actual use of the model is stated and not overstated | Model lifecycle: deployment status is explicit; an unapproved model is not described as in use |
| C-27 | Every mitigation result is measured against the same unmitigated baseline, so the three techniques are comparable. | src/hmda/fairness/mitigate.py:635 | tests/test_mitigate.py::test_every_technique_is_measured_against_the_same_unmitigated_baseline | Model validation: benchmarking is against a common reference | Independent review and validation: comparability of evidence |
| C-28 | The economic assumptions behind the margin number live in a config file and are printed in the output header with an ILLUSTRATIVE banner, so the profit number is falsifiable. | src/hmda/model/economics.py:92 | tests/test_economics.py::test_the_header_prints_both_assumptions_the_gate_names | Model development: assumptions are documented and their effect on output is shown | Model design: assumptions are documented, justified, and their sensitivity understood |
| C-29 | A null or out-of-range economic assumption is refused rather than silently defaulted. | src/hmda/model/economics.py:151 | tests/test_economics.py::test_a_null_assumption_is_refused_rather_than_defaulted | Model implementation: input validation, failure is loud rather than silent | Model design and development: input validation and controls |
| C-30 | No protected attribute is offered to an applicant as an actionable recourse lever. | src/hmda/fairness/recourse.py:70 | tests/test_recourse.py::test_actionable_features_hold_no_protected_attribute | Model use: the result is constrained to the use it is fit for | Model limitations and appropriate use are documented |
| C-31 | A recourse group below the count floor is flagged and gets no number, rather than being dropped. | src/hmda/fairness/recourse.py:327 | tests/test_recourse.py::test_small_group_is_flagged_not_dropped_and_gets_no_number | Model validation: limitations of the result are identified rather than hidden | Model limitations are documented and communicated |
| C-32 | The explainability pack states that drivers are descriptive of the sample, not causal claims about the model's treatment of a group. | src/hmda/governance/explain.py:465 | tests/test_explain.py::test_rendered_doc_states_drivers_are_descriptive_not_causal | Model validation: the interpretation of model output is bounded by what the evidence supports | Model limitations are documented and communicated to users |
| C-33 | The explainability run is deterministic across two full runs, so an explanation quoted today reproduces tomorrow. | src/hmda/governance/explain.py:346 | tests/test_explain.py::test_run_explain_is_deterministic_across_two_full_runs | Governance, policies and controls: documentation sufficient for an independent party to reproduce the result | Independent review and validation: a reviewer can reproduce the result |
| C-34 | Every explainability driver label resolves to a data-dictionary entry; no unexplained raw column name is surfaced to a reader. | src/hmda/governance/explain.py:397 | tests/test_explain.py::test_every_reported_overall_driver_resolves_to_a_data_dictionary_entry | Governance and documentation: a complete data dictionary supports the model record | Data: model inputs are defined in documentation |
| C-35 | The CLI states on every disparity output that a flag is a statistical disparity warranting review and NEVER a finding of discrimination. | src/hmda/cli.py:266 | tests/test_disparity_language.py::test_air_output_disclaims_a_finding_of_discrimination | Model use: the permitted interpretation of the output is stated with the output | Model limitations and appropriate use are communicated to users |
| C-36 | No individual lender is ever named as discriminating; flagged lender identity is an internal artifact only. | src/hmda/fairness/air.py:116 | tests/test_no_lender_named.py::test_air_output_names_no_flagged_lender | Model use: constraints on the disclosure and use of model output | Model limitations and appropriate use are documented |
| C-37 | The model card states verbatim that this model is not used to make credit decisions. | src/hmda/governance/model_card.py:18 | tests/test_model_card_language.py::test_the_shipped_model_card_contains_the_sentence_verbatim | Governance and documentation: a model record states intended and out-of-scope use | Model rationale and design: intended use and exclusions are documented |
| C-38 | Every figure and table in the report carries a lineage record: source file, checksum, SQL file and commit. | src/hmda/governance/lineage.py:13 | NOT IMPLEMENTED | Governance, policies and controls: documentation and auditability of reported results | Model lifecycle governance: results are traceable to their inputs |

---

## The one control that is NOT IMPLEMENTED, and why

This is listed, not hidden, because M10 counts documented controls and
`hmda verify --governance` prints the backed and unbacked counts separately.

- **C-38.** The lineage record is still a stub: `record_lineage` and
  `verify_lineage` in `src/hmda/governance/lineage.py` both raise
  `NotImplementedError`. The committed figures have a source entry in
  `results/figures_provenance.json`, but not the full record C-38 asks for
  (checksum, SQL file and commit). A test written against the stub could
  only check that the stub exists, which says nothing about the control.
  C-38 stays `NOT IMPLEMENTED` until there is behaviour to protect.

### How C-35, C-36 and C-37 are tested

- **C-35** and **C-36** are tested through the real CLI.
  `tests/test_disparity_language.py` and `tests/test_no_lender_named.py`
  run it through click's `CliRunner` against `--source fixture` and assert
  on the rendered output, not on a copy of it. Both were shown to fail when
  the behaviour is removed. On 2026-09-13, deleting the `click.echo`
  disclaimers turned 4 of the 5 language tests red, and printing the
  flagged LEI list turned 2 of the 4 lender tests red.
- **C-37** is tested against the shipped card.
  `tests/test_model_card_language.py` asserts that
  `NO_CREDIT_DECISIONS_SENTENCE` still states the exclusion and that
  `docs/MODEL-CARD.md` contains it verbatim. `docs/MODEL-CARD.md` is
  generated by `render_model_card`, and `verify_card` (behind
  `hmda verify --card`) reports a card that is missing the sentence
  (`tests/test_model_card.py::test_a_card_missing_the_c37_sentence_fails_verification`).

---

## Sources and their limits

**Binding sourcing statement.** SR 11-7 and OSFI E-23 were **not** opened
when this file was written. The repository's rule is to cite a source only
after reading it, so a clause or section number that was not read is not
written down. Every citation in the two framework columns therefore:

1. names the framework and the **principle in words**, not a section number;
2. is drawn from general knowledge of the two documents and is **approximate**
   until checked against the published text;
3. must be verified against the primary sources below before any of this is
   presented externally.

Primary sources to verify against, by title (no URL was fetched, so none is
given here):

- Board of Governors of the Federal Reserve System, **SR 11-7**, "Guidance on
  Model Risk Management", 4 April 2011, and its identical OCC companion,
  **OCC Bulletin 2011-12**, "Sound Practices for Model Risk Management:
  Supervisory Guidance on Model Risk Management".
- Office of the Superintendent of Financial Institutions (Canada),
  **Guideline E-23, "Model Risk Management"**. Note that E-23 has been
  revised since its original issue; check which version applies before
  citing it.

The one thing in the framework columns that is **not** approximate is the
naming of SR 11-7's three core validation elements (evaluation of conceptual
soundness, ongoing monitoring, and outcomes analysis) and the term
"effective challenge". Those are standard and are used here as headings in
`docs/VALIDATION.md`.
