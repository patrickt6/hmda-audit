<!--
GENERATED FILE. Do not hand-edit.
Regenerate: hmda report --model-card
Check:      hmda verify --card   (fails if any number here is not a measured fact)
Generator:  src/hmda/governance/model_card.py
-->

# Model card: HMDA mortgage denial model

This card is generated from the audited database and from the recorded measurement runs. It is not written by hand. Every number below appears in the fact table at the end of the card with the basis it was measured on and a locator a reviewer can open.

## What the model is

Two models over United States mortgage application records: a base-rate reference that scores every applicant with the same denial rate, and a gradient-boosted challenger trained on 16 loan and property attributes. The study question is whether denial outcomes differ across protected groups once loan and property attributes are held level.

## Intended use

Offline fairness measurement and model-risk documentation over historical, public application records. The model exists to support an audit: to rank applications by denial risk so that outcome gaps between groups can be compared at a matched risk level, and to give a mitigation study something to mitigate.

## Out-of-scope use

This model is not used to make credit decisions.

It was never deployed, never served a request, never scored a real applicant, and never influenced any lending outcome. It ran offline over records of applications that were already decided by their lenders years before this work began. It is not underwriting software, not a pricing tool, not a scoring service, and not evidence about any named institution.

It is also not a finding of discrimination. A measured gap between groups is a statistical disparity that warrants review. Proving discrimination requires evidence this data does not hold.

## Data

Public national mortgage application records: 36,734,685 applications from 5,329 lenders over 3 years, 2023 to 2025. Counts are read from the database at render time, over every row, not from a sample.

Training and evaluation for the recorded model runs used a stratified sample of 1,500,000 rows rather than the whole file, because the model stage is memory-bound where the aggregation stages are not. Every model number in this card is therefore a sample number and says so.

## Protected-attribute handling

Race, ethnicity, sex and age are not model inputs. The feature matrix is built from an allowlist, so a protected column cannot become a feature by being forgotten, and the build raises if a protected column reaches the matrix. Protected attributes are used only on the measurement side, to group outcomes after the fact.

Excluding them does not make the model blind to them. Loan and property attributes carry information about who applied, so a gap can survive the exclusion. That is the reason the audit measures outcomes rather than trusting the input list.

Race is unreported on 26.73% of national applications. That share is large enough to move any group comparison, so it travels with every disparity number in this project.

## How well it ranks

Against a base-rate guess, which is the honest floor, the logistic reference ranks denials 57% better than chance and the gradient-boosted challenger 72% better, on a sample of 1,500,000 rows, repeated at three random seeds that agreed to the printed precision. Better than chance at ranking is not the same as accurate for an individual applicant, and this card makes no individual-level accuracy claim.

## Mitigation, and what it costs

A mitigation study measured what closing the gap costs. Applying a separate approval cut-off per group cut the widest between-group approval-rate gap by 55% to 61%, at a cost of 0.56 to 0.59 accuracy points, on a sample of 1,500,000 rows across three seeds. The range is the result. No single seed is the headline.

That technique applies a different approval cut-off by race, which is disparate treatment on its face in the United States. It is measured for completeness of the trade-off study and is not presented as approved practice.

## Known limits

- **The recourse study is unmeasurable, not merely unmeasured.** It asked how much an applicant would have to change to cross the approval line, compared across groups. The per-group medians reorder between random seeds, so the comparison has no stable answer at this sample size. It is reported as a negative result and its medians are not quoted.
- **The pair of groups defining the widest gap is unstable.** The gap metric is a maximum over all group pairs, and the pair realising that maximum changed between seeds. The magnitude is reportable; the identity of the two groups is not, and is named nowhere in this project as a result.
- **Protected attributes are not model inputs**, so the model cannot be read as measuring a lender's intent, and the disparities measured here are outcome differences, not mechanisms.
- **Model numbers are sample-based.** The counts and shares in the data section cover every row; the ranking and mitigation numbers do not, and are labelled with their sample size wherever they appear.
- **The margin study is not trustworthy enough to quote.** It has one seed and no measured spread, and its sign reversed against the earlier fixture run. No money figure appears in this card.
- **Speed and memory are measured, but not quoted here.** The comparison against a single-machine pandas baseline was run on the full national file (results/engineering.json, from bench/compare_pandas.py). It describes the pipeline, not the model, so this card does not repeat the figures.
- **Race is unreported on a large minority of records** (see the data section). Group comparisons are conditional on who reported.
- **No lender is named** anywhere in this project, by policy, and no result is attributed to an institution.

## Governance

38 controls are documented for this project, of which 37 are backed by an automated test. Both numbers travel together: the first counts what is written down, the second counts what is checked.

## Fact table: every number in this card

| number | value | basis | source |
| --- | --- | --- | --- |
| applications | 36,734,685 | full-file, measured | `SELECT count(*) FROM lar -- data/hmda.duckdb table lar` |
| lenders | 5,329 | full-file, measured | `SELECT COUNT(DISTINCT lei) FROM lar -- data/hmda.duckdb table lar` |
| years | 3 | full-file, measured | `SELECT DISTINCT activity_year FROM lar -- data/hmda.duckdb table lar` |
| first_year | 2023 | full-file, measured | `SELECT min(activity_year) FROM lar -- data/hmda.duckdb table lar` |
| last_year | 2025 | full-file, measured | `SELECT max(activity_year) FROM lar -- data/hmda.duckdb table lar` |
| race_not_available_share | 26.73% | full-file, measured | `SELECT count(*) FILTER (WHERE derived_race = 'Race Not Available') / count(*) FROM lar -- data/hmda.duckdb table lar` |
| sample_n | 1,500,000 | sample-based, n=1,500,000, seeds 0/1/2 | `docs/VERIFICATION.md 'Recorded model runs', ranking-quality status line` |
| ranking_logistic | 57% | sample-based, n=1,500,000, seeds 0/1/2 | `docs/VERIFICATION.md 'Recorded model runs', ranking quality` |
| ranking_boosted | 72% | sample-based, n=1,500,000, seeds 0/1/2 | `docs/VERIFICATION.md 'Recorded model runs', ranking quality` |
| gap_cut_low | 55% | sample-based, n=1,500,000, seeds 0/1/2 | `docs/VERIFICATION.md 'Recorded model runs', mitigation` |
| gap_cut_high | 61% | sample-based, n=1,500,000, seeds 0/1/2 | `docs/VERIFICATION.md 'Recorded model runs', mitigation` |
| accuracy_cost_low | 0.56 | sample-based, n=1,500,000, seeds 0/1/2 | `docs/VERIFICATION.md 'Recorded model runs', mitigation table` |
| accuracy_cost_high | 0.59 | sample-based, n=1,500,000, seeds 0/1/2 | `docs/VERIFICATION.md 'Recorded model runs', mitigation table` |
| controls_documented | 38 | registry count | `docs/CONTROLS.md rows parsed by hmda.governance.controls.load_controls` |
| controls_backed | 37 | registry count | `docs/CONTROLS.md rows whose test column is not 'NOT IMPLEMENTED'` |
| feature_columns | 16 | code constant | `src/hmda/model/features.py FEATURE_COLUMNS` |
