# LIMITS

What this repo cannot conclude.

- **No individual lender is ever shown to discriminate.** Every disparity
  number is a screening signal, not a legal finding. Any lender-level
  figure's caption carries: "This shows a disparity that warrants review,
  not proof of discrimination by any named lender."
- **`derived_race` is a derived, coarse field**, not self-identification.
  Applicants may report more than one race, and the derived field folds
  those answers into one category.
- **"Race Not Available" is treated differently by different parts of the
  audit, and is never silently dropped.** In the four-fifths screen it is
  kept as its own group: the SQL has no filter on race values
  (`sql/air_by_group.sql`, `sql/air_by_lei_group.sql`), so it gets a ratio
  and can be flagged, and it can be a lender's reference group if it has
  the highest approval rate there (`results/four_fifths.json` gives it
  ratio 0.9441 nationally). In the mitigation study it is not a racial
  group and carries no disparity claim (`NON_GROUP_VALUES` in
  `src/hmda/fairness/mitigate.py`). Its share of applications is printed
  beside the four-fifths table (`hmda audit --air`, 26.73% nationally,
  `results/denial_rates.json` key `race_not_available_share_pct`) and
  beside the controlled comparison (control C-14), because a race gap
  quoted without the share of rows that have no race is misleading.
- **The mitigation layer is an offline counterfactual study.** It reports
  what a mitigated model would have done on historical applications. It is
  never described as deployed, live, or in production. Nothing in this repo
  scores a live applicant.
- **Per-group threshold selection (the post-processing mitigation
  technique) uses protected-class membership at decision time**, which is
  itself legally contested practice in the US. It is measured to complete
  the trade-off study and is not presented as approved practice.
- **The recourse study (M13) ended as UNMEASURABLE.** The method ran, but
  the per-group numbers were not stable across seeds, and the order of the
  groups reversed. That is recorded as the result. No proxy number is
  reported in its place. The evidence is in [`docs/RECOURSE.md`](RECOURSE.md).
