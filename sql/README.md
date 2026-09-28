# sql/

Every aggregation used by this repo lives here as a reviewable `.sql` file.
SQL lives in files, not in strings, so a governance reviewer
must be able to read the aggregation. No `.py` file under `src/` may
build a `SELECT` with an f-string, enforced by
`grep -rn "f\"SELECT\|f'SELECT" src/` returning nothing.

Currently holds the adverse-impact and denial-rate queries
(`air_by_group.sql`, `air_by_lei_group.sql`, `rates_by_group.sql`), and
`air_by_group_decisions_only.sql`, the national ratios counting lender
decisions only (action_taken 1, 2 and 3). The screen does not use that
last one; `scripts/four_fifths_decisions_only.py` runs it to show how much
the definition of approval moves the national ratios.
