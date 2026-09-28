# Claims

Each row of the README's results table carries an invisible marker such as
`<!-- ct:hmda-four-fifths -->`. The marker links the row to a claim in the
claimtrail store in `.claimtrail/` (claimtrail is installed from PyPI).
`scripts/record_claims.py` builds that store from the committed files in
`results/`. It registers each results file as a computation and states one
claim per row, each with structured assertions such as
`outputs.lenders_flagged == 614`.

CI (`.github/workflows/claims.yml`) runs four steps on every push:

```
python scripts/record_claims.py          # results still match their records
claimtrail check --paper hmda-audit      # every structured assertion holds
claimtrail audit-report README.md        # every number in a marked row is supported
claimtrail verifications --check-chain   # the verification log is intact
```

`audit-report` reads every number in a marked row and fails if one of them
is not in the linked record, within the precision it is written with. So a
README edit that changes 614 to 615 fails CI, and so does a results file
that changes under an existing claim.

CI cannot rerun the national audit, because the 36.7M rows are not in CI.
It checks that the README agrees with the committed results, not that the
results are right. `docs/VERIFICATION.md` covers that. The test count
(`hmda-tests`) is a reported number, read from `results/metrics_ledger.json`.
A rerun can be logged against it by hand:

```
.venv/bin/pytest -q                      # note the "N passed" line
echo '{"passed": N, "skipped": M}' > /tmp/run.json
claimtrail verify <computation id> --against /tmp/run.json
```

## Changing a number after review

`scripts/record_claims.py` fails with a collision when a results file under
an existing claim changes. That is on purpose. When a code fix moves a
number, record the change like this:

1. Commit the code fix first. Rerun the command that produces the number at
   that commit.
2. Update the number in the results file, or in
   `scripts/gen_dashboard_artifacts.py` for the files it writes, and add a
   `code_sha` field with the fix commit to the changed entry. Then run
   `.venv/bin/python scripts/gen_dashboard_artifacts.py` and restore any
   results file that should not change (`git diff --stat results/`).
3. Update the `expect` values in `scripts/record_claims.py` and the README
   row.
4. Run `.venv/bin/python scripts/record_claims.py --reviewed <claim ids>`,
   naming only the claims you reviewed. claimtrail puts `code_sha` into the
   record id, so the new result gets a new record and the old record stays
   in the store. The claim row is then replaced in place. Run the script
   again with no flag; it must exit cleanly, since that is what CI runs.
5. Run `claimtrail check --paper hmda-audit`, `claimtrail audit-report README.md`,
   `claimtrail verifications --check-chain` and `claimtrail lint`. Do not run
   `claimtrail gc`: it would delete the old records.

claimtrail 0.5.1's `claim()` raises a collision when a claim id is stated
again with a different record, and it has no `force` option. So step 4
replaces the row itself, and only for the ids passed to `--reviewed`, after
`claim()` has checked every `expect` value against the new record.
