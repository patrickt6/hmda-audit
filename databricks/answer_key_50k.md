# Answer key: four-fifths ratio by derived_race on tests/fixtures/hmda_50k.parquet

Computed with DuckDB 2026-09-28 on the raw fixture (action_taken as text; '3' = denied; '6' = purchased loan, excluded). Reference group: White. air = approval_rate / White approval_rate; below 0.8 = flag.
This is the raw-file version, not the cleaned `frame` the repo's src/hmda/fairness/air.py uses, so it can differ from repo outputs.

Query:
```sql
with g as (
  select derived_race as grp, count(*) n,
         sum(case when action_taken='3' then 1 else 0 end) denials,
         1.0 - sum(case when action_taken='3' then 1 else 0 end)*1.0/count(*) approval_rate
  from 'tests/fixtures/hmda_50k.parquet'
  where action_taken <> '6'
  group by derived_race)
select grp, n, denials, round(approval_rate,4) approval_rate,
       round(approval_rate / (select approval_rate from g where grp='White'),4) air
from g order by n desc
```

| group | n | denials | approval_rate | air |
|---|---|---|---|---|
| White | 27426 | 4152 | 0.8486 | 1.0 |
| Race Not Available | 9166 | 1791 | 0.8046 | 0.9481 |
| Black or African American | 4997 | 1390 | 0.7218 | 0.8506 |
| Asian | 1201 | 176 | 0.8535 | 1.0057 |
| Joint | 1075 | 133 | 0.8763 | 1.0326 |
| American Indian or Alaska Native | 263 | 82 | 0.6882 | 0.811 |
| 2 or more minority races | 101 | 23 | 0.7723 | 0.91 |
| Native Hawaiian or Other Pacific Islander | 58 | 20 | 0.6552 | 0.7721 |
| Free Form Text Only | 13 | 6 | 0.5385 | 0.6345 |
