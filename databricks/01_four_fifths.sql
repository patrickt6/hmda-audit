-- Four-fifths (adverse impact ratio) screen by race, in Databricks Spark SQL.
-- Same logic as sql/air_by_group.sql (the DuckDB version), run on the Delta table.
--
-- The rule: approval_rate(group) / approval_rate(reference group).
-- Below 0.8 means "flag this for review". A flag is not proof of discrimination.

-- A CTE (WITH ... AS) is a named temporary result, used by the SELECT below.
WITH g AS (
  SELECT
    derived_race AS grp,                                        -- the group we compare
    count(*) AS n,                                              -- applications in the group
    sum(CASE WHEN action_taken = '3' THEN 1 ELSE 0 END) AS denials,  -- HMDA code 3 = denied
    1.0 - sum(CASE WHEN action_taken = '3' THEN 1 ELSE 0 END) * 1.0 / count(*) AS approval_rate
  FROM workspace.hmda.lar_50k                                   -- the Delta table (catalog.schema.table)
  WHERE action_taken <> '6'                                     -- code 6 = loan bought by the lender, no decision made, so exclude
  GROUP BY derived_race
)
SELECT
  grp,
  n,
  denials,
  round(approval_rate, 4) AS approval_rate,
  -- scalar subquery: the White approval rate, used as the reference
  round(approval_rate / (SELECT approval_rate FROM g WHERE grp = 'White'), 4) AS air,
  approval_rate / (SELECT approval_rate FROM g WHERE grp = 'White') < 0.8 AS flag
FROM g
ORDER BY n DESC;
