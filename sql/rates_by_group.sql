-- Denial rate by protected-class group.
--
-- {group_column} is an IDENTIFIER placeholder. It is substituted by
-- hmda.fairness.rates._render_sql() only after the caller's column name has
-- been checked against hmda.fairness.rates.ALLOWED_GROUP_COLUMNS and matched
-- against ^[A-Za-z_][A-Za-z0-9_]*$ -- this file never receives untrusted
-- input, and the aggregation logic below is fixed and reviewable regardless
-- of which allowed column is grouped on.
--
-- Purchased loans (action_taken = 6) are excluded here as a defensive
-- second exclusion: the caller's frame is expected to already be the
-- post-waterfall analysis set, but this repeats the
-- exclusion so the query is correct even if that upstream step has not
-- run yet.
--
-- lei is bound as a query PARAMETER (a value, never an identifier), so it
-- is safe to pass straight from the CLI's --min-count-adjacent lender scope.
select
    {group_column} as group_value,
    count(*) as applications,
    sum(case when action_taken = 3 then 1 else 0 end) as denials
from frame
where action_taken != 6
  and (? is null or lei = ?)
group by {group_column}
order by group_value
