-- Same aggregation as air_by_group.sql, but grouped by lei AND
-- {group_column} in one pass, for hmda.fairness.air.flagged_lenders
-- (the "screened K lenders" count). Querying once per
-- group_column across every lender, instead of once per lender, is the
-- difference between one DuckDB scan and hundreds of them over the same
-- frame; the aggregation logic itself is identical to air_by_group.sql.
--
-- Same identifier-placeholder rule as the other two files in this
-- directory: {group_column} is substituted only after
-- hmda.fairness.rates._validate_group_column has checked it against
-- ALLOWED_GROUP_COLUMNS and ^[A-Za-z_][A-Za-z0-9_]*$.
select
    lei,
    {group_column} as group_value,
    count(*) as denominator,
    sum(case when action_taken = 3 then 1 else 0 end) as denials,
    1.0 - (sum(case when action_taken = 3 then 1 else 0 end) * 1.0 / count(*)) as approval_rate
from frame
where action_taken != 6
  and lei is not null
group by lei, {group_column}
order by lei, group_value
