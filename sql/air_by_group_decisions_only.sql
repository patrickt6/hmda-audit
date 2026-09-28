-- Approval rate by protected-class group, counting lender decisions only.
-- Same shape and placeholder rules as sql/air_by_group.sql. The one
-- difference is the row filter: this query keeps action_taken 1, 2 and 3
-- (originated, approved but not accepted, denied), the same rows the
-- models use (src/hmda/model/features.py). air_by_group.sql, which the
-- four-fifths screen uses, keeps every code except 6, so withdrawn
-- applications (4), incomplete files (5) and preapproval codes (7, 8)
-- count as not denied there. This file exists so the effect of that
-- choice on the national ratios can be read from a results file
-- (results/four_fifths_decisions_only.json). The screen does not use it.
select
    {group_column} as group_value,
    count(*) as denominator,
    sum(case when action_taken = 3 then 1 else 0 end) as denials,
    1.0 - (sum(case when action_taken = 3 then 1 else 0 end) * 1.0 / count(*)) as approval_rate
from frame
where action_taken in (1, 2, 3)
  and (? is null or lei = ?)
group by {group_column}
order by group_value
