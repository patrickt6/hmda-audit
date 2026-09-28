-- Approval rate by protected-class group, for the four-fifths adverse
-- impact ratio. Same identifier-placeholder and parameter
-- rules as sql/rates_by_group.sql -- {group_column} is substituted only
-- after hmda.fairness.air validates it against ALLOWED_GROUP_COLUMNS and
-- ^[A-Za-z_][A-Za-z0-9_]*$; lei is bound as a value parameter, never spliced
-- into the text.
--
-- approval_rate is exposed directly (rather than making the caller derive
-- it from denial_rate) because the four-fifths rule is defined on approval
-- rates: ratio = approval_rate(group) / approval_rate(reference_group).
select
    {group_column} as group_value,
    count(*) as denominator,
    sum(case when action_taken = 3 then 1 else 0 end) as denials,
    1.0 - (sum(case when action_taken = 3 then 1 else 0 end) * 1.0 / count(*)) as approval_rate
from frame
where action_taken != 6
  and (? is null or lei = ?)
group by {group_column}
order by group_value
