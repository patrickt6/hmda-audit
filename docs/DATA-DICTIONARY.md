# DATA-DICTIONARY
GENERATED from `data/hmda.duckdb`'s real header, 99 columns, 133289 total rows across the state-years ingested so far (measured 2026-09-11, `hmda verify --counts` and this script). Never hand-typed. Command: `.venv/bin/python -c "from hmda..."` reading `describe lar` and per-column DISTINCT/COUNT queries.

## Header stability
Confirmed MEASURED 2026-09-11: DC's 2023, 2024 and 2025 parquet files (fetched live via `hmda ingest`) have **identical** 99-column headers, same order. Command: `describe select * from read_parquet('data/parquet/DC_<year>.parquet')` for each year, compared as tuples: all three are equal. No reconciliation step is needed for the years/states ingested so far.

## Columns
| Column | DuckDB dtype | Distinct values | Null share | Sentinels found |
|---|---|---|---|---|
| activity_year | VARCHAR | 3 | 0.0000 | none found |
| lei | VARCHAR | 1017 | 0.0000 | none found |
| derived_msa-md | VARCHAR | 6 | 0.0000 | none found |
| state_code | VARCHAR | 3 | 0.0000 | none found |
| county_code | VARCHAR | 39 | 0.0000 | NA (1254) |
| census_tract | VARCHAR | 554 | 0.0000 | NA (1336) |
| conforming_loan_limit | VARCHAR | 4 | 0.0000 | NA (1052) |
| derived_loan_product_type | VARCHAR | 8 | 0.0000 | none found |
| derived_dwelling_category | VARCHAR | 4 | 0.0000 | none found |
| derived_ethnicity | VARCHAR | 5 | 0.0000 | none found |
| derived_race | VARCHAR | 9 | 0.0000 | none found |
| derived_sex | VARCHAR | 4 | 0.0000 | none found |
| action_taken | VARCHAR | 8 | 0.0000 | none found |
| purchaser_type | VARCHAR | 11 | 0.0000 | none found |
| preapproval | VARCHAR | 2 | 0.0000 | none found |
| loan_type | VARCHAR | 4 | 0.0000 | none found |
| loan_purpose | VARCHAR | 6 | 0.0000 | none found |
| lien_status | VARCHAR | 2 | 0.0000 | none found |
| reverse_mortgage | VARCHAR | 3 | 0.0000 | none found |
| open-end_line_of_credit | VARCHAR | 3 | 0.0000 | none found |
| business_or_commercial_purpose | VARCHAR | 3 | 0.0000 | none found |
| loan_amount | VARCHAR | 721 | 0.0000 | none found |
| loan_to_value_ratio | VARCHAR | 33833 | 0.0000 | NA (44453), Exempt (3111) |
| interest_rate | VARCHAR | 2286 | 0.0000 | NA (44506) |
| rate_spread | VARCHAR | 11021 | 0.0000 | NA (66899) |
| hoepa_status | VARCHAR | 3 | 0.0000 | none found |
| total_loan_costs | VARCHAR | 48163 | 0.0000 | NA (66751) |
| total_points_and_fees | VARCHAR | 381 | 0.0000 | NA (128855) |
| origination_charges | VARCHAR | 27518 | 0.0000 | NA (65693) |
| discount_points | VARCHAR | 19649 | 0.2712 | NA (66475) |
| lender_credits | VARCHAR | 9608 | 0.3376 | NA (66591) |
| loan_term | VARCHAR | 341 | 0.0000 | NA (1370) |
| prepayment_penalty_term | VARCHAR | 9 | 0.0000 | NA (127206) |
| intro_rate_period | VARCHAR | 54 | 0.0000 | NA (102270) |
| negative_amortization | VARCHAR | 3 | 0.0000 | none found |
| interest_only_payment | VARCHAR | 3 | 0.0000 | none found |
| balloon_payment | VARCHAR | 3 | 0.0000 | none found |
| other_nonamortizing_features | VARCHAR | 3 | 0.0000 | none found |
| property_value | VARCHAR | 807 | 0.0000 | NA (27328), Exempt (3112) |
| construction_method | VARCHAR | 2 | 0.0000 | none found |
| occupancy_type | VARCHAR | 3 | 0.0000 | none found |
| manufactured_home_secured_property_type | VARCHAR | 4 | 0.0000 | none found |
| manufactured_home_land_property_interest | VARCHAR | 6 | 0.0000 | none found |
| total_units | VARCHAR | 9 | 0.0000 | none found |
| multifamily_affordable_units | VARCHAR | 29 | 0.0000 | NA (129440) |
| income | VARCHAR | 2280 | 0.0000 | NA (18598) |
| debt_to_income_ratio | VARCHAR | 21 | 0.0000 | NA (45584) |
| applicant_credit_score_type | VARCHAR | 15 | 0.0000 | none found |
| co-applicant_credit_score_type | VARCHAR | 16 | 0.0000 | none found |
| applicant_ethnicity-1 | VARCHAR | 8 | 0.0005 | none found |
| applicant_ethnicity-2 | VARCHAR | 6 | 0.9735 | none found |
| applicant_ethnicity-3 | VARCHAR | 5 | 0.9992 | none found |
| applicant_ethnicity-4 | VARCHAR | 2 | 1.0000 | none found |
| applicant_ethnicity-5 | VARCHAR | 1 | 1.0000 | none found |
| co-applicant_ethnicity-1 | VARCHAR | 9 | 0.0001 | none found |
| co-applicant_ethnicity-2 | VARCHAR | 6 | 0.9901 | none found |
| co-applicant_ethnicity-3 | VARCHAR | 4 | 0.9999 | none found |
| co-applicant_ethnicity-4 | VARCHAR | 0 | 1.0000 | none found |
| co-applicant_ethnicity-5 | VARCHAR | 0 | 1.0000 | none found |
| applicant_ethnicity_observed | VARCHAR | 3 | 0.0000 | none found |
| co-applicant_ethnicity_observed | VARCHAR | 4 | 0.0000 | none found |
| applicant_race-1 | VARCHAR | 18 | 0.0002 | none found |
| applicant_race-2 | VARCHAR | 16 | 0.9726 | none found |
| applicant_race-3 | VARCHAR | 15 | 0.9962 | none found |
| applicant_race-4 | VARCHAR | 12 | 0.9995 | none found |
| applicant_race-5 | VARCHAR | 6 | 0.9999 | none found |
| co-applicant_race-1 | VARCHAR | 19 | 0.0000 | none found |
| co-applicant_race-2 | VARCHAR | 16 | 0.9901 | none found |
| co-applicant_race-3 | VARCHAR | 12 | 0.9989 | none found |
| co-applicant_race-4 | VARCHAR | 6 | 0.9999 | none found |
| co-applicant_race-5 | VARCHAR | 4 | 0.9999 | none found |
| applicant_race_observed | VARCHAR | 3 | 0.0000 | none found |
| co-applicant_race_observed | VARCHAR | 4 | 0.0000 | none found |
| applicant_sex | VARCHAR | 5 | 0.0000 | none found |
| co-applicant_sex | VARCHAR | 6 | 0.0000 | none found |
| applicant_sex_observed | VARCHAR | 3 | 0.0000 | none found |
| co-applicant_sex_observed | VARCHAR | 4 | 0.0000 | none found |
| applicant_age | VARCHAR | 8 | 0.0000 | 8888 (17284) |
| co-applicant_age | VARCHAR | 9 | 0.0000 | none found |
| applicant_age_above_62 | VARCHAR | 3 | 0.0000 | NA (17284) |
| co-applicant_age_above_62 | VARCHAR | 3 | 0.0000 | NA (84658) |
| submission_of_application | VARCHAR | 4 | 0.0000 | none found |
| initially_payable_to_institution | VARCHAR | 4 | 0.0000 | none found |
| aus-1 | VARCHAR | 8 | 0.0000 | none found |
| aus-2 | VARCHAR | 6 | 0.9476 | none found |
| aus-3 | VARCHAR | 6 | 0.9735 | none found |
| aus-4 | VARCHAR | 4 | 0.9835 | none found |
| aus-5 | VARCHAR | 4 | 0.9856 | none found |
| denial_reason-1 | VARCHAR | 11 | 0.0000 | none found |
| denial_reason-2 | VARCHAR | 9 | 0.9671 | none found |
| denial_reason-3 | VARCHAR | 9 | 0.9946 | none found |
| denial_reason-4 | VARCHAR | 8 | 0.9993 | none found |
| tract_population | VARCHAR | 515 | 0.0000 | none found |
| tract_minority_population_percent | VARCHAR | 522 | 0.0000 | none found |
| ffiec_msa_md_median_family_income | VARCHAR | 14 | 0.0000 | none found |
| tract_to_msa_income_percentage | VARCHAR | 674 | 0.0000 | none found |
| tract_owner_occupied_units | VARCHAR | 472 | 0.0000 | none found |
| tract_one_to_four_family_homes | VARCHAR | 487 | 0.0000 | none found |
| tract_median_age_of_housing_units | VARCHAR | 74 | 0.0000 | none found |
