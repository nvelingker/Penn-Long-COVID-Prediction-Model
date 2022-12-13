

@transform_pandas(
    Output(rid="ri.vector.main.execute.7641dae2-3118-4a2c-8a89-e4f646cbf18f"),
    aggregate_person=Input(rid="ri.vector.main.execute.0c8d244b-83b0-4a73-9488-3db78097ac5a"),
    sql_pivot_vax_person=Input(rid="ri.vector.main.execute.c2687a32-aea0-4394-ae9d-b488d148563e")
)
SELECT distinct
    a.person_id, 
    a.data_partner_id,
    b.vaccine_txn,
    datediff(1_vax_date, '2020-01-01') date_1_vax,
    datediff(2_vax_date, 1_vax_date) date_diff_1_2,
    datediff(3_vax_date, 2_vax_date) date_diff_2_3, 
    datediff(4_vax_date, 3_vax_date) date_diff_3_4, 
    case when 1_vax_type != 2_vax_type and 1_vax_date != 2_vax_date then 1 else 0 end as switch_1_2,
    case when 2_vax_type != 3_vax_type and 2_vax_date != 3_vax_date then 1 else 0 end as switch_2_3,
    case when 3_vax_type != 4_vax_type and 3_vax_date != 4_vax_date then 1 else 0 end as switch_3_4,
    1_vax_date,
    1_vax_type,
    2_vax_date,
    2_vax_type,
    3_vax_date,
    3_vax_type,
    4_vax_date,
    4_vax_type
FROM sql_pivot_vax_person a LEFT JOIN aggregate_person b on a.person_id = b.person_id
WHERE vaccine_txn < 5

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9392c81b-bbbf-4e66-a366-a2e7e4f9db7b"),
    aggregate_person_testing=Input(rid="ri.foundry.main.dataset.60eebc27-2e46-4a09-a76b-bd61122a81fd"),
    sql_pivot_vax_person_testing=Input(rid="ri.foundry.main.dataset.9b008b1f-ae3c-4b82-b2fb-f8d9c7a122e9")
)
SELECT distinct
    a.person_id, 
    a.data_partner_id,
    b.vaccine_txn,
    datediff(1_vax_date, '2020-01-01') date_1_vax,
    datediff(2_vax_date, 1_vax_date) date_diff_1_2,
    datediff(3_vax_date, 2_vax_date) date_diff_2_3, 
    datediff(4_vax_date, 3_vax_date) date_diff_3_4, 
    case when 1_vax_type != 2_vax_type and 1_vax_date != 2_vax_date then 1 else 0 end as switch_1_2,
    case when 2_vax_type != 3_vax_type and 2_vax_date != 3_vax_date then 1 else 0 end as switch_2_3,
    case when 3_vax_type != 4_vax_type and 3_vax_date != 4_vax_date then 1 else 0 end as switch_3_4,
    1_vax_date,
    1_vax_type,
    2_vax_date,
    2_vax_type,
    3_vax_date,
    3_vax_type,
    4_vax_date,
    4_vax_type
FROM sql_pivot_vax_person_testing a LEFT JOIN aggregate_person_testing b on a.person_id = b.person_id
WHERE vaccine_txn < 5

@transform_pandas(
    Output(rid="ri.vector.main.execute.0c8d244b-83b0-4a73-9488-3db78097ac5a"),
    deduplicated=Input(rid="ri.vector.main.execute.ec478f23-d29c-4d13-924b-e3b462b7a054")
)
SELECT person_id, data_partner_id, count(vax_date) as vaccine_txn
FROM deduplicated
group by person_id, data_partner_id

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.60eebc27-2e46-4a09-a76b-bd61122a81fd"),
    deduplicated_testing=Input(rid="ri.foundry.main.dataset.407bb4de-2a25-4520-8e03-f1e07031a43f")
)
SELECT person_id, data_partner_id, count(vax_date) as vaccine_txn
FROM deduplicated_testing
group by person_id, data_partner_id

@transform_pandas(
    Output(rid="ri.vector.main.execute.20bfed79-2fea-446f-a920-88f0dbc22bc2"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5"),
    manifest_safe_harbor=Input(rid="ri.foundry.main.dataset.b4407989-1851-4e07-a13f-0539fae10f26")
)
/* Import the drug_exposure, concept_set_members, and manifest_safe_harbor tables */

SELECT distinct de.person_id,
    de.data_partner_id,
    de.drug_exposure_start_date,
    de.drug_source_concept_name,
    de.drug_concept_id,

    case when concept_id in (702677, 702678, 724907, 37003432, 37003435, 37003436) then 'pfizer'
        when concept_id in (724906, 37003518) then 'moderna'
        when concept_id in (702866, 739906) then 'janssen'
        when concept_id in (724905) then 'astrazeneca'
        else null
        end as vax_type
    
FROM custom_concept_set_members cs
INNER JOIN drug_exposure de on cs.concept_id = de.drug_concept_id
INNER JOIN manifest_safe_harbor m on de.data_partner_id = m.data_partner_id

where codeset_id = 600531961
    and drug_exposure_start_date is not null
    -- Very conservative filters to allow for date shifting
    and drug_exposure_start_date > '2018-12-20'
    and drug_exposure_start_date < (m.run_date + 356*2)
    and (
        (drug_type_concept_id not in (38000177,32833,38000175,32838,32839)) 
        or (drug_type_concept_id = 38000177 AND m.cdm_name = 'ACT')
    )

@transform_pandas(
    Output(rid="ri.vector.main.execute.41c51a4c-2331-41c3-acad-6b3c1be58064"),
    manifest_safe_harbor=Input(rid="ri.foundry.main.dataset.b4407989-1851-4e07-a13f-0539fae10f26"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3")
)
SELECT DISTINCT person_id,
    po.data_partner_id,
    procedure_date,

    case when procedure_concept_id = 766238 then 'pfizer'
        when procedure_concept_id = 766239 then 'moderna'
        when procedure_concept_id = 766241 then 'janssen'
        else null
        end as vax_type

FROM procedure_occurrence po
INNER JOIN manifest_safe_harbor m ON m.data_partner_id = po.data_partner_id
where procedure_concept_id IN (766238, 766239, 766241) and po.data_partner_id = 406
    -- Very conservative filters to allow for date shifting
    and procedure_date > '2018-12-20'
    and procedure_date < (m.run_date + 356*2)
    and procedure_date is not null

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.74b5dd29-49ed-48ff-b6bf-da1c13614821"),
    manifest_safe_harbor_testing_copy=Input(rid="ri.foundry.main.dataset.f756c161-a369-4a22-9591-03ace0f5d1a5"),
    procedure_occurrence_testing_copy=Input(rid="ri.foundry.main.dataset.2d76588c-fe75-4d07-8044-f054444ec728")
)
SELECT DISTINCT person_id,
    po.data_partner_id,
    procedure_date,

    case when procedure_concept_id = 766238 then 'pfizer'
        when procedure_concept_id = 766239 then 'moderna'
        when procedure_concept_id = 766241 then 'janssen'
        else null
        end as vax_type

FROM procedure_occurrence_testing_copy po
INNER JOIN manifest_safe_harbor_testing_copy  m ON m.data_partner_id = po.data_partner_id
where procedure_concept_id IN (766238, 766239, 766241) and po.data_partner_id = 406
    -- Very conservative filters to allow for date shifting
    and procedure_date > '2018-12-20'
    and procedure_date < (m.run_date + 356*2)
    and procedure_date is not null

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.6f224cbc-ab43-44f9-bf66-f1716c7b1aa7"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    drug_exposure_testing_copy=Input(rid="ri.foundry.main.dataset.6223d2b6-e8b8-4d48-8c4c-81dd2959d131"),
    manifest_safe_harbor_testing_copy=Input(rid="ri.foundry.main.dataset.f756c161-a369-4a22-9591-03ace0f5d1a5")
)
/* Import the drug_exposure_testing_copy, concept_set_members, and manifest_safe_harbor_testing tables */

SELECT distinct de.person_id,
    de.data_partner_id,
    de.drug_exposure_start_date,
    de.drug_source_concept_name,
    de.drug_concept_id,

    case when concept_id in (702677, 702678, 724907, 37003432, 37003435, 37003436) then 'pfizer'
        when concept_id in (724906, 37003518) then 'moderna'
        when concept_id in (702866, 739906) then 'janssen'
        when concept_id in (724905) then 'astrazeneca'
        else null
        end as vax_type
    
FROM custom_concept_set_members cs
INNER JOIN drug_exposure_testing_copy de on cs.concept_id = de.drug_concept_id
INNER JOIN manifest_safe_harbor_testing_copy m on de.data_partner_id = m.data_partner_id

where codeset_id = 600531961
    and drug_exposure_start_date is not null
    -- Very conservative filters to allow for date shifting
    and drug_exposure_start_date > '2018-12-20'
    and drug_exposure_start_date < (m.run_date + 356*2)
    and (
        (drug_type_concept_id not in (38000177,32833,38000175,32838,32839)) 
        or (drug_type_concept_id = 38000177 AND m.cdm_name = 'ACT')
    )

@transform_pandas(
    Output(rid="ri.vector.main.execute.8a3be0e3-a478-40ab-83b1-7289e3fc5136"),
    baseline_vaccines=Input(rid="ri.vector.main.execute.20bfed79-2fea-446f-a920-88f0dbc22bc2"),
    baseline_vaccines_from_proc=Input(rid="ri.vector.main.execute.41c51a4c-2331-41c3-acad-6b3c1be58064")
)
--create minimal long dataset of vaccine transactions to transpose 

SELECT DISTINCT person_id, 
    data_partner_id,
    drug_exposure_start_date as vax_date,
    vax_type
FROM baseline_vaccines

UNION 

SELECT DISTINCT person_id,
    data_partner_id,
    procedure_date as vax_date,
    vax_type
FROM baseline_vaccines_from_proc

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.783be2cb-5a74-4652-baf6-b0b7b5b6d046"),
    baseline_vaccines_from_proc_testing=Input(rid="ri.foundry.main.dataset.74b5dd29-49ed-48ff-b6bf-da1c13614821"),
    baseline_vaccines_testing=Input(rid="ri.foundry.main.dataset.6f224cbc-ab43-44f9-bf66-f1716c7b1aa7")
)
SELECT DISTINCT person_id, 
    data_partner_id,
    drug_exposure_start_date as vax_date,
    vax_type
FROM baseline_vaccines_testing

UNION 

SELECT DISTINCT person_id,
    data_partner_id,
    procedure_date as vax_date,
    vax_type
FROM baseline_vaccines_from_proc_testing

@transform_pandas(
    Output(rid="ri.vector.main.execute.c2687a32-aea0-4394-ae9d-b488d148563e"),
    deduplicated=Input(rid="ri.vector.main.execute.ec478f23-d29c-4d13-924b-e3b462b7a054")
)

select * from (
    select person_id, data_partner_id, 
    row_number() over (partition by person_id order by person_id, vax_date) as number, vax_type, vax_date 
    from deduplicated 
) A 

pivot (
    max(vax_date) as vax_date, max(vax_type) as vax_type 
    for number in (1,2,3,4) 
)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9b008b1f-ae3c-4b82-b2fb-f8d9c7a122e9"),
    deduplicated_testing=Input(rid="ri.foundry.main.dataset.407bb4de-2a25-4520-8e03-f1e07031a43f")
)

select * from (
    select person_id, data_partner_id, 
    row_number() over (partition by person_id order by person_id, vax_date) as number, vax_type, vax_date 
    from deduplicated_testing
) A 

pivot (
    max(vax_date) as vax_date, max(vax_type) as vax_type 
    for number in (1,2,3,4) 
)

