

@transform_pandas(
    Output(rid="ri.vector.main.execute.7641dae2-3118-4a2c-8a89-e4f646cbf18f"),
    aggregate_person=Input(rid="ri.vector.main.execute.0c8d244b-83b0-4a73-9488-3db78097ac5a"),
    sql_pivot_vax_person=Input(rid="ri.vector.main.execute.c2687a32-aea0-4394-ae9d-b488d148563e")
)
SELECT distinct
    a.person_id, 
    a.data_partner_id,
    b.vaccine_txn,
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
    aggregate_person_testing=Input(rid="ri.vector.main.execute.47a29e5a-88f1-4a48-8615-bb98ab911fca"),
    sql_pivot_vax_person_testing=Input(rid="ri.vector.main.execute.b7372302-6638-40c4-ad0c-4f6ab67373da")
)
SELECT distinct
    a.person_id, 
    a.data_partner_id,
    b.vaccine_txn,
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
    Output(rid="ri.vector.main.execute.47a29e5a-88f1-4a48-8615-bb98ab911fca"),
    deduplicated_testing=Input(rid="ri.vector.main.execute.708dc926-ae90-4f99-bb13-f3957d642c78")
)
SELECT person_id, data_partner_id, count(vax_date) as vaccine_txn
FROM deduplicated_testing
group by person_id, data_partner_id

@transform_pandas(
    Output(rid="ri.vector.main.execute.20bfed79-2fea-446f-a920-88f0dbc22bc2"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
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
    
FROM concept_set_members cs
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
    Output(rid="ri.vector.main.execute.9367a283-bd9d-45c9-9415-67c37443fbcc"),
    manifest_safe_harbor_testing=Input(rid="ri.foundry.main.dataset.7a5c5585-1c69-4bf5-9757-3fd0d0a209a2"),
    procedure_occurrence_testing=Input(rid="ri.foundry.main.dataset.88523aaa-75c3-4b55-a79a-ebe27e40ba4f")
)
SELECT DISTINCT person_id,
    po.data_partner_id,
    procedure_date,

    case when procedure_concept_id = 766238 then 'pfizer'
        when procedure_concept_id = 766239 then 'moderna'
        when procedure_concept_id = 766241 then 'janssen'
        else null
        end as vax_type

FROM procedure_occurrence_testing po
INNER JOIN manifest_safe_harbor_testing m ON m.data_partner_id = po.data_partner_id
where procedure_concept_id IN (766238, 766239, 766241) and po.data_partner_id = 406
    -- Very conservative filters to allow for date shifting
    and procedure_date > '2018-12-20'
    and procedure_date < (m.run_date + 356*2)
    and procedure_date is not null

@transform_pandas(
    Output(rid="ri.vector.main.execute.73881ed9-9110-4370-85ab-4c40e879b3ba"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    drug_exposure_testing=Input(rid="ri.foundry.main.dataset.26a51cab-0279-45a6-bbc0-f44a12b52f9c"),
    manifest_safe_harbor_testing=Input(rid="ri.foundry.main.dataset.7a5c5585-1c69-4bf5-9757-3fd0d0a209a2")
)
/* Import the drug_exposure_testing, concept_set_members, and manifest_safe_harbor_testing tables */

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
    
FROM concept_set_members cs
INNER JOIN drug_exposure_testing de on cs.concept_id = de.drug_concept_id
INNER JOIN manifest_safe_harbor_testing m on de.data_partner_id = m.data_partner_id

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
    Output(rid="ri.vector.main.execute.de1a2a39-7020-47f6-bb12-0a2e2ccec6a1"),
    baseline_vaccines_from_proc_testing=Input(rid="ri.vector.main.execute.9367a283-bd9d-45c9-9415-67c37443fbcc"),
    baseline_vaccines_testing=Input(rid="ri.vector.main.execute.73881ed9-9110-4370-85ab-4c40e879b3ba")
)
--create minimal long dataset of vaccine transactions to transpose 

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
    Output(rid="ri.vector.main.execute.b7372302-6638-40c4-ad0c-4f6ab67373da"),
    deduplicated_testing=Input(rid="ri.vector.main.execute.708dc926-ae90-4f99-bb13-f3957d642c78")
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4e2bf601-5e1d-4116-9dee-f3baefd298c9"),
    train_test_model=Input(rid="ri.foundry.main.dataset.ea6c836a-9d51-4402-b1b7-0e30fb514fc8")
)
SELECT *
FROM train_test_model

