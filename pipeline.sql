

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

