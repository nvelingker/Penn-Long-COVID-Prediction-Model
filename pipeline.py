from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

import pandas as pd
from sklearn.linear_model import LogisticRegression

@transform_pandas(
    Output(rid="ri.vector.main.execute.fab1d6ae-e7fb-434e-bf54-ddc10591ac6d"),
    LL_DO_NOT_DELETE_REQUIRED_concept_sets_all=Input(rid="ri.foundry.main.dataset.029aa987-cfef-48fc-bf45-cffd3792cd93"),
    LL_concept_sets_fusion_everyone=Input(rid="ri.foundry.main.dataset.b36c87be-4e43-4f55-a1b2-fc48b0576a77")
)
#The purpose of this node is to optimize the user's experience connecting a customized concept set "fusion sheet" input data frame to replace LL_concept_sets_fusion_everyone.

def customized_concept_set_input(LL_concept_sets_fusion_everyone, LL_DO_NOT_DELETE_REQUIRED_concept_sets_all):
    PM_LL_DO_NOT_DELETE_REQUIRED_concept_sets_all = LL_DO_NOT_DELETE_REQUIRED_concept_sets_all

    required = LL_DO_NOT_DELETE_REQUIRED_concept_sets_all
    customizable = LL_concept_sets_fusion_everyone
    
    df = required.join(customizable, on = required.columns, how = 'outer')
    
    return df

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.vector.main.execute.4156447b-8bad-4af5-9150-010e304fa65a"),
    LL_DO_NOT_DELETE_REQUIRED_concept_sets_all=Input(rid="ri.foundry.main.dataset.029aa987-cfef-48fc-bf45-cffd3792cd93"),
    LL_concept_sets_fusion_everyone=Input(rid="ri.foundry.main.dataset.b36c87be-4e43-4f55-a1b2-fc48b0576a77")
)
#The purpose of this node is to optimize the user's experience connecting a customized concept set "fusion sheet" input data frame to replace LL_concept_sets_fusion_everyone.

def customized_concept_set_input_testing(LL_concept_sets_fusion_everyone, LL_DO_NOT_DELETE_REQUIRED_concept_sets_all):
    PM_LL_DO_NOT_DELETE_REQUIRED_concept_sets_all = LL_DO_NOT_DELETE_REQUIRED_concept_sets_all

    required = LL_DO_NOT_DELETE_REQUIRED_concept_sets_all
    customizable = LL_concept_sets_fusion_everyone
    
    df = required.join(customizable, on = required.columns, how = 'outer')
    
    return df

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.vector.main.execute.ec478f23-d29c-4d13-924b-e3b462b7a054"),
    distinct_vax_person=Input(rid="ri.vector.main.execute.8a3be0e3-a478-40ab-83b1-7289e3fc5136")
)
# Records for shots are often duplicated on different days, especially
# in the procedures table for site 406. For that site, if shots are
# less than 21 days apart, use the earlier date and drop the latter.

# 2711123981570257420

from pyspark.sql.window import Window
import pyspark.sql.functions as f

def deduplicated(distinct_vax_person):

    ################################################################################
    # 1. Resolve same day vaccinations with conflicting types. If one is null, use #
    #    the other. With multiple valid types, make null.                          #
    ################################################################################

    # Filter down to unique combinations of person, day, and vaccine type then drop
    # null values.
    vax_types = distinct_vax_person.dropDuplicates(
        ['person_id', 'vax_date', 'vax_type']
    ).filter(
        "vax_type is not NULL"
    )

    # Count number of types per person and day
    w = Window.partitionBy('person_id', 'vax_date')
    count_type = vax_types.select(
        'person_id', 
        'vax_date',
        'vax_type',
        f.count('person_id').over(w).alias('n')
    )

    # Drop rows with multiple values so they end up null after future join
    vax_types = count_type.filter(
        count_type.n == 1
    ).drop('n')

    # Drop original vax_type and merge this new one back into dataframe
    df = distinct_vax_person.drop(
        'vax_type'
    ).join(vax_types, on=['person_id', 'vax_date'], how='left')

    ################################################################################
    # 2. Deduplicate vaccines that are too close to be reasonable. Site 406 has    #
    #    extra issues due to using procedures table, so be more aggressive there.  #
    ################################################################################

    # Window by person_id
    w = Window.partitionBy('person_id').orderBy('vax_date')
    
    # Get difference between each shot in days
    df = df.withColumn(
        'lag_date', f.lag('vax_date', default='2000-01-01').over(w)
    ).withColumn(
        'date_diff', f.datediff('vax_date', 'lag_date')
    )

    # For site 406, filter if less than 14. For everyone else, filter if less than 5
    df = df.filter(
        (
            (df.data_partner_id == 406) & (df.date_diff >= 14)
        ) | (
            (df.data_partner_id != 406) & (df.date_diff >= 5)
        )
    )

    return df

@transform_pandas(
    Output(rid="ri.vector.main.execute.708dc926-ae90-4f99-bb13-f3957d642c78"),
    distinct_vax_person_testing=Input(rid="ri.vector.main.execute.de1a2a39-7020-47f6-bb12-0a2e2ccec6a1")
)
# Records for shots are often duplicated on different days, especially
# in the procedures table for site 406. For that site, if shots are
# less than 21 days apart, use the earlier date and drop the latter.

# 2711123981570257420

from pyspark.sql.window import Window
import pyspark.sql.functions as f

def deduplicated_testing(distinct_vax_person_testing):
    distinct_vax_person = distinct_vax_person_testing

    ################################################################################
    # 1. Resolve same day vaccinations with conflicting types. If one is null, use #
    #    the other. With multiple valid types, make null.                          #
    ################################################################################

    # Filter down to unique combinations of person, day, and vaccine type then drop
    # null values.
    vax_types = distinct_vax_person.dropDuplicates(
        ['person_id', 'vax_date', 'vax_type']
    ).filter(
        "vax_type is not NULL"
    )

    # Count number of types per person and day
    w = Window.partitionBy('person_id', 'vax_date')
    count_type = vax_types.select(
        'person_id', 
        'vax_date',
        'vax_type',
        f.count('person_id').over(w).alias('n')
    )

    # Drop rows with multiple values so they end up null after future join
    vax_types = count_type.filter(
        count_type.n == 1
    ).drop('n')

    # Drop original vax_type and merge this new one back into dataframe
    df = distinct_vax_person.drop(
        'vax_type'
    ).join(vax_types, on=['person_id', 'vax_date'], how='left')

    ################################################################################
    # 2. Deduplicate vaccines that are too close to be reasonable. Site 406 has    #
    #    extra issues due to using procedures table, so be more aggressive there.  #
    ################################################################################

    # Window by person_id
    w = Window.partitionBy('person_id').orderBy('vax_date')
    
    # Get difference between each shot in days
    df = df.withColumn(
        'lag_date', f.lag('vax_date', default='2000-01-01').over(w)
    ).withColumn(
        'date_diff', f.datediff('vax_date', 'lag_date')
    )

    # For site 406, filter if less than 14. For everyone else, filter if less than 5
    df = df.filter(
        (
            (df.data_partner_id == 406) & (df.date_diff >= 14)
        ) | (
            (df.data_partner_id != 406) & (df.date_diff >= 5)
        )
    )

    return df

@transform_pandas(
    Output(rid="ri.vector.main.execute.8dc65c1f-39e5-4bb7-b5c0-161a2f87aa0e"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    location=Input(rid="ri.foundry.main.dataset.4805affe-3a77-4260-8da5-4f9ff77f51ab"),
    manifest_safe_harbor=Input(rid="ri.foundry.main.dataset.b4407989-1851-4e07-a13f-0539fae10f26"),
    microvisits_to_macrovisits=Input(rid="ri.foundry.main.dataset.d77a701f-34df-48a1-a71c-b28112a07ffa"),
    person=Input(rid="ri.foundry.main.dataset.f71ffe18-6969-4a24-b81c-0e06a1ae9316")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave. More information can be found in the README linked here (https://unite.nih.gov/workspace/report/ri.report.main.report.855e1f58-bf44-4343-9721-8b4c878154fe).
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node gathers some commonly used facts about these patients from the "person" and "location" tables, as well as some facts about the patient's institution (from the "manifest" table).  Available age, race, and locations data (including SDOH variables for L3 only) is gathered at this node.  The patient’s total number of visits as well as the number of days in their observation period is calculated from the “microvisits_to_macrovisits” table in this node.  These facts will eventually be joined with the final patient-level table in the final node.

def everyone_cohort_de_id(concept_set_members, person, location, manifest_safe_harbor, microvisits_to_macrovisits):
        
    """
    Select proportion of enclave patients to use: A value of 1.0 indicates the pipeline will use all patients in the persons table.  
    A value less than 1.0 takes a random sample of the patients with a value of 0.001 (for example) representing a 0.1% sample of the persons table will be used.
    """
    proportion_of_patients_to_use = 1.0

    concepts_df = concept_set_members
    
    person_sample = person \
        .select('person_id','year_of_birth','month_of_birth','day_of_birth','ethnicity_concept_name','race_concept_name','gender_concept_name','location_id','data_partner_id') \
        .distinct() \
        .sample(False, proportion_of_patients_to_use, 111)

    visits_df = microvisits_to_macrovisits.select("person_id", "macrovisit_start_date", "visit_start_date")

    manifest_df = manifest_safe_harbor \
        .select('data_partner_id','run_date','cdm_name','cdm_version','shift_date_yn','max_num_shift_days') \
        .withColumnRenamed("run_date", "data_extraction_date")

    location_df = location \
        .dropDuplicates(subset=['location_id']) \
        .select('location_id','city','state','zip','county') \
        .withColumnRenamed('zip','postal_code')   
    
    #join in location_df data to person_sample dataframe 
    df = person_sample.join(location_df, 'location_id', 'left')

    #join in manifest_df information
    df = df.join(manifest_df, 'data_partner_id','inner')
    df = df.withColumn('max_num_shift_days', F.when(F.col('max_num_shift_days')=="", F.lit('0')).otherwise(F.regexp_replace(F.lower('max_num_shift_days'), 'na', '0')))
    
    #calculate date of birth for all patients
    df = df.withColumn("new_year_of_birth", F.when(F.col('year_of_birth').isNull(),1)
                                                .otherwise(F.col('year_of_birth')))
    df = df.withColumn("new_month_of_birth", F.when(F.col('month_of_birth').isNull(), 7)
                                                .when(F.col('month_of_birth')==0, 7)
                                                .otherwise(F.col('month_of_birth')))
    df = df.withColumn("new_day_of_birth", F.when(F.col('day_of_birth').isNull(), 1)
                                                .when(F.col('day_of_birth')==0, 1)
                                                .otherwise(F.col('day_of_birth')))

    df = df.withColumn("date_of_birth", F.concat_ws("-", F.col("new_year_of_birth"), F.col("new_month_of_birth"), F.col("new_day_of_birth")))
    df = df.withColumn("date_of_birth", F.to_date("date_of_birth", format=None)) 

    #convert date of birth string to date and apply min and max reasonable birthdate filter parameters, inclusive
    max_shift_as_int = df.withColumn("shift_days_as_int", F.col('max_num_shift_days').cast(IntegerType())) \
        .select(F.max('shift_days_as_int')) \
        .head()[0]

    min_reasonable_dob = "1902-01-01"
    max_reasonable_dob = F.date_add(F.current_date(), max_shift_as_int)

    df = df.withColumn("date_of_birth", F.when(F.col('date_of_birth').between(min_reasonable_dob, max_reasonable_dob), F.col('date_of_birth')).otherwise(None))

    df = df.withColumn("age", F.floor(F.months_between(max_reasonable_dob, "date_of_birth", roundOff=False)/12))

    H = ['Hispanic']
    A = ['Asian', 'Asian Indian', 'Bangladeshi', 'Bhutanese', 'Burmese', 'Cambodian', 'Chinese', 'Filipino', 'Hmong', 'Indonesian', 'Japanese', 'Korean', 'Laotian', 'Malaysian', 'Maldivian', 'Nepalese', 'Okinawan', 'Pakistani', 'Singaporean', 'Sri Lankan', 'Taiwanese', 'Thai', 'Vietnamese']
    B_AA = ['African', 'African American', 'Barbadian', 'Black', 'Black or African American', 'Dominica Islander', 'Haitian', 'Jamaican', 'Madagascar', 'Trinidadian', 'West Indian']
    W = ['White']
    NH_PI = ['Melanesian', 'Micronesian', 'Native Hawaiian or Other Pacific Islander', 'Other Pacific Islander', 'Polynesian']
    AI_AN = ['American Indian or Alaska Native']
    O = ['More than one race', 'Multiple race', 'Multiple races', 'Other', 'Other Race']
    U = ['Asian or Pacific Islander', 'No Information', 'No matching concept', 'Refuse to Answer', 'Unknown', 'Unknown racial group']

    df = df.withColumn("race_ethnicity", F.when(F.col("ethnicity_concept_name") == 'Hispanic or Latino', "Hispanic or Latino Any Race")
                        .when(F.col("race_concept_name").isin(H), "Hispanic or Latino Any Race")
                        .when(F.col("race_concept_name").isin(A), "Asian Non-Hispanic")
                        .when(F.col("race_concept_name").isin(B_AA), "Black or African American Non-Hispanic")
                        .when(F.col("race_concept_name").isin(W), "White Non-Hispanic")
                        .when(F.col("race_concept_name").isin(NH_PI), "Native Hawaiian or Other Pacific Islander Non-Hispanic") 
                        .when(F.col("race_concept_name").isin(AI_AN), "American Indian or Alaska Native Non-Hispanic")
                        .when(F.col("race_concept_name").isin(O), "Other Non-Hispanic")
                        .when(F.col("race_concept_name").isin(U), "Unknown")
                        .otherwise("Unknown"))

    #create visit counts/obs period dataframes
    hosp_visits = visits_df.where(F.col("macrovisit_start_date").isNotNull()) \
        .orderBy("visit_start_date") \
        .coalesce(1) \
        .dropDuplicates(["person_id", "macrovisit_start_date"]) #hospital

    non_hosp_visits = visits_df.where(F.col("macrovisit_start_date").isNull()) \
        .dropDuplicates(["person_id", "visit_start_date"]) #non-hospital
        
    visits_df = hosp_visits.union(non_hosp_visits) #join the two

    #total number of visits
    visits_count = visits_df.groupBy("person_id") \
        .count() \
        .select("person_id", F.col('count').alias('total_visits'))

    #obs period in days 
    observation_period = visits_df.groupby('person_id').agg(
        F.max('visit_start_date').alias('pt_max_visit_date'),
        F.min('visit_start_date').alias('pt_min_visit_date')) \
        .withColumn('observation_period', F.datediff('pt_max_visit_date', 'pt_min_visit_date')) \
        .select('person_id', 'observation_period')
    
    #join visit counts/obs periods dataframes with main dataframe
    df = df.join(visits_count, "person_id", "left")
    df = df.join(observation_period, "person_id", "left")

    #LEVEL 2 ONLY
    df = df.withColumn('max_num_shift_days', F.concat(F.col('max_num_shift_days'), F.lit(" + 180"))).withColumn('shift_date_yn', F.lit('Y'))

    df = df.select('person_id',
        'total_visits',
        'observation_period',
        'gender_concept_name',
        'city',
        'state',
        'postal_code',
        'county',
        'age',
        'race_ethnicity',
        'data_partner_id',
        'data_extraction_date',
        'cdm_name',
        'cdm_version',
        'shift_date_yn',
        'max_num_shift_days')

    return df
    

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.vector.main.execute.0ca5c190-d2ae-492a-b291-66fd8092b269"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    location_testing=Input(rid="ri.foundry.main.dataset.06b728e0-0262-4a7a-b9b7-fe91c3f7da34"),
    manifest_safe_harbor_testing=Input(rid="ri.foundry.main.dataset.7a5c5585-1c69-4bf5-9757-3fd0d0a209a2"),
    microvisits_to_macrovisits_testing=Input(rid="ri.foundry.main.dataset.f5008fa4-e736-4244-88e1-1da7a68efcdb"),
    person_testing=Input(rid="ri.foundry.main.dataset.06629068-25fc-4802-9b31-ead4ed515da4")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave. More information can be found in the README linked here (https://unite.nih.gov/workspace/report/ri.report.main.report.855e1f58-bf44-4343-9721-8b4c878154fe).
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node gathers some commonly used facts about these patients from the "person" and "location" tables, as well as some facts about the patient's institution (from the "manifest" table).  Available age, race, and locations data (including SDOH variables for L3 only) is gathered at this node.  The patient’s total number of visits as well as the number of days in their observation period is calculated from the “microvisits_to_macrovisits” table in this node.  These facts will eventually be joined with the final patient-level table in the final node.

def everyone_cohort_de_id_testing(concept_set_members, person_testing, location_testing, manifest_safe_harbor_testing, microvisits_to_macrovisits_testing):
        
    """
    Select proportion of enclave patients to use: A value of 1.0 indicates the pipeline will use all patients in the persons table.  
    A value less than 1.0 takes a random sample of the patients with a value of 0.001 (for example) representing a 0.1% sample of the persons table will be used.
    """
    proportion_of_patients_to_use = 1.0

    concepts_df = concept_set_members
    
    person_sample = person_testing \
        .select('person_id','year_of_birth','month_of_birth','day_of_birth','ethnicity_concept_name','race_concept_name','gender_concept_name','location_id','data_partner_id') \
        .distinct() \
        .sample(False, proportion_of_patients_to_use, 111)

    visits_df = microvisits_to_macrovisits_testing.select("person_id", "macrovisit_start_date", "visit_start_date")

    manifest_df = manifest_safe_harbor_testing \
        .select('data_partner_id','run_date','cdm_name','cdm_version','shift_date_yn','max_num_shift_days') \
        .withColumnRenamed("run_date", "data_extraction_date")

    location_df = location_testing \
        .dropDuplicates(subset=['location_id']) \
        .select('location_id','city','state','zip','county') \
        .withColumnRenamed('zip','postal_code')   
    
    #join in location_df data to person_sample dataframe 
    df = person_sample.join(location_df, 'location_id', 'left')

    #join in manifest_df information
    df = df.join(manifest_df, 'data_partner_id','inner')
    df = df.withColumn('max_num_shift_days', F.when(F.col('max_num_shift_days')=="", F.lit('0')).otherwise(F.regexp_replace(F.lower('max_num_shift_days'), 'na', '0')))
    
    #calculate date of birth for all patients
    df = df.withColumn("new_year_of_birth", F.when(F.col('year_of_birth').isNull(),1)
                                                .otherwise(F.col('year_of_birth')))
    df = df.withColumn("new_month_of_birth", F.when(F.col('month_of_birth').isNull(), 7)
                                                .when(F.col('month_of_birth')==0, 7)
                                                .otherwise(F.col('month_of_birth')))
    df = df.withColumn("new_day_of_birth", F.when(F.col('day_of_birth').isNull(), 1)
                                                .when(F.col('day_of_birth')==0, 1)
                                                .otherwise(F.col('day_of_birth')))

    df = df.withColumn("date_of_birth", F.concat_ws("-", F.col("new_year_of_birth"), F.col("new_month_of_birth"), F.col("new_day_of_birth")))
    df = df.withColumn("date_of_birth", F.to_date("date_of_birth", format=None)) 

    #convert date of birth string to date and apply min and max reasonable birthdate filter parameters, inclusive
    max_shift_as_int = df.withColumn("shift_days_as_int", F.col('max_num_shift_days').cast(IntegerType())) \
        .select(F.max('shift_days_as_int')) \
        .head()[0]

    min_reasonable_dob = "1902-01-01"
    max_reasonable_dob = F.date_add(F.current_date(), max_shift_as_int)

    df = df.withColumn("date_of_birth", F.when(F.col('date_of_birth').between(min_reasonable_dob, max_reasonable_dob), F.col('date_of_birth')).otherwise(None))

    df = df.withColumn("age", F.floor(F.months_between(max_reasonable_dob, "date_of_birth", roundOff=False)/12))

    H = ['Hispanic']
    A = ['Asian', 'Asian Indian', 'Bangladeshi', 'Bhutanese', 'Burmese', 'Cambodian', 'Chinese', 'Filipino', 'Hmong', 'Indonesian', 'Japanese', 'Korean', 'Laotian', 'Malaysian', 'Maldivian', 'Nepalese', 'Okinawan', 'Pakistani', 'Singaporean', 'Sri Lankan', 'Taiwanese', 'Thai', 'Vietnamese']
    B_AA = ['African', 'African American', 'Barbadian', 'Black', 'Black or African American', 'Dominica Islander', 'Haitian', 'Jamaican', 'Madagascar', 'Trinidadian', 'West Indian']
    W = ['White']
    NH_PI = ['Melanesian', 'Micronesian', 'Native Hawaiian or Other Pacific Islander', 'Other Pacific Islander', 'Polynesian']
    AI_AN = ['American Indian or Alaska Native']
    O = ['More than one race', 'Multiple race', 'Multiple races', 'Other', 'Other Race']
    U = ['Asian or Pacific Islander', 'No Information', 'No matching concept', 'Refuse to Answer', 'Unknown', 'Unknown racial group']

    df = df.withColumn("race_ethnicity", F.when(F.col("ethnicity_concept_name") == 'Hispanic or Latino', "Hispanic or Latino Any Race")
                        .when(F.col("race_concept_name").isin(H), "Hispanic or Latino Any Race")
                        .when(F.col("race_concept_name").isin(A), "Asian Non-Hispanic")
                        .when(F.col("race_concept_name").isin(B_AA), "Black or African American Non-Hispanic")
                        .when(F.col("race_concept_name").isin(W), "White Non-Hispanic")
                        .when(F.col("race_concept_name").isin(NH_PI), "Native Hawaiian or Other Pacific Islander Non-Hispanic") 
                        .when(F.col("race_concept_name").isin(AI_AN), "American Indian or Alaska Native Non-Hispanic")
                        .when(F.col("race_concept_name").isin(O), "Other Non-Hispanic")
                        .when(F.col("race_concept_name").isin(U), "Unknown")
                        .otherwise("Unknown"))

    #create visit counts/obs period dataframes
    hosp_visits = visits_df.where(F.col("macrovisit_start_date").isNotNull()) \
        .orderBy("visit_start_date") \
        .coalesce(1) \
        .dropDuplicates(["person_id", "macrovisit_start_date"]) #hospital

    non_hosp_visits = visits_df.where(F.col("macrovisit_start_date").isNull()) \
        .dropDuplicates(["person_id", "visit_start_date"]) #non-hospital
        
    visits_df = hosp_visits.union(non_hosp_visits) #join the two

    #total number of visits
    visits_count = visits_df.groupBy("person_id") \
        .count() \
        .select("person_id", F.col('count').alias('total_visits'))

    #obs period in days 
    observation_period = visits_df.groupby('person_id').agg(
        F.max('visit_start_date').alias('pt_max_visit_date'),
        F.min('visit_start_date').alias('pt_min_visit_date')) \
        .withColumn('observation_period', F.datediff('pt_max_visit_date', 'pt_min_visit_date')) \
        .select('person_id', 'observation_period')
    
    #join visit counts/obs periods dataframes with main dataframe
    df = df.join(visits_count, "person_id", "left")
    df = df.join(observation_period, "person_id", "left")

    #LEVEL 2 ONLY
    df = df.withColumn('max_num_shift_days', F.concat(F.col('max_num_shift_days'), F.lit(" + 180"))).withColumn('shift_date_yn', F.lit('Y'))

    df = df.select('person_id',
        'total_visits',
        'observation_period',
        'gender_concept_name',
        'city',
        'state',
        'postal_code',
        'county',
        'age',
        'race_ethnicity',
        'data_partner_id',
        'data_extraction_date',
        'cdm_name',
        'cdm_version',
        'shift_date_yn',
        'max_num_shift_days')

    return df
    

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.vector.main.execute.fda1eca7-ed23-46f1-96f1-3c771194bd5b"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2"),
    customized_concept_set_input=Input(rid="ri.vector.main.execute.fab1d6ae-e7fb-434e-bf54-ddc10591ac6d"),
    everyone_cohort_de_id=Input(rid="ri.vector.main.execute.8dc65c1f-39e5-4bb7-b5c0-161a2f87aa0e")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node filters the condition_eras table for rows that have a condition_concept_id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these conditions are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_conditions_of_interest(everyone_cohort_de_id, concept_set_members, condition_occurrence, customized_concept_set_input):

    #bring in only cohort patient ids
    persons = everyone_cohort_de_id.select('person_id')
    #filter observations table to only cohort patients    
    conditions_df = condition_occurrence \
        .select('person_id', 'condition_start_date', 'condition_concept_id') \
        .where(F.col('condition_start_date').isNotNull()) \
        .withColumnRenamed('condition_start_date','visit_date') \
        .withColumnRenamed('condition_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the conditions domain
    fusion_df = customized_concept_set_input \
        .filter(customized_concept_set_input.domain.contains('condition')) \
        .select('concept_set_name','indicator_prefix')
    #filter concept set members table to only concept ids for the conditions of interest
    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where(F.col('is_most_recent_version')=='true') \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')

    #find conditions information based on matching concept ids for conditions of interest
    df = conditions_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for conditions of interest    
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)
   
    return df
    

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.vector.main.execute.53ac67c6-4b83-4354-9591-d4ed40cac3ae"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    condition_occurrence_testing=Input(rid="ri.foundry.main.dataset.3e01546f-f110-4c67-a6db-9063d2939a74"),
    customized_concept_set_input_testing=Input(rid="ri.vector.main.execute.4156447b-8bad-4af5-9150-010e304fa65a"),
    everyone_cohort_de_id_testing=Input(rid="ri.vector.main.execute.0ca5c190-d2ae-492a-b291-66fd8092b269")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node filters the condition_eras table for rows that have a condition_concept_id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these conditions are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_conditions_of_interest_testing(everyone_cohort_de_id_testing, concept_set_members, condition_occurrence_testing, customized_concept_set_input_testing):
    everyone_cohort_de_id = everyone_cohort_de_id_testing
    customized_concept_set_input = customized_concept_set_input_testing

    #bring in only cohort patient ids
    persons = everyone_cohort_de_id_testing.select('person_id')
    
    #filter observations table to only cohort patients    
    conditions_df = condition_occurrence_testing \
        .select('person_id', 'condition_start_date', 'condition_concept_id') \
        .where(F.col('condition_start_date').isNotNull()) \
        .withColumnRenamed('condition_start_date','visit_date') \
        .withColumnRenamed('condition_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the conditions domain
    fusion_df = customized_concept_set_input \
        .filter(customized_concept_set_input.domain.contains('condition')) \
        .select('concept_set_name','indicator_prefix')

    #filter concept set members table to only concept ids for the conditions of interest
    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where(F.col('is_most_recent_version')=='true') \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')

    #find conditions information based on matching concept ids for conditions of interest
    df = conditions_df.join(concepts_df, 'concept_id', 'right')

    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for conditions of interest    
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)
   
    return df
    

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.vector.main.execute.ae98106d-77ad-4cd5-9ee4-066d00ab0098"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    customized_concept_set_input=Input(rid="ri.vector.main.execute.fab1d6ae-e7fb-434e-bf54-ddc10591ac6d"),
    device_exposure=Input(rid="ri.foundry.main.dataset.c1fd6d67-fc80-4747-89ca-8eb04efcb874"),
    everyone_cohort_de_id=Input(rid="ri.vector.main.execute.8dc65c1f-39e5-4bb7-b5c0-161a2f87aa0e")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_devices_of_interest(device_exposure, everyone_cohort_de_id, concept_set_members, customized_concept_set_input):

    #bring in only cohort patient ids
    persons = everyone_cohort_de_id.select('person_id')
    #filter device exposure table to only cohort patients
    devices_df = device_exposure \
        .select('person_id','device_exposure_start_date','device_concept_id') \
        .where(F.col('device_exposure_start_date').isNotNull()) \
        .withColumnRenamed('device_exposure_start_date','visit_date') \
        .withColumnRenamed('device_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the devices domain
    fusion_df = customized_concept_set_input \
        .filter(customized_concept_set_input.domain.contains('device')) \
        .select('concept_set_name','indicator_prefix')
    #filter concept set members table to only concept ids for the devices of interest
    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where(F.col('is_most_recent_version')=='true') \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')
        
    #find device exposure information based on matching concept ids for devices of interest
    df = devices_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for devices of interest
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df
    

#################################################
## Global imports and functions included below ##
#################################################

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.905df3fe-1777-4ec7-93ae-e5ac46e50762"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    customized_concept_set_input_testing=Input(rid="ri.vector.main.execute.4156447b-8bad-4af5-9150-010e304fa65a"),
    device_exposure_testing=Input(rid="ri.foundry.main.dataset.7e24a101-2206-45d9-bcaa-b9d84bd2f990"),
    everyone_cohort_de_id_testing=Input(rid="ri.vector.main.execute.0ca5c190-d2ae-492a-b291-66fd8092b269")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_devices_of_interest_testing(device_exposure_testing, everyone_cohort_de_id_testing, concept_set_members, customized_concept_set_input_testing):
    everyone_cohort_de_id = everyone_cohort_de_id_testing
    customized_concept_set_input = customized_concept_set_input_testing

    #bring in only cohort patient ids
    persons = everyone_cohort_de_id_testing.select('person_id')
    #filter device exposure table to only cohort patients
    devices_df = device_exposure_testing \
        .select('person_id','device_exposure_start_date','device_concept_id') \
        .where(F.col('device_exposure_start_date').isNotNull()) \
        .withColumnRenamed('device_exposure_start_date','visit_date') \
        .withColumnRenamed('device_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the devices domain
    fusion_df = customized_concept_set_input \
        .filter(customized_concept_set_input.domain.contains('device')) \
        .select('concept_set_name','indicator_prefix')
    #filter concept set members table to only concept ids for the devices of interest
    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where(F.col('is_most_recent_version')=='true') \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')
        
    #find device exposure information based on matching concept ids for devices of interest
    df = devices_df.join(concepts_df, 'concept_id', 'right')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for devices of interest
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df
    

#################################################
## Global imports and functions included below ##
#################################################

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.78498d4e-7181-4e75-9d9d-7468b64c5ee0"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    customized_concept_set_input=Input(rid="ri.vector.main.execute.fab1d6ae-e7fb-434e-bf54-ddc10591ac6d"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5"),
    everyone_cohort_de_id=Input(rid="ri.vector.main.execute.8dc65c1f-39e5-4bb7-b5c0-161a2f87aa0e")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_drugs_of_interest(concept_set_members, drug_exposure, everyone_cohort_de_id, customized_concept_set_input):
  
    #bring in only cohort patient ids
    persons = everyone_cohort_de_id.select('person_id')
    #filter drug exposure table to only cohort patients    
    drug_df = drug_exposure \
        .select('person_id','drug_exposure_start_date','drug_concept_id') \
        .where(F.col('drug_exposure_start_date').isNotNull()) \
        .withColumnRenamed('drug_exposure_start_date','visit_date') \
        .withColumnRenamed('drug_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the drug domain
    fusion_df = customized_concept_set_input \
        .filter(customized_concept_set_input.domain.contains('drug')) \
        .select('concept_set_name','indicator_prefix')
    #filter concept set members table to only concept ids for the drugs of interest
    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where(F.col('is_most_recent_version')=='true') \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')
        
    #find drug exposure information based on matching concept ids for drugs of interest
    df = drug_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for drugs of interest
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df
    

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.vector.main.execute.49a46a48-21e3-4f9d-ae9e-67c7f3120fef"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    customized_concept_set_input_testing=Input(rid="ri.vector.main.execute.4156447b-8bad-4af5-9150-010e304fa65a"),
    drug_exposure_testing=Input(rid="ri.foundry.main.dataset.26a51cab-0279-45a6-bbc0-f44a12b52f9c"),
    everyone_cohort_de_id_testing=Input(rid="ri.vector.main.execute.0ca5c190-d2ae-492a-b291-66fd8092b269")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_drugs_of_interest_testing(concept_set_members, drug_exposure_testing, everyone_cohort_de_id_testing, customized_concept_set_input_testing):
    everyone_cohort_de_id = everyone_cohort_de_id_testing
    customized_concept_set_input = customized_concept_set_input_testing
  
    #bring in only cohort patient ids
    persons = everyone_cohort_de_id_testing.select('person_id')
    #filter drug exposure table to only cohort patients    
    drug_df = drug_exposure_testing \
        .select('person_id','drug_exposure_start_date','drug_concept_id') \
        .where(F.col('drug_exposure_start_date').isNotNull()) \
        .withColumnRenamed('drug_exposure_start_date','visit_date') \
        .withColumnRenamed('drug_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the drug domain
    fusion_df = customized_concept_set_input \
        .filter(customized_concept_set_input.domain.contains('drug')) \
        .select('concept_set_name','indicator_prefix')
    #filter concept set members table to only concept ids for the drugs of interest
    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where(F.col('is_most_recent_version')=='true') \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')
        
    #find drug exposure information based on matching concept ids for drugs of interest
    df = drug_df.join(concepts_df, 'concept_id', 'right')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for drugs of interest
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df
    

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.vector.main.execute.606758b7-753d-4fe7-a9ff-190c18388d2d"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    everyone_cohort_de_id=Input(rid="ri.vector.main.execute.8dc65c1f-39e5-4bb7-b5c0-161a2f87aa0e"),
    measurement=Input(rid="ri.foundry.main.dataset.5c8b84fb-814b-4ee5-a89a-9525f4a617c7")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node filters the measurements table for rows that have a measurement_concept_id associated with one of the concept sets described in the data dictionary in the README.  Indicator names for a positive COVID PCR or AG test, negative COVID PCR or AG test, positive COVID antibody test, and negative COVID antibody test are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date. It also finds the harmonized value as a number for BMI measurements and collapses these values to unique instances on the basis of patient and visit date.  Measurement BMI cutoffs included are intended for adults. Analyses focused on pediatric measurements should use different bounds for BMI measurements. 

def everyone_measurements_of_interest(measurement, concept_set_members, everyone_cohort_de_id):
    
    #bring in only cohort patient ids
    persons = everyone_cohort_de_id.select('person_id')
    #filter procedure occurrence table to only cohort patients    
    df = measurement \
        .select('person_id','measurement_date','measurement_concept_id','harmonized_value_as_number','value_as_concept_id') \
        .where(F.col('measurement_date').isNotNull()) \
        .withColumnRenamed('measurement_date','visit_date') \
        .join(persons,'person_id','inner')
        
    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where(F.col('is_most_recent_version')=='true')
          
    #Find BMI closest to COVID using both reported/observed BMI and calculated BMI using height and weight.  Cutoffs for reasonable height, weight, and BMI are provided and can be changed by the template user.
    lowest_acceptable_BMI = 10
    highest_acceptable_BMI = 100
    lowest_acceptable_weight = 5 #in kgs
    highest_acceptable_weight = 300 #in kgs
    lowest_acceptable_height = .6 #in meters
    highest_acceptable_height = 2.43 #in meters

    bmi_codeset_ids = list(concepts_df.where(
        (concepts_df.concept_set_name=="body mass index") 
        & (concepts_df.is_most_recent_version=='true')
        ).select('concept_id').toPandas()['concept_id'])
    weight_codeset_ids = list(concepts_df.where(
        (concepts_df.concept_set_name=="Body weight (LG34372-9 and SNOMED)") 
        & (concepts_df.is_most_recent_version=='true')
        ).select('concept_id').toPandas()['concept_id'])
    height_codeset_ids = list(concepts_df.where(
        (concepts_df.concept_set_name=="Height (LG34373-7 + SNOMED)") 
        & (concepts_df.is_most_recent_version=='true')
        ).select('concept_id').toPandas()['concept_id'])
    
    pcr_ag_test_ids = list(concepts_df.where(
        (concepts_df.concept_set_name=="ATLAS SARS-CoV-2 rt-PCR and AG") 
        & (concepts_df.is_most_recent_version=='true')
        ).select('concept_id').toPandas()['concept_id'])
    antibody_test_ids = list(concepts_df.where(
        (concepts_df.concept_set_name=="Atlas #818 [N3C] CovidAntibody retry") 
        & (concepts_df.is_most_recent_version=='true')
        ).select('concept_id').toPandas()['concept_id'])
    covid_positive_measurement_ids = list(concepts_df.where(
        (concepts_df.concept_set_name=="ResultPos") 
        & (concepts_df.is_most_recent_version=='true')
        ).select('concept_id').toPandas()['concept_id'])
    covid_negative_measurement_ids = list(concepts_df.where(
        (concepts_df.concept_set_name=="ResultNeg") 
        & (concepts_df.is_most_recent_version=='true')
        ).select('concept_id').toPandas()['concept_id'])
 
    #add value columns for rows associated with the above concept sets, but only include BMI or height or weight when in reasonable range
    BMI_df = df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('Recorded_BMI', F.when(df.measurement_concept_id.isin(bmi_codeset_ids) & df.harmonized_value_as_number.between(lowest_acceptable_BMI, highest_acceptable_BMI), df.harmonized_value_as_number).otherwise(0)) \
        .withColumn('height', F.when(df.measurement_concept_id.isin(height_codeset_ids) & df.harmonized_value_as_number.between(lowest_acceptable_height, highest_acceptable_height), df.harmonized_value_as_number).otherwise(0)) \
        .withColumn('weight', F.when(df.measurement_concept_id.isin(weight_codeset_ids) & df.harmonized_value_as_number.between(lowest_acceptable_weight, highest_acceptable_weight), df.harmonized_value_as_number).otherwise(0)) 
        
    labs_df = df.withColumn('PCR_AG_Pos', F.when(df.measurement_concept_id.isin(pcr_ag_test_ids) & df.value_as_concept_id.isin(covid_positive_measurement_ids), 1).otherwise(0)) \
        .withColumn('PCR_AG_Neg', F.when(df.measurement_concept_id.isin(pcr_ag_test_ids) & df.value_as_concept_id.isin(covid_negative_measurement_ids), 1).otherwise(0)) \
        .withColumn('Antibody_Pos', F.when(df.measurement_concept_id.isin(antibody_test_ids) & df.value_as_concept_id.isin(covid_positive_measurement_ids), 1).otherwise(0)) \
        .withColumn('Antibody_Neg', F.when(df.measurement_concept_id.isin(antibody_test_ids) & df.value_as_concept_id.isin(covid_negative_measurement_ids), 1).otherwise(0))
     
    #collapse all reasonable values to unique person and visit rows
    BMI_df = BMI_df.groupby('person_id', 'visit_date').agg(
    F.max('Recorded_BMI').alias('Recorded_BMI'),
    F.max('height').alias('height'),
    F.max('weight').alias('weight'))
    labs_df = labs_df.groupby('person_id', 'visit_date').agg(
    F.max('PCR_AG_Pos').alias('PCR_AG_Pos'),
    F.max('PCR_AG_Neg').alias('PCR_AG_Neg'),
    F.max('Antibody_Pos').alias('Antibody_Pos'),
    F.max('Antibody_Neg').alias('Antibody_Neg'))

    #add a calculated BMI for each visit date when height and weight available.  Note that if only one is available, it will result in zero
    #subsequent filter out rows that would have resulted from unreasonable calculated_BMI being used as best_BMI for the visit 
    BMI_df = BMI_df.withColumn('calculated_BMI', (BMI_df.weight/(BMI_df.height*BMI_df.height)))
    BMI_df = BMI_df.withColumn('BMI', F.when(BMI_df.Recorded_BMI>0, BMI_df.Recorded_BMI).otherwise(BMI_df.calculated_BMI)) \
        .select('person_id','visit_date','BMI')
    BMI_df = BMI_df.filter((BMI_df.BMI<=highest_acceptable_BMI) & (BMI_df.BMI>=lowest_acceptable_BMI)) \
        .withColumn('BMI_rounded', F.round(BMI_df.BMI)) \
        .drop('BMI')
    BMI_df = BMI_df.withColumn('OBESITY', F.when(BMI_df.BMI_rounded>=30, 1).otherwise(0))

    #join BMI_df with labs_df to retain all lab results with only reasonable BMI_rounded and OBESITY flags
    df = labs_df.join(BMI_df, on=['person_id', 'visit_date'], how='left') 

    return df

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.vector.main.execute.b6e9730e-6d5d-4ee7-80e9-34db9d4f89cf"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    everyone_cohort_de_id_testing=Input(rid="ri.vector.main.execute.0ca5c190-d2ae-492a-b291-66fd8092b269"),
    measurement_testing=Input(rid="ri.foundry.main.dataset.b7749e49-cf01-4d0a-a154-2f00eecab21e")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node filters the measurements table for rows that have a measurement_concept_id associated with one of the concept sets described in the data dictionary in the README.  Indicator names for a positive COVID PCR or AG test, negative COVID PCR or AG test, positive COVID antibody test, and negative COVID antibody test are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date. It also finds the harmonized value as a number for BMI measurements and collapses these values to unique instances on the basis of patient and visit date.  Measurement BMI cutoffs included are intended for adults. Analyses focused on pediatric measurements should use different bounds for BMI measurements. 

def everyone_measurements_of_interest_testing(measurement_testing, concept_set_members, everyone_cohort_de_id_testing):
    everyone_cohort_de_id = everyone_cohort_de_id_testing
    
    #bring in only cohort patient ids
    persons = everyone_cohort_de_id_testing.select('person_id')
    #filter procedure occurrence table to only cohort patients    
    df = measurement_testing \
        .select('person_id','measurement_date','measurement_concept_id','harmonized_value_as_number','value_as_concept_id') \
        .where(F.col('measurement_date').isNotNull()) \
        .withColumnRenamed('measurement_date','visit_date') \
        .join(persons,'person_id','inner')
        
    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where(F.col('is_most_recent_version')=='true')
          
    #Find BMI closest to COVID using both reported/observed BMI and calculated BMI using height and weight.  Cutoffs for reasonable height, weight, and BMI are provided and can be changed by the template user.
    lowest_acceptable_BMI = 10
    highest_acceptable_BMI = 100
    lowest_acceptable_weight = 5 #in kgs
    highest_acceptable_weight = 300 #in kgs
    lowest_acceptable_height = .6 #in meters
    highest_acceptable_height = 2.43 #in meters

    bmi_codeset_ids = list(concepts_df.where(
        (concepts_df.concept_set_name=="body mass index") 
        & (concepts_df.is_most_recent_version=='true')
        ).select('concept_id').toPandas()['concept_id'])
    weight_codeset_ids = list(concepts_df.where(
        (concepts_df.concept_set_name=="Body weight (LG34372-9 and SNOMED)") 
        & (concepts_df.is_most_recent_version=='true')
        ).select('concept_id').toPandas()['concept_id'])
    height_codeset_ids = list(concepts_df.where(
        (concepts_df.concept_set_name=="Height (LG34373-7 + SNOMED)") 
        & (concepts_df.is_most_recent_version=='true')
        ).select('concept_id').toPandas()['concept_id'])
    
    pcr_ag_test_ids = list(concepts_df.where(
        (concepts_df.concept_set_name=="ATLAS SARS-CoV-2 rt-PCR and AG") 
        & (concepts_df.is_most_recent_version=='true')
        ).select('concept_id').toPandas()['concept_id'])
    antibody_test_ids = list(concepts_df.where(
        (concepts_df.concept_set_name=="Atlas #818 [N3C] CovidAntibody retry") 
        & (concepts_df.is_most_recent_version=='true')
        ).select('concept_id').toPandas()['concept_id'])
    covid_positive_measurement_ids = list(concepts_df.where(
        (concepts_df.concept_set_name=="ResultPos") 
        & (concepts_df.is_most_recent_version=='true')
        ).select('concept_id').toPandas()['concept_id'])
    covid_negative_measurement_ids = list(concepts_df.where(
        (concepts_df.concept_set_name=="ResultNeg") 
        & (concepts_df.is_most_recent_version=='true')
        ).select('concept_id').toPandas()['concept_id'])
 
    #add value columns for rows associated with the above concept sets, but only include BMI or height or weight when in reasonable range
    BMI_df = df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('Recorded_BMI', F.when(df.measurement_concept_id.isin(bmi_codeset_ids) & df.harmonized_value_as_number.between(lowest_acceptable_BMI, highest_acceptable_BMI), df.harmonized_value_as_number).otherwise(0)) \
        .withColumn('height', F.when(df.measurement_concept_id.isin(height_codeset_ids) & df.harmonized_value_as_number.between(lowest_acceptable_height, highest_acceptable_height), df.harmonized_value_as_number).otherwise(0)) \
        .withColumn('weight', F.when(df.measurement_concept_id.isin(weight_codeset_ids) & df.harmonized_value_as_number.between(lowest_acceptable_weight, highest_acceptable_weight), df.harmonized_value_as_number).otherwise(0)) 
        
    labs_df = df.withColumn('PCR_AG_Pos', F.when(df.measurement_concept_id.isin(pcr_ag_test_ids) & df.value_as_concept_id.isin(covid_positive_measurement_ids), 1).otherwise(0)) \
        .withColumn('PCR_AG_Neg', F.when(df.measurement_concept_id.isin(pcr_ag_test_ids) & df.value_as_concept_id.isin(covid_negative_measurement_ids), 1).otherwise(0)) \
        .withColumn('Antibody_Pos', F.when(df.measurement_concept_id.isin(antibody_test_ids) & df.value_as_concept_id.isin(covid_positive_measurement_ids), 1).otherwise(0)) \
        .withColumn('Antibody_Neg', F.when(df.measurement_concept_id.isin(antibody_test_ids) & df.value_as_concept_id.isin(covid_negative_measurement_ids), 1).otherwise(0))
     
    #collapse all reasonable values to unique person and visit rows
    BMI_df = BMI_df.groupby('person_id', 'visit_date').agg(
    F.max('Recorded_BMI').alias('Recorded_BMI'),
    F.max('height').alias('height'),
    F.max('weight').alias('weight'))
    labs_df = labs_df.groupby('person_id', 'visit_date').agg(
    F.max('PCR_AG_Pos').alias('PCR_AG_Pos'),
    F.max('PCR_AG_Neg').alias('PCR_AG_Neg'),
    F.max('Antibody_Pos').alias('Antibody_Pos'),
    F.max('Antibody_Neg').alias('Antibody_Neg'))

    #add a calculated BMI for each visit date when height and weight available.  Note that if only one is available, it will result in zero
    #subsequent filter out rows that would have resulted from unreasonable calculated_BMI being used as best_BMI for the visit 
    BMI_df = BMI_df.withColumn('calculated_BMI', (BMI_df.weight/(BMI_df.height*BMI_df.height)))
    BMI_df = BMI_df.withColumn('BMI', F.when(BMI_df.Recorded_BMI>0, BMI_df.Recorded_BMI).otherwise(BMI_df.calculated_BMI)) \
        .select('person_id','visit_date','BMI')
    BMI_df = BMI_df.filter((BMI_df.BMI<=highest_acceptable_BMI) & (BMI_df.BMI>=lowest_acceptable_BMI)) \
        .withColumn('BMI_rounded', F.round(BMI_df.BMI)) \
        .drop('BMI')
    BMI_df = BMI_df.withColumn('OBESITY', F.when(BMI_df.BMI_rounded>=30, 1).otherwise(0))

    #join BMI_df with labs_df to retain all lab results with only reasonable BMI_rounded and OBESITY flags
    df = labs_df.join(BMI_df, on=['person_id', 'visit_date'], how='left') 

    return df

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.vector.main.execute.415c777e-5e75-4689-be8f-637f374cf040"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    customized_concept_set_input=Input(rid="ri.vector.main.execute.fab1d6ae-e7fb-434e-bf54-ddc10591ac6d"),
    everyone_cohort_de_id=Input(rid="ri.vector.main.execute.8dc65c1f-39e5-4bb7-b5c0-161a2f87aa0e"),
    observation=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_observations_of_interest(observation, concept_set_members, everyone_cohort_de_id, customized_concept_set_input):
   
    #bring in only cohort patient ids
    persons = everyone_cohort_de_id.select('person_id')
    #filter observations table to only cohort patients    
    observations_df = observation \
        .select('person_id','observation_date','observation_concept_id') \
        .where(F.col('observation_date').isNotNull()) \
        .withColumnRenamed('observation_date','visit_date') \
        .withColumnRenamed('observation_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the observations domain
    fusion_df = customized_concept_set_input \
        .filter(customized_concept_set_input.domain.contains('observation')) \
        .select('concept_set_name','indicator_prefix')
    #filter concept set members table to only concept ids for the observations of interest
    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where(F.col('is_most_recent_version')=='true') \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')

    #find observations information based on matching concept ids for observations of interest
    df = observations_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for observations of interest    
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df

@transform_pandas(
    Output(rid="ri.vector.main.execute.944ed1b3-eaf6-4134-b52b-3507c0267a90"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    customized_concept_set_input_testing=Input(rid="ri.vector.main.execute.4156447b-8bad-4af5-9150-010e304fa65a"),
    everyone_cohort_de_id_testing=Input(rid="ri.vector.main.execute.0ca5c190-d2ae-492a-b291-66fd8092b269"),
    observation_testing=Input(rid="ri.foundry.main.dataset.fc1ce22e-9cf6-4335-8ca7-aa8c733d506d")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_observations_of_interest_testing(observation_testing, concept_set_members, everyone_cohort_de_id_testing, customized_concept_set_input_testing):
    everyone_cohort_de_id = everyone_cohort_de_id_testing
    customized_concept_set_input = customized_concept_set_input_testing
   
    #bring in only cohort patient ids
    persons = everyone_cohort_de_id_testing.select('person_id')
    #filter observations table to only cohort patients    
    observations_df = observation_testing \
        .select('person_id','observation_date','observation_concept_id') \
        .where(F.col('observation_date').isNotNull()) \
        .withColumnRenamed('observation_date','visit_date') \
        .withColumnRenamed('observation_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the observations domain
    fusion_df = customized_concept_set_input \
        .filter(customized_concept_set_input.domain.contains('observation')) \
        .select('concept_set_name','indicator_prefix')
    #filter concept set members table to only concept ids for the observations of interest
    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where(F.col('is_most_recent_version')=='true') \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')

    #find observations information based on matching concept ids for observations of interest
    df = observations_df.join(concepts_df, 'concept_id', 'right')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for observations of interest    
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df

@transform_pandas(
    Output(rid="ri.vector.main.execute.873aa4ba-3bc4-4864-bbab-5d11ddd4884b"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    customized_concept_set_input=Input(rid="ri.vector.main.execute.fab1d6ae-e7fb-434e-bf54-ddc10591ac6d"),
    everyone_cohort_de_id=Input(rid="ri.vector.main.execute.8dc65c1f-39e5-4bb7-b5c0-161a2f87aa0e"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_procedures_of_interest(everyone_cohort_de_id, concept_set_members, procedure_occurrence, customized_concept_set_input):
  
    #bring in only cohort patient ids
    persons = everyone_cohort_de_id.select('person_id')
    #filter procedure occurrence table to only cohort patients    
    procedures_df = procedure_occurrence \
        .select('person_id','procedure_date','procedure_concept_id') \
        .where(F.col('procedure_date').isNotNull()) \
        .withColumnRenamed('procedure_date','visit_date') \
        .withColumnRenamed('procedure_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the procedure domain
    fusion_df = customized_concept_set_input \
        .filter(customized_concept_set_input.domain.contains('procedure')) \
        .select('concept_set_name','indicator_prefix')
    #filter concept set members table to only concept ids for the procedures of interest
    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where(F.col('is_most_recent_version')=='true') \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')
 
    #find procedure occurrence information based on matching concept ids for procedures of interest
    df = procedures_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for procedures of interest    
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df
    

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.vector.main.execute.ff49312e-197f-4017-a29c-930f887d57ff"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    customized_concept_set_input_testing=Input(rid="ri.vector.main.execute.4156447b-8bad-4af5-9150-010e304fa65a"),
    everyone_cohort_de_id_testing=Input(rid="ri.vector.main.execute.0ca5c190-d2ae-492a-b291-66fd8092b269"),
    procedure_occurrence_testing=Input(rid="ri.foundry.main.dataset.88523aaa-75c3-4b55-a79a-ebe27e40ba4f")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_procedures_of_interest_testing(everyone_cohort_de_id_testing, concept_set_members, procedure_occurrence_testing, customized_concept_set_input_testing):
    everyone_cohort_de_id = everyone_cohort_de_id_testing
    customized_concept_set_input = customized_concept_set_input_testing
  
    #bring in only cohort patient ids
    persons = everyone_cohort_de_id_testing.select('person_id')
    #filter procedure occurrence table to only cohort patients    
    procedures_df = procedure_occurrence_testing \
        .select('person_id','procedure_date','procedure_concept_id') \
        .where(F.col('procedure_date').isNotNull()) \
        .withColumnRenamed('procedure_date','visit_date') \
        .withColumnRenamed('procedure_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the procedure domain
    fusion_df = customized_concept_set_input \
        .filter(customized_concept_set_input.domain.contains('procedure')) \
        .select('concept_set_name','indicator_prefix')
    #filter concept set members table to only concept ids for the procedures of interest
    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where(F.col('is_most_recent_version')=='true') \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')
 
    #find procedure occurrence information based on matching concept ids for procedures of interest
    df = procedures_df.join(concepts_df, 'concept_id', 'right')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for procedures of interest    
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df
    

#################################################
## Global imports and functions included below ##
#################################################

