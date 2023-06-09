from src.global_code import *
from src.utils import *
from pyspark.sql.types import StringType
def custom_concept_set_members(concept_set_members):
    df = concept_set_members
    max_codeset_id = int(df.agg({"codeset_id":"max"}).collect()[0][0])
    more = df.limit(1).toPandas()
    #concept_id, concept_name, concept_set_name (all other fields will be autoassigned)
    #ADD NEW SETS HERE
    data = [
        ["40170911", "liraglutide", "liraglutide_penn"],
        ["19127775","prednisone 5 MG Oral Capsule", "predisone_penn"],
        ["19078925","midazolam 5 MG/ML Injectable Solution", "midazolam_penn"],
        ["45774751","empagliflozin", "empagliflozin_penn"],
        ["40170911", "liraglutide", "liraglutide_penn"],
        ["4185623", "Fall risk assessment", "fall_risk_assessment_penn"],
        ["40762523","Fall risk total [Morse Fall Scale]", "fall_risk_assessment_penn"],
        ["43018325", "Performance of Urinary Filtration, Continuous, Greater than 18 hours Per Day", "urinary_filtration_penn"],
        ["3661408", "Pneumonia caused by SARS-CoV-2", "pneumonia_penn"],
        ["42538827", "Uses contraception", "contraception_penn"],
        ["43018325", "Performance of Urinary Filtration, Continuous, Greater than 18 hours Per Day", "urinary_filtration_penn"],
        ["21494995", "Pain assessment [Interpretation]", "pain_assessment_penn"],
        ["3035482", "Pain duration - Reported", "pain_duration_penn"],
        ["4271661", "Characteristic of pain", "characteristic_pain_penn"],
        ["1367500", "losartan", "losartan_penn"],
        ["903963", "triamcinolone", "triamcinolone_penn"],
        ["1336926", "tadalafil", "tadalafil_penn"],
        ["1367571", "heparin", "heparin_penn"],
        ["1112921", "ipratropium", "ipratropium_penn"],
        ["798875", "clonazepam 0.5 MG Oral Tablet", "clonazepam_penn"],
        ["713823", "ropinirole", "ropinirole_penn"],
        ["19045045", "ergocalciferol", "ergocalciferol_penn"],
        ["1154343", "albuterol", "albuterol_penn"],
        ["19095164", "cholecalciferol", "cholecalciferol_penn"],
        ["1149380", "fluticasone", "fluticasone_penn"],
        ["753626", "propofol", "propofol_penn"],
        ["950637", "tacrolimus", "tacrolimus_penn"],
        ["975125", "hydrocortisone", "hydrocortisone_penn"],
        ["1308738", "vitamin B12", "B12_penn"],
        ["1136601", "benzonatate", "benzonatate_penn"],
        ["1192218","levalbuterol","levalbuterol_penn"],
        ["1545958", "atorvastatin", "atorvastatin_penn"],
        ["924566","tamsulosin","tamsulosin_penn"],
        ["2108253","Collection of blood specimen from a completely implantable venous access device", "venous_implant_penn"],
        ["74582","Primary malignant neoplasm of rectum", "neoplasm_penn"],
        ["4218813","Third trimester pregnancy","pregnant_penn"],
        ["19003999","mycophenolate mofetil", "mofetil_penn"],
        ["950637", "tacrolimus", "tacrolimus_penn"],
        ["1551860","pravastatin","pravastatin_penn"],
        ["1501700","levothyroxine","levothyroxine_penn"],
        ["1149380","fluticasone","fluticasone_penn"],
        ["2514406","Initial hospital care, per day, for the evaluation and management of a patient, which requires these 3 key components: A comprehensive history; A comprehensive examination; and Medical decision making of high complexity. Counseling and/or coordination of care with other physicians, other qualified health care professionals, or agencies are provided consistent with the nature of the problem(s) and the patient's and/or family's needs. Usually, the problem(s) requiring admission are of high severity. Typically, 70 minutes are spent at the bedside and on the patient's hospital floor or unit.", "hospitalized_penn"],
        ["2514527","Periodic comprehensive preventive medicine reevaluation and management of an individual including an age and gender appropriate history, examination, counseling/anticipatory guidance/risk factor reduction interventions, and the ordering of laboratory/diagnostic procedures, established patient; 18-39 years","periodic_checkup_penn"],
        ["19095164","cholecalciferol","cholecalciferol_penn"],
        ["923645","omeprazole","omeprazole_penn"],
        ["1136601","benzonatate","benzonatate_penn"],
        ["2787823","Assistance with Respiratory Ventilation, Less than 24 Consecutive Hours, Continuous Positive Airway Pressure","ventilator_penn"],
        ["2788038","Respiratory Ventilation, Greater than 96 Consecutive Hours", "ventilator_penn"],
        ["1781162","Assistance with Respiratory Ventilation, Greater than 96 Consecutive Hours, High Nasal Flow/Velocity", "ventilator_penn"],
        ["1781160", "Assistance with Respiratory Ventilation, Less than 24 Consecutive Hours, High Nasal Flow/Velocity", "ventilator_penn"],
        ["2788037", "Respiratory Ventilation, 24-96 Consecutive Hours", "ventilator_penn"],
        ["4230167", "Artificial respiration", "ventilator_penn"],
        ["2745444", "Insertion of Endotracheal Airway into Trachea, Via Natural or Artificial Opening", "tracheostomy_penn"],
        ["2106562", "Tracheostomy, planned (separate procedure)", "tracheostomy_penn"],
        ["2786229", "Introduction of Anti-inflammatory into Peripheral Vein, Percutaneous Approach", "antiinflammatory_penn"],
        ["2787749", "Introduction of Anti-inflammatory into Mouth and Pharynx, External Approach", "antiinflammatory_penn"],
        ["1332418","amlodipine","amlodipine_penn"],
        ["435788","Disorder of phosphorus metabolism","metabolism_disorder_penn"],
        ["2106281", "Most recent systolic blood pressure less than 130 mm Hg (DM), (HTN, CKD, CAD)", "bloodpressure_penn"],
        ["257907","Disorder of lung","lungdisorder_penn"],
        ["1567198","insulin aspart, human", "insulin_penn"],
        ["739138", "sertraline", "sertraline_penn"]
    ]
    data = [[int(r[0]), r[1], r[2]] for r in data]
    #
    #codeset_id, concept_id, concept_set_name, is_most_recent (true),version (1), concept_name, archived (false)
    new_sets = {}
    for concept_id, concept_name, concept_set_name in data:
        if concept_set_name not in new_sets:
            max_codeset_id += 1
            new_sets[concept_set_name] = max_codeset_id
        more.loc[len(more.index)] = [new_sets[concept_set_name], concept_id, concept_set_name, True, 1, concept_name, False]
    more = more.iloc[1: , :]
    spark = SparkSession.builder.master("local[1]").getOrCreate()
    more = spark.createDataFrame(more, df.schema)
    mems = more.union(df)
    return mems

def everyone_cohort_de_id( person, microvisits_to_macrovisits, custom_concept_set_members):
    concept_set_members = custom_concept_set_members

    """
    Select proportion of enclave patients to use: A value of 1.0 indicates the pipeline will use all patients in the persons table.
    A value less than 1.0 takes a random sample of the patients with a value of 0.001 (for example) representing a 0.1% sample of the persons table will be used.
    """
    proportion_of_patients_to_use = 1.0

    concepts_df = concept_set_members

    df = person \
        .select('person_id','year_of_birth','month_of_birth','day_of_birth') \
        .distinct() \
        .sample(False, proportion_of_patients_to_use, 111)

    visits_df = microvisits_to_macrovisits.select("person_id", "visit_start_date")



    #join in location_df data to person_sample dataframe


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



    min_reasonable_dob = "1902-01-01"
    max_reasonable_dob = F.current_date()

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




    #create visit counts/obs period dataframes
    hosp_visits = visits_df.where(F.col("visit_start_date").isNotNull()) \
        .orderBy("visit_start_date") \
        .coalesce(1) \
        .dropDuplicates(["person_id", "visit_start_date"]) #hospital

    non_hosp_visits = visits_df.where(F.col("visit_start_date").isNull()) \
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


    df = df.select('person_id',
        'total_visits',
        'observation_period',
        'age',)

    return df

def everyone_conditions_of_interest(everyone_cohort_de_id, condition_occurrence, customized_concept_set_input, custom_concept_set_members):
    concept_set_members = custom_concept_set_members

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
        .where((F.col('is_most_recent_version')=='true') | (F.col('concept_set_name') == 'Long-COVID (PASC)') | (F.col('is_most_recent_version')==True)) \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')

    #find conditions information based on matching concept ids for conditions of interest
    df = conditions_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for conditions of interest
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df

def everyone_observations_of_interest(observation, everyone_cohort_de_id, customized_concept_set_input, custom_concept_set_members):
    concept_set_members = custom_concept_set_members

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
        .where((F.col('is_most_recent_version')=='true') | (F.col('is_most_recent_version')==True)) \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')

    #find observations information based on matching concept ids for observations of interest
    df = observations_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for observations of interest
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df

def everyone_procedures_of_interest(everyone_cohort_de_id, procedure_occurrence, customized_concept_set_input, custom_concept_set_members):
    concept_set_members = custom_concept_set_members

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
        .where((F.col('is_most_recent_version')=='true') | (F.col('is_most_recent_version')==True)) \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')

    #find procedure occurrence information based on matching concept ids for procedures of interest
    df = procedures_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for procedures of interest
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df

def everyone_devices_of_interest(device_exposure, everyone_cohort_de_id, customized_concept_set_input, custom_concept_set_members):
    concept_set_members = custom_concept_set_members

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
        .where((F.col('is_most_recent_version')=='true')  | (F.col('is_most_recent_version')==True)) \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')

    #find device exposure information based on matching concept ids for devices of interest
    df = devices_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for devices of interest
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df

def everyone_drugs_of_interest( drug_exposure, everyone_cohort_de_id, customized_concept_set_input, first_covid_positive, custom_concept_set_members):
    concept_set_members = custom_concept_set_members

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
        .where((F.col('is_most_recent_version')=='true') | (F.col('is_most_recent_version')==True)) \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix', 'concept_set_name')

    #find drug exposure information based on matching concept ids for drugs of interest
    df = drug_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for drugs of interest
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0) \
        .join(first_covid_positive, 'person_id', 'leftouter') \
        .withColumn('BEFORE_FCP', F.when(F.datediff(F.col('visit_date'), F.col('first_covid_positive')) < 0, 1).otherwise(0)) \
        .withColumn('DAYS_SINCE_FCP', F.datediff(F.col('visit_date'), F.col('first_covid_positive'))) \
        .drop(F.col('first_covid_positive'))

    return df

def everyone_measurements_of_interest(measurement, everyone_cohort_de_id, custom_concept_set_members):
    concept_set_members = custom_concept_set_members

    #bring in only cohort patient ids
    persons = everyone_cohort_de_id.select('person_id', 'gender_concept_name')
    #filter procedure occurrence table to only cohort patients
    df = measurement \
        .select('person_id','measurement_date','measurement_concept_id','value_as_number','value_as_concept_id','value_as_number') \
        .where(F.col('measurement_date').isNotNull()) \
        .withColumnRenamed('measurement_date','visit_date') \
        .join(persons,'person_id','inner')

    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where((F.col('is_most_recent_version')=='true') | (F.col('is_most_recent_version')==True))

    #Find BMI closest to COVID using both reported/observed BMI and calculated BMI using height and weight.  Cutoffs for reasonable height, weight, and BMI are provided and can be changed by the template user.
    lowest_acceptable_BMI = 10
    highest_acceptable_BMI = 100
    lowest_acceptable_weight = 5 #in kgs
    highest_acceptable_weight = 300 #in kgs
    lowest_acceptable_height = .6 #in meters
    highest_acceptable_height = 2.43 #in meters
#40762499
    blood_oxygen_codeset_id=[40762499] # normal people: 75-100
    lowest_blood_oxygen = 20
    highest_blood_oxygen = 100

    blood_sodium_codeset_id=[3019550]    # normal people: 137-145
    lowest_blood_sodium = 90
    highest_blood_sodium = 200

    blood_hemoglobin_codeset_id=[3000963]  # normal people: 11-16
    lowest_blood_hemoglobin = 3
    highest_blood_hemoglobin = 40

    respiratory_rate_codeset_id=[3024171]  # normal people: 12-20
    lowest_respiratory_rate=5
    highest_respiratory_rate=60

    blood_Creatinine_codeset_id=[3016723]  # normal people: 0.6-1.3
    lowest_blood_Creatinine = 0.2
    highest_blood_Creatinine = 5

    blood_UreaNitrogen_codeset_id=[3013682]  # normal people: 10-20
    lowest_blood_UreaNitrogen = 3
    highest_blood_UreaNitrogen = 80

    blood_Potassium_codeset_id=[3023103]  # normal people: 3.5-5.0 mEq/L
    lowest_blood_Potassium = 1
    highest_blood_Potassium = 30

    blood_Chloride_codeset_id=[3014576]  # normal people: 96-106 mEq/L
    lowest_blood_Chloride = 60
    highest_blood_Chloride = 300

    blood_Calcium_codeset_id=[3006906]  # normal people: 8.5-10.2 mg/dL
    lowest_blood_Calcium = 3
    highest_blood_Calcium = 30

    MCV_codeset_id=[3023599]  # normal people: 80-100 fl
    lowest_MCV = 50
    highest_MCV = 300

    Erythrocytes_codeset_id=[3020416]  # normal people: 4-6 million cells per microliter
    lowest_Erythrocytes = 1
    highest_Erythrocytes = 20

    MCHC_codeset_id=[3009744]  # normal people: 31-37 g/dL
    lowest_MCHC = 10
    highest_MCHC = 60

    Systolic_blood_pressure_codeset_id=[3004249]
    lowest_Systolic_blood_pressure = 0
    highest_Systolic_blood_pressure = 500

    Diastolic_blood_pressure_codeset_id=[3012888,4154790]
    lowest_Diastolic_blood_pressure = 0
    highest_Diastolic_blood_pressure = 500

    heart_rate_codeset_id=[3027018]  # normal people: 60-100 per min
    lowest_heart_rate = 10
    highest_heart_rate = 300

    temperature_codeset_id=[3020891]  # normal people: 36-38
    lowest_temperature = 35
    highest_temperature = 43

    blood_Glucose_codeset_id=[3004501]  # normal people:
    lowest_blood_Glucose = 50
    highest_blood_Glucose = 500

    blood_Platelets_codeset_id=[3024929]  # normal people: 130-459
    lowest_blood_Platelets = 50
    highest_blood_Platelets = 1000

    blood_Hematocrit_codeset_id=[3023314]  # normal people: 30-54
    lowest_blood_Hematocrit = 10
    highest_blood_Hematocrit = 150

    blood_Leukocytes_codeset_id=[3000905]  # normal people: 4-11
    lowest_blood_Leukocytes = 1
    highest_blood_Leukocytes = 30

    blood_Bilirubin_codeset_id=[3024128]  # normal people: 0.1-1.5
    lowest_blood_Bilirubin = 0.02
    highest_blood_Bilirubin = 5

    blood_Albumin_codeset_id=[3024561]  # normal people: 3.5-5.0
    lowest_blood_Albumin = 1
    highest_blood_Albumin = 30

    ####
    blood_Troponin_codeset_id=[3033745]  # normal people: 0-0.01
    lowest_blood_Troponin = 0
    highest_blood_Troponin = 1

    blood_Procalcitonin_codeset_id=[44817130]  # normal people: 0-0.1
    lowest_blood_Procalcitonin = 0
    highest_blood_Procalcitonin = 1

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
    BMI_df = df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('Recorded_BMI', F.when(df.measurement_concept_id.isin(bmi_codeset_ids) & df.value_as_number.between(lowest_acceptable_BMI, highest_acceptable_BMI), df.value_as_number).otherwise(0)) \
        .withColumn('height', F.when(df.measurement_concept_id.isin(height_codeset_ids) & df.value_as_number.between(lowest_acceptable_height, highest_acceptable_height), df.value_as_number).otherwise(0)) \
        .withColumn('weight', F.when(df.measurement_concept_id.isin(weight_codeset_ids) & df.value_as_number.between(lowest_acceptable_weight, highest_acceptable_weight), df.value_as_number).otherwise(0))

    blood_oxygen_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('Oxygen_saturation', F.when(df.measurement_concept_id.isin(blood_oxygen_codeset_id) & df.value_as_number.between(lowest_blood_oxygen, highest_blood_oxygen), df.value_as_number).otherwise(0))

    blood_sodium_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_sodium', F.when(df.measurement_concept_id.isin(blood_sodium_codeset_id) & df.value_as_number.between(lowest_blood_sodium, highest_blood_sodium), df.value_as_number).otherwise(0))


    blood_hemoglobin_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_hemoglobin', F.when(df.measurement_concept_id.isin(blood_hemoglobin_codeset_id) & df.value_as_number.between(lowest_blood_hemoglobin, highest_blood_hemoglobin), df.value_as_number).otherwise(0))

    respiratory_rate_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('respiratory_rate', F.when(df.measurement_concept_id.isin(respiratory_rate_codeset_id) & df.value_as_number.between(lowest_respiratory_rate, highest_respiratory_rate), df.value_as_number).otherwise(0))

    blood_Creatinine_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Creatinine', F.when(df.measurement_concept_id.isin(blood_Creatinine_codeset_id) & df.value_as_number.between(lowest_blood_Creatinine, highest_blood_Creatinine), df.value_as_number).otherwise(0))

    blood_UreaNitrogen_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_UreaNitrogen', F.when(df.measurement_concept_id.isin(blood_UreaNitrogen_codeset_id) & df.value_as_number.between(lowest_blood_UreaNitrogen, highest_blood_UreaNitrogen), df.value_as_number).otherwise(0))

    blood_Potassium_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Potassium', F.when(df.measurement_concept_id.isin(blood_Potassium_codeset_id) & df.value_as_number.between(lowest_blood_Potassium, highest_blood_Potassium), df.value_as_number).otherwise(0))

    blood_Chloride_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Chloride', F.when(df.measurement_concept_id.isin(blood_Chloride_codeset_id) & df.value_as_number.between(lowest_blood_Chloride, highest_blood_Chloride), df.value_as_number).otherwise(0))

    blood_Calcium_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Calcium', F.when(df.measurement_concept_id.isin(blood_Calcium_codeset_id) & df.value_as_number.between(lowest_blood_Calcium, highest_blood_Calcium), df.value_as_number).otherwise(0))

    MCV_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('MCV', F.when(df.measurement_concept_id.isin(MCV_codeset_id) & df.value_as_number.between(lowest_MCV, highest_MCV), df.value_as_number).otherwise(0))

    Erythrocytes_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('Erythrocytes', F.when(df.measurement_concept_id.isin(Erythrocytes_codeset_id) & df.value_as_number.between(lowest_Erythrocytes, highest_Erythrocytes), df.value_as_number).otherwise(0))

    MCHC_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('MCHC', F.when(df.measurement_concept_id.isin(MCHC_codeset_id) & df.value_as_number.between(lowest_MCHC, highest_MCHC), df.value_as_number).otherwise(0))

    Systolic_blood_pressure_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('Systolic_blood_pressure', F.when(df.measurement_concept_id.isin(Systolic_blood_pressure_codeset_id) & df.value_as_number.between(lowest_Systolic_blood_pressure, highest_Systolic_blood_pressure), df.value_as_number).otherwise(0))

    Diastolic_blood_pressure_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('Diastolic_blood_pressure', F.when(df.measurement_concept_id.isin(Diastolic_blood_pressure_codeset_id) & df.value_as_number.between(lowest_Diastolic_blood_pressure, highest_Diastolic_blood_pressure), df.value_as_number).otherwise(0))

    heart_rate_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('heart_rate', F.when(df.measurement_concept_id.isin(heart_rate_codeset_id) & df.value_as_number.between(lowest_heart_rate, highest_heart_rate), df.value_as_number).otherwise(0))

    temperature_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('temperature', F.when(df.measurement_concept_id.isin(temperature_codeset_id) & df.value_as_number.between(lowest_temperature, highest_temperature), df.value_as_number).otherwise(0))

    blood_Glucose_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Glucose', F.when(df.measurement_concept_id.isin(blood_Glucose_codeset_id) & df.value_as_number.between(lowest_blood_Glucose, highest_blood_Glucose), df.value_as_number).otherwise(0))

    blood_Platelets_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Platelets', F.when(df.measurement_concept_id.isin(blood_Platelets_codeset_id) & df.value_as_number.between(lowest_blood_Platelets, highest_blood_Platelets), df.value_as_number).otherwise(0))

    blood_Hematocrit_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Hematocrit', F.when(df.measurement_concept_id.isin(blood_Hematocrit_codeset_id) & df.value_as_number.between(lowest_blood_Hematocrit, highest_blood_Hematocrit), df.value_as_number).otherwise(0))

    blood_Leukocytes_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Leukocytes', F.when(df.measurement_concept_id.isin(blood_Leukocytes_codeset_id) & df.value_as_number.between(lowest_blood_Leukocytes, highest_blood_Leukocytes), df.value_as_number).otherwise(0))

    blood_Bilirubin_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Bilirubin', F.when(df.measurement_concept_id.isin(blood_Bilirubin_codeset_id) & df.value_as_number.between(lowest_blood_Bilirubin, highest_blood_Bilirubin), df.value_as_number).otherwise(0))

    blood_Albumin_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Albumin', F.when(df.measurement_concept_id.isin(blood_Albumin_codeset_id) & df.value_as_number.between(lowest_blood_Albumin, highest_blood_Albumin), df.value_as_number).otherwise(0))

    ####
    blood_Troponin_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Troponin', F.when(df.measurement_concept_id.isin(blood_Troponin_codeset_id) & df.value_as_number.between(lowest_blood_Troponin, highest_blood_Troponin), df.value_as_number).otherwise(0))

    blood_Procalcitonin_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Procalcitonin', F.when(df.measurement_concept_id.isin(blood_Procalcitonin_codeset_id) & df.value_as_number.between(lowest_blood_Procalcitonin, highest_blood_Procalcitonin), df.value_as_number).otherwise(0))

    labs_df = df.withColumn('PCR_AG_Pos', F.when(df.measurement_concept_id.isin(pcr_ag_test_ids) & df.value_as_concept_id.isin(covid_positive_measurement_ids), 1).otherwise(0)) \
        .withColumn('PCR_AG_Neg', F.when(df.measurement_concept_id.isin(pcr_ag_test_ids) & df.value_as_concept_id.isin(covid_negative_measurement_ids), 1).otherwise(0)) \
        .withColumn('Antibody_Pos', F.when(df.measurement_concept_id.isin(antibody_test_ids) & df.value_as_concept_id.isin(covid_positive_measurement_ids), 1).otherwise(0)) \
        .withColumn('Antibody_Neg', F.when(df.measurement_concept_id.isin(antibody_test_ids) & df.value_as_concept_id.isin(covid_negative_measurement_ids), 1).otherwise(0)) \
        .withColumn('SEX', F.when(F.col('gender_concept_name') == 'FEMALE', 1).otherwise(0))

    #collapse all reasonable values to unique person and visit rows
    BMI_df = BMI_df.groupby('person_id', 'visit_date').agg(
    F.max('Recorded_BMI').alias('Recorded_BMI'),
    F.max('height').alias('height'),
    F.max('weight').alias('weight'))
    labs_df = labs_df.groupby('person_id', 'visit_date').agg(
    F.max('PCR_AG_Pos').alias('PCR_AG_Pos'),
    F.max('PCR_AG_Neg').alias('PCR_AG_Neg'),
    F.max('Antibody_Pos').alias('Antibody_Pos'),
    F.max('Antibody_Neg').alias('Antibody_Neg'),
    F.max('SEX').alias('SEX'))

    blood_oxygen_df = blood_oxygen_df.groupby('person_id', 'visit_date').agg(
    F.max('Oxygen_saturation').alias('Oxygen_saturation')
    )

    blood_sodium_df = blood_sodium_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_sodium').alias('blood_sodium')
    )

    blood_hemoglobin_df = blood_hemoglobin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_hemoglobin').alias('blood_hemoglobin')
    )

    respiratory_rate_df = respiratory_rate_df.groupby('person_id', 'visit_date').agg(
    F.max('respiratory_rate').alias('respiratory_rate')
    )

    blood_Creatinine_df = blood_Creatinine_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Creatinine').alias('blood_Creatinine')
    )

    blood_UreaNitrogen_df = blood_UreaNitrogen_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_UreaNitrogen').alias('blood_UreaNitrogen')
    )

    blood_Potassium_df = blood_Potassium_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Potassium').alias('blood_Potassium')
    )

    blood_Chloride_df = blood_Chloride_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Chloride').alias('blood_Chloride')
    )

    blood_Calcium_df = blood_Calcium_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Calcium').alias('blood_Calcium')
    )

    MCV_df = MCV_df.groupby('person_id', 'visit_date').agg(
    F.max('MCV').alias('MCV')
    )

    Erythrocytes_df = Erythrocytes_df.groupby('person_id', 'visit_date').agg(
    F.max('Erythrocytes').alias('Erythrocytes')
    )

    MCHC_df = MCHC_df.groupby('person_id', 'visit_date').agg(
    F.max('MCHC').alias('MCHC')
    )

    Systolic_blood_pressure_df = Systolic_blood_pressure_df.groupby('person_id', 'visit_date').agg(
    F.max('Systolic_blood_pressure').alias('Systolic_blood_pressure')
    )

    Diastolic_blood_pressure_df = Diastolic_blood_pressure_df.groupby('person_id', 'visit_date').agg(
    F.max('Diastolic_blood_pressure').alias('Diastolic_blood_pressure')
    )

    heart_rate_df = heart_rate_df.groupby('person_id', 'visit_date').agg(
    F.max('heart_rate').alias('heart_rate')
    )

    temperature_df = temperature_df.groupby('person_id', 'visit_date').agg(
    F.max('temperature').alias('temperature')
    )

    blood_Glucose_df = blood_Glucose_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Glucose').alias('blood_Glucose')
    )

    blood_Platelets_df = blood_Platelets_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Platelets').alias('blood_Platelets')
    )

    blood_Hematocrit_df = blood_Hematocrit_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Hematocrit').alias('blood_Hematocrit')
    )

    blood_Leukocytes_df = blood_Leukocytes_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Leukocytes').alias('blood_Leukocytes')
    )

    blood_Bilirubin_df = blood_Bilirubin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Bilirubin').alias('blood_Bilirubin')
    )

    blood_Albumin_df = blood_Albumin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Albumin').alias('blood_Albumin')
    )
    ###
    blood_Troponin_df = blood_Troponin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Troponin').alias('blood_Troponin')
    )

    blood_Procalcitonin_df = blood_Procalcitonin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Procalcitonin').alias('blood_Procalcitonin')
    )

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
    df = labs_df.join(BMI_df, on=['person_id', 'visit_date'], how='left').join(blood_oxygen_df, on=['person_id', 'visit_date'], how='left').join(blood_sodium_df, on=['person_id', 'visit_date'], how='left').join(blood_hemoglobin_df, on=['person_id', 'visit_date'], how='left').join(respiratory_rate_df, on=['person_id', 'visit_date'], how='left').join(blood_Creatinine_df, on=['person_id', 'visit_date'], how='left').join(blood_UreaNitrogen_df, on=['person_id', 'visit_date'], how='left').join(blood_Potassium_df, on=['person_id', 'visit_date'], how='left').join(blood_Chloride_df, on=['person_id', 'visit_date'], how='left').join(blood_Calcium_df, on=['person_id', 'visit_date'], how='left').join(MCV_df, on=['person_id', 'visit_date'], how='left').join(Erythrocytes_df, on=['person_id', 'visit_date'], how='left').join(MCHC_df, on=['person_id', 'visit_date'], how='left').join(Systolic_blood_pressure_df, on=['person_id', 'visit_date'], how='left').join(Diastolic_blood_pressure_df, on=['person_id', 'visit_date'], how='left').join(heart_rate_df, on=['person_id', 'visit_date'], how='left').join(temperature_df, on=['person_id', 'visit_date'], how='left').join(blood_Glucose_df, on=['person_id', 'visit_date'], how='left').join(blood_Platelets_df, on=['person_id', 'visit_date'], how='left').join(blood_Hematocrit_df, on=['person_id', 'visit_date'], how='left').join(blood_Leukocytes_df, on=['person_id', 'visit_date'], how='left').join(blood_Bilirubin_df, on=['person_id', 'visit_date'], how='left').join(blood_Albumin_df, on=['person_id', 'visit_date'], how='left').join(blood_Troponin_df, on=['person_id', 'visit_date'], how='left').join(blood_Procalcitonin_df, on=['person_id', 'visit_date'], how='left')

    return df


def everyone_conditions_of_interest(everyone_cohort_de_id, condition_occurrence, customized_concept_set_input, custom_concept_set_members):
    concept_set_members = custom_concept_set_members

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
        .where((F.col('is_most_recent_version')=='true') | (F.col('concept_set_name') == 'Long-COVID (PASC)') | (F.col('is_most_recent_version')==True)) \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')

    #find conditions information based on matching concept ids for conditions of interest
    df = conditions_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for conditions of interest
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df

def custom_sets(LL_concept_sets_fusion_everyone):
    df = LL_concept_sets_fusion_everyone.toPandas()
    df.loc[len(df.index)] = ['ventilator', 'VENTILATOR', 'device']
    df.loc[len(df.index)] = ['anxiety-broad', 'ANXIETY', 'observation,condition']
    df.loc[len(df.index)] = ['diabetes-broad', 'DIABETESCOMPLICATED', 'observation,condition']
    df.loc[len(df.index)] = ['dyspnea-broad', 'DYSPNEA', 'condition,observation']
    df.loc[len(df.index)] = ['mental-broad', 'MENTAL', 'condition,observation']
    df.loc[len(df.index)] = ['insomnia-broad', 'INSOMNIA', 'condition,observation']
    df.loc[len(df.index)] = ['palpitations-broad', 'PALPITATIONS', 'condition,observation']
    df.loc[len(df.index)] = ['NIH Systemic Corticosteroids', 'SYSTEMICCORTICOSTEROIDS', 'drug']
    df.loc[len(df.index)] = ['anosmia-broad', 'ANOSMIA', 'condition,observation']
    df.loc[len(df.index)] = ['systemic steroids ITM', 'STEROIDS', 'drug']
    df.loc[len(df.index)] = ['prednisone, prednisolone, methylprednisolone, dexamethasone', 'PREDNISONE', 'drug']
    df.loc[len(df.index)] = ['ARIScience - Respiratory Disorder - JA', 'RESPIRATORY', 'condition,observation']
    df.loc[len(df.index)] = ['ARIScience - Lung Disorder - JA', 'RESPIRATORY', 'condition,observation']
    df.loc[len(df.index)] = ['dexamethasone', 'DEXAMETHASONE', 'drug']
    df.loc[len(df.index)] = ['Long Hauler symptoms from LANCET paper', 'LANCET', 'condition,observation']
    df.loc[len(df.index)] = ['Systemic Antibiotics', 'ANTIBIOTICS', 'drug']
    df.loc[len(df.index)] = ['Antibiotics_wide', 'ANTIBIOTICS', 'drug']
    df.loc[len(df.index)] = ['liraglutide_penn', 'LIRAGLUTIDE', 'drug']
    df.loc[len(df.index)] = ['prednisone_penn', 'PREDNISONE', 'drug']
    df.loc[len(df.index)] = ['midazolam_penn', 'MIDAZOLAM', 'drug']
    df.loc[len(df.index)] = ['empagliflozin_penn', 'empagliflozin', 'drug']
    df.loc[len(df.index)] = ['fall_risk_assessment_penn', 'FALL_RISK', 'procedure, observation']
    df.loc[len(df.index)] = ['urinary_filtration_penn', 'URINARY_FILTRATION', 'procedure']
    df.loc[len(df.index)] = ['pneumonia_penn', 'pneumonia', 'condition']
    df.loc[len(df.index)] = ['contraception_penn', 'contraception', 'condition']

    df.loc[len(df.index)] = ["albuterol_penn", "ALBUTEROL", "drug"]
    df.loc[len(df.index)] = ["cholecalciferol_penn", "CHOLECALCIFEROL", "drug"]
    df.loc[len(df.index)] = ["propofol_penn", "PROPOFOL", "drug"]
    df.loc[len(df.index)] = ["tacrolimus_penn", "TACROLIMUS", "drug"]
    df.loc[len(df.index)] = ["hydrocortisone_penn", "HYDROCORTISONE", "drug"]
    df.loc[len(df.index)] = ["B12_penn", "VITAMIN_B12", "drug"]
    df.loc[len(df.index)] = ["benzonatate_penn", "BENZONATATE", "drug"]

    df.loc[len(df.index)] = ['pain_assessment_penn', 'PAIN_ASSESSMENT', 'observation']
    df.loc[len(df.index)] = ["pain_duration_penn", "PAIN_DURATION", "observation"]
    df.loc[len(df.index)] = ["characteristic_pain_penn", "PAIN_CHARACTERISTIC", "observation"]
    df.loc[len(df.index)] = ["losartan_penn", "LOSARTAN", "drug"]
    df.loc[len(df.index)] = ["triamcinolone_penn", "TRIAMCINOLONE", "drug"]
    df.loc[len(df.index)] = ["tadalafil_penn", "TADALAFIL", "drug"]
    df.loc[len(df.index)] = ["heparin_penn", "HEPARIN", "drug"]
    df.loc[len(df.index)] = ["ipratropium_penn", "IPRATROPIUM", "drug"]
    df.loc[len(df.index)] = ["clonazepam_penn", "CLONEAZEPAM", "drug"]
    df.loc[len(df.index)] = ["ropinirole_penn", "ROPINIROLE", "drug"]
    #df.loc[len(df.index)] = ["ergocalciferol_penn", "ERGOCALCIFEROL", "drug"]
    df.loc[len(df.index)] = ["levalbuterol_penn", "LEVALBUTEROL", "drug"]
    df.loc[len(df.index)] = ["atorvastatin_penn", "ATORVASTATIN", "drug"]
    df.loc[len(df.index)] = ["tamsulosin_penn", "TAMSULOSIN", "drug"]
    df.loc[len(df.index)] = ["venous_implant_penn", "VENOUSIMPLANT", "procedure"]
    df.loc[len(df.index)] = ["pregnant_penn", "PREGNANT", "condition"]
    df.loc[len(df.index)] = ["mofetil_penn", "MOFETIL", "drug"]
    #df.loc[len(df.index)] = ["tacrolimus_penn", "TACROLIMUS", "drug"]
    #df.loc[len(df.index)] = ["pravastatin_penn", "PRAVASTATIN", "drug"]
    df.loc[len(df.index)] = ["levothyroxine_penn", "LEVOTHYROXINE", "drug"]
    df.loc[len(df.index)] = ["fluticasone_penn", "FLUTICASONE", "drug"]
    df.loc[len(df.index)] = ["hospitalized_penn", "HOSPITALIZED", "procedure"]
    df.loc[len(df.index)] = ["periodic_checkup_penn", "PERIODICCHECKUP", "procedure"]
    df.loc[len(df.index)] = ["cholecalciferol_penn", "CHOLECALCIFEROL", "drug"]
    df.loc[len(df.index)] = ["omeprazole_penn", "OMEPRAZOLE", "drug"]
    df.loc[len(df.index)] = ["benzonatate_penn", "BENZONATATE", "drug"]
    df.loc[len(df.index)] = ["ventilator_penn", "VENTILATOR", "procedure"]
    df.loc[len(df.index)] = ["tracheostomy_penn", "TRACHEOSTOMY", "procedure"]
    df.loc[len(df.index)] = ["antiinflammatory_penn", "ANTIINFLAM", "procedure"]
    df.loc[len(df.index)] = ["electrocardiogram_penn", "ELECTROCARDIOGRAM", "procedure"]
    df.loc[len(df.index)] = ["respinfection_penn", "RESPINF", "condition"]
    df.loc[len(df.index)] = ["amlodipine_penn", "AMLODIPINE", "drug"]
    df.loc[len(df.index)] = ["metabolism_disorder_penn", "METADISORDER", "condition"]
    df.loc[len(df.index)] = ["bloodpressure_penn", "BLOODPRESSURE", "observation"]
    df.loc[len(df.index)] = ["lungdisorder_penn", "LUNGDISORDER", "condition"]
    df.loc[len(df.index)] = ["sertraline_penn", "SERTRALINE", "drug"]
    df.loc[len(df.index)] = ["insulin_penn", "INSULIN", "drug"]



    print(df)
    return df

def customized_concept_set_input( LL_DO_NOT_DELETE_REQUIRED_concept_sets_all, custom_sets):
    PM_LL_DO_NOT_DELETE_REQUIRED_concept_sets_all = LL_DO_NOT_DELETE_REQUIRED_concept_sets_all

    required = LL_DO_NOT_DELETE_REQUIRED_concept_sets_all
    customizable = custom_sets

    df = required.join(customizable, on = required.columns, how = 'outer')


    return df

def everyone_observations_of_interest(observation, everyone_cohort_de_id, customized_concept_set_input, custom_concept_set_members):
    concept_set_members = custom_concept_set_members

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
        .where((F.col('is_most_recent_version')=='true') | (F.col('is_most_recent_version')==True)) \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')

    #find observations information based on matching concept ids for observations of interest
    df = observations_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for observations of interest
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df

def everyone_procedures_of_interest(everyone_cohort_de_id, procedure_occurrence, customized_concept_set_input, custom_concept_set_members):
    concept_set_members = custom_concept_set_members

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
        .where((F.col('is_most_recent_version')=='true') | (F.col('is_most_recent_version')==True)) \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix')

    #find procedure occurrence information based on matching concept ids for procedures of interest
    df = procedures_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for procedures of interest
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)

    return df

def everyone_drugs_of_interest( drug_exposure, everyone_cohort_de_id, customized_concept_set_input, custom_concept_set_members):
    concept_set_members = custom_concept_set_members

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
        .where((F.col('is_most_recent_version')=='true') | (F.col('is_most_recent_version')==True)) \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix', 'concept_set_name')

    #find drug exposure information based on matching concept ids for drugs of interest
    df = drug_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for drugs of interest
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0)
    return df

def everyone_measurements_of_interest(measurement, everyone_cohort_de_id, custom_concept_set_members):
    concept_set_members = custom_concept_set_members

    #bring in only cohort patient ids
    persons = everyone_cohort_de_id.select('person_id')
    #filter procedure occurrence table to only cohort patients
    df = measurement \
        .select('person_id','measurement_date','measurement_concept_id','value_as_concept_id','value_as_number') \
        .where(F.col('measurement_date').isNotNull()) \
        .withColumnRenamed('measurement_date','visit_date') \
        .join(persons,'person_id','inner')

    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where((F.col('is_most_recent_version')=='true') | (F.col('is_most_recent_version')==True))

    #Find BMI closest to COVID using both reported/observed BMI and calculated BMI using height and weight.  Cutoffs for reasonable height, weight, and BMI are provided and can be changed by the template user.
    lowest_acceptable_BMI = 10
    highest_acceptable_BMI = 100
    lowest_acceptable_weight = 5 #in kgs
    highest_acceptable_weight = 300 #in kgs
    lowest_acceptable_height = .6 #in meters
    highest_acceptable_height = 2.43 #in meters
#40762499
    blood_oxygen_codeset_id=[40762499] # normal people: 75-100
    lowest_blood_oxygen = 20
    highest_blood_oxygen = 100

    blood_sodium_codeset_id=[3019550]    # normal people: 137-145
    lowest_blood_sodium = 90
    highest_blood_sodium = 200

    blood_hemoglobin_codeset_id=[3000963]  # normal people: 11-16
    lowest_blood_hemoglobin = 3
    highest_blood_hemoglobin = 40

    respiratory_rate_codeset_id=[3024171]  # normal people: 12-20
    lowest_respiratory_rate=5
    highest_respiratory_rate=60

    blood_Creatinine_codeset_id=[3016723]  # normal people: 0.6-1.3
    lowest_blood_Creatinine = 0.2
    highest_blood_Creatinine = 5

    blood_UreaNitrogen_codeset_id=[3013682]  # normal people: 10-20
    lowest_blood_UreaNitrogen = 3
    highest_blood_UreaNitrogen = 80

    blood_Potassium_codeset_id=[3023103]  # normal people: 3.5-5.0 mEq/L
    lowest_blood_Potassium = 1
    highest_blood_Potassium = 30

    blood_Chloride_codeset_id=[3014576]  # normal people: 96-106 mEq/L
    lowest_blood_Chloride = 60
    highest_blood_Chloride = 300

    blood_Calcium_codeset_id=[3006906]  # normal people: 8.5-10.2 mg/dL
    lowest_blood_Calcium = 3
    highest_blood_Calcium = 30

    MCV_codeset_id=[3023599]  # normal people: 80-100 fl
    lowest_MCV = 50
    highest_MCV = 300

    Erythrocytes_codeset_id=[3020416]  # normal people: 4-6 million cells per microliter
    lowest_Erythrocytes = 1
    highest_Erythrocytes = 20

    MCHC_codeset_id=[3009744]  # normal people: 31-37 g/dL
    lowest_MCHC = 10
    highest_MCHC = 60

    Systolic_blood_pressure_codeset_id=[3004249]
    lowest_Systolic_blood_pressure = 0
    highest_Systolic_blood_pressure = 500

    Diastolic_blood_pressure_codeset_id=[3012888,4154790]
    lowest_Diastolic_blood_pressure = 0
    highest_Diastolic_blood_pressure = 500

    heart_rate_codeset_id=[3027018]  # normal people: 60-100 per min
    lowest_heart_rate = 10
    highest_heart_rate = 300

    temperature_codeset_id=[3020891]  # normal people: 36-38
    lowest_temperature = 35
    highest_temperature = 43

    blood_Glucose_codeset_id=[3004501]  # normal people:
    lowest_blood_Glucose = 50
    highest_blood_Glucose = 500

    blood_Platelets_codeset_id=[3024929]  # normal people: 130-459
    lowest_blood_Platelets = 50
    highest_blood_Platelets = 1000

    blood_Hematocrit_codeset_id=[3023314]  # normal people: 30-54
    lowest_blood_Hematocrit = 10
    highest_blood_Hematocrit = 150

    blood_Leukocytes_codeset_id=[3000905]  # normal people: 4-11
    lowest_blood_Leukocytes = 1
    highest_blood_Leukocytes = 30

    blood_Bilirubin_codeset_id=[3024128]  # normal people: 0.1-1.5
    lowest_blood_Bilirubin = 0.02
    highest_blood_Bilirubin = 5

    blood_Albumin_codeset_id=[3024561]  # normal people: 3.5-5.0
    lowest_blood_Albumin = 1
    highest_blood_Albumin = 30

    ####
    blood_Troponin_codeset_id=[3033745]  # normal people: 0-0.01
    lowest_blood_Troponin = 0
    highest_blood_Troponin = 1

    blood_Procalcitonin_codeset_id=[44817130]  # normal people: 0-0.1
    lowest_blood_Procalcitonin = 0
    highest_blood_Procalcitonin = 1


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
    BMI_df = df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('Recorded_BMI', F.when(df.measurement_concept_id.isin(bmi_codeset_ids) & df.value_as_number.between(lowest_acceptable_BMI, highest_acceptable_BMI), df.value_as_number).otherwise(0)) \
        .withColumn('height', F.when(df.measurement_concept_id.isin(height_codeset_ids) & df.value_as_number.between(lowest_acceptable_height, highest_acceptable_height), df.value_as_number).otherwise(0)) \
        .withColumn('weight', F.when(df.measurement_concept_id.isin(weight_codeset_ids) & df.value_as_number.between(lowest_acceptable_weight, highest_acceptable_weight), df.value_as_number).otherwise(0))

    blood_oxygen_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('Oxygen_saturation', F.when(df.measurement_concept_id.isin(blood_oxygen_codeset_id) & df.value_as_number.between(lowest_blood_oxygen, highest_blood_oxygen), df.value_as_number).otherwise(0))

    blood_sodium_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_sodium', F.when(df.measurement_concept_id.isin(blood_sodium_codeset_id) & df.value_as_number.between(lowest_blood_sodium, highest_blood_sodium), df.value_as_number).otherwise(0))


    blood_hemoglobin_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_hemoglobin', F.when(df.measurement_concept_id.isin(blood_hemoglobin_codeset_id) & df.value_as_number.between(lowest_blood_hemoglobin, highest_blood_hemoglobin), df.value_as_number).otherwise(0))

    respiratory_rate_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('respiratory_rate', F.when(df.measurement_concept_id.isin(respiratory_rate_codeset_id) & df.value_as_number.between(lowest_respiratory_rate, highest_respiratory_rate), df.value_as_number).otherwise(0))

    blood_Creatinine_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Creatinine', F.when(df.measurement_concept_id.isin(blood_Creatinine_codeset_id) & df.value_as_number.between(lowest_blood_Creatinine, highest_blood_Creatinine), df.value_as_number).otherwise(0))

    blood_UreaNitrogen_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_UreaNitrogen', F.when(df.measurement_concept_id.isin(blood_UreaNitrogen_codeset_id) & df.value_as_number.between(lowest_blood_UreaNitrogen, highest_blood_UreaNitrogen), df.value_as_number).otherwise(0))

    blood_Potassium_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Potassium', F.when(df.measurement_concept_id.isin(blood_Potassium_codeset_id) & df.value_as_number.between(lowest_blood_Potassium, highest_blood_Potassium), df.value_as_number).otherwise(0))

    blood_Chloride_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Chloride', F.when(df.measurement_concept_id.isin(blood_Chloride_codeset_id) & df.value_as_number.between(lowest_blood_Chloride, highest_blood_Chloride), df.value_as_number).otherwise(0))

    blood_Calcium_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Calcium', F.when(df.measurement_concept_id.isin(blood_Calcium_codeset_id) & df.value_as_number.between(lowest_blood_Calcium, highest_blood_Calcium), df.value_as_number).otherwise(0))

    MCV_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('MCV', F.when(df.measurement_concept_id.isin(MCV_codeset_id) & df.value_as_number.between(lowest_MCV, highest_MCV), df.value_as_number).otherwise(0))

    Erythrocytes_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('Erythrocytes', F.when(df.measurement_concept_id.isin(Erythrocytes_codeset_id) & df.value_as_number.between(lowest_Erythrocytes, highest_Erythrocytes), df.value_as_number).otherwise(0))

    MCHC_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('MCHC', F.when(df.measurement_concept_id.isin(MCHC_codeset_id) & df.value_as_number.between(lowest_MCHC, highest_MCHC), df.value_as_number).otherwise(0))

    Systolic_blood_pressure_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('Systolic_blood_pressure', F.when(df.measurement_concept_id.isin(Systolic_blood_pressure_codeset_id) & df.value_as_number.between(lowest_Systolic_blood_pressure, highest_Systolic_blood_pressure), df.value_as_number).otherwise(0))

    Diastolic_blood_pressure_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('Diastolic_blood_pressure', F.when(df.measurement_concept_id.isin(Diastolic_blood_pressure_codeset_id) & df.value_as_number.between(lowest_Diastolic_blood_pressure, highest_Diastolic_blood_pressure), df.value_as_number).otherwise(0))

    heart_rate_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('heart_rate', F.when(df.measurement_concept_id.isin(heart_rate_codeset_id) & df.value_as_number.between(lowest_heart_rate, highest_heart_rate), df.value_as_number).otherwise(0))

    temperature_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('temperature', F.when(df.measurement_concept_id.isin(temperature_codeset_id) & df.value_as_number.between(lowest_temperature, highest_temperature), df.value_as_number).otherwise(0))

    blood_Glucose_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Glucose', F.when(df.measurement_concept_id.isin(blood_Glucose_codeset_id) & df.value_as_number.between(lowest_blood_Glucose, highest_blood_Glucose), df.value_as_number).otherwise(0))

    blood_Platelets_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Platelets', F.when(df.measurement_concept_id.isin(blood_Platelets_codeset_id) & df.value_as_number.between(lowest_blood_Platelets, highest_blood_Platelets), df.value_as_number).otherwise(0))

    blood_Hematocrit_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Hematocrit', F.when(df.measurement_concept_id.isin(blood_Hematocrit_codeset_id) & df.value_as_number.between(lowest_blood_Hematocrit, highest_blood_Hematocrit), df.value_as_number).otherwise(0))

    blood_Leukocytes_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Leukocytes', F.when(df.measurement_concept_id.isin(blood_Leukocytes_codeset_id) & df.value_as_number.between(lowest_blood_Leukocytes, highest_blood_Leukocytes), df.value_as_number).otherwise(0))

    blood_Bilirubin_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Bilirubin', F.when(df.measurement_concept_id.isin(blood_Bilirubin_codeset_id) & df.value_as_number.between(lowest_blood_Bilirubin, highest_blood_Bilirubin), df.value_as_number).otherwise(0))

    blood_Albumin_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Albumin', F.when(df.measurement_concept_id.isin(blood_Albumin_codeset_id) & df.value_as_number.between(lowest_blood_Albumin, highest_blood_Albumin), df.value_as_number).otherwise(0))

    ####
    blood_Troponin_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Troponin', F.when(df.measurement_concept_id.isin(blood_Troponin_codeset_id) & df.value_as_number.between(lowest_blood_Troponin, highest_blood_Troponin), df.value_as_number).otherwise(0))

    blood_Procalcitonin_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Procalcitonin', F.when(df.measurement_concept_id.isin(blood_Procalcitonin_codeset_id) & df.value_as_number.between(lowest_blood_Procalcitonin, highest_blood_Procalcitonin), df.value_as_number).otherwise(0))

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
    F.max('Antibody_Neg').alias('Antibody_Neg')
    )

    blood_oxygen_df = blood_oxygen_df.groupby('person_id', 'visit_date').agg(
    F.max('Oxygen_saturation').alias('Oxygen_saturation')
    )

    blood_sodium_df = blood_sodium_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_sodium').alias('blood_sodium')
    )

    blood_hemoglobin_df = blood_hemoglobin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_hemoglobin').alias('blood_hemoglobin')
    )

    respiratory_rate_df = respiratory_rate_df.groupby('person_id', 'visit_date').agg(
    F.max('respiratory_rate').alias('respiratory_rate')
    )

    blood_Creatinine_df = blood_Creatinine_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Creatinine').alias('blood_Creatinine')
    )

    blood_UreaNitrogen_df = blood_UreaNitrogen_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_UreaNitrogen').alias('blood_UreaNitrogen')
    )

    blood_Potassium_df = blood_Potassium_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Potassium').alias('blood_Potassium')
    )

    blood_Chloride_df = blood_Chloride_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Chloride').alias('blood_Chloride')
    )

    blood_Calcium_df = blood_Calcium_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Calcium').alias('blood_Calcium')
    )

    MCV_df = MCV_df.groupby('person_id', 'visit_date').agg(
    F.max('MCV').alias('MCV')
    )

    Erythrocytes_df = Erythrocytes_df.groupby('person_id', 'visit_date').agg(
    F.max('Erythrocytes').alias('Erythrocytes')
    )

    MCHC_df = MCHC_df.groupby('person_id', 'visit_date').agg(
    F.max('MCHC').alias('MCHC')
    )

    Systolic_blood_pressure_df = Systolic_blood_pressure_df.groupby('person_id', 'visit_date').agg(
    F.max('Systolic_blood_pressure').alias('Systolic_blood_pressure')
    )

    Diastolic_blood_pressure_df = Diastolic_blood_pressure_df.groupby('person_id', 'visit_date').agg(
    F.max('Diastolic_blood_pressure').alias('Diastolic_blood_pressure')
    )

    heart_rate_df = heart_rate_df.groupby('person_id', 'visit_date').agg(
    F.max('heart_rate').alias('heart_rate')
    )

    temperature_df = temperature_df.groupby('person_id', 'visit_date').agg(
    F.max('temperature').alias('temperature')
    )

    blood_Glucose_df = blood_Glucose_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Glucose').alias('blood_Glucose')
    )

    blood_Platelets_df = blood_Platelets_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Platelets').alias('blood_Platelets')
    )

    blood_Hematocrit_df = blood_Hematocrit_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Hematocrit').alias('blood_Hematocrit')
    )

    blood_Leukocytes_df = blood_Leukocytes_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Leukocytes').alias('blood_Leukocytes')
    )

    blood_Bilirubin_df = blood_Bilirubin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Bilirubin').alias('blood_Bilirubin')
    )

    blood_Albumin_df = blood_Albumin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Albumin').alias('blood_Albumin')
    )
    ###
    blood_Troponin_df = blood_Troponin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Troponin').alias('blood_Troponin')
    )

    blood_Procalcitonin_df = blood_Procalcitonin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Procalcitonin').alias('blood_Procalcitonin')
    )

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
    df = labs_df.join(BMI_df, on=['person_id', 'visit_date'], how='left').join(blood_oxygen_df, on=['person_id', 'visit_date'], how='left').join(blood_sodium_df, on=['person_id', 'visit_date'], how='left').join(blood_hemoglobin_df, on=['person_id', 'visit_date'], how='left').join(respiratory_rate_df, on=['person_id', 'visit_date'], how='left').join(blood_Creatinine_df, on=['person_id', 'visit_date'], how='left').join(blood_UreaNitrogen_df, on=['person_id', 'visit_date'], how='left').join(blood_Potassium_df, on=['person_id', 'visit_date'], how='left').join(blood_Chloride_df, on=['person_id', 'visit_date'], how='left').join(blood_Calcium_df, on=['person_id', 'visit_date'], how='left').join(MCV_df, on=['person_id', 'visit_date'], how='left').join(Erythrocytes_df, on=['person_id', 'visit_date'], how='left').join(MCHC_df, on=['person_id', 'visit_date'], how='left').join(Systolic_blood_pressure_df, on=['person_id', 'visit_date'], how='left').join(Diastolic_blood_pressure_df, on=['person_id', 'visit_date'], how='left').join(heart_rate_df, on=['person_id', 'visit_date'], how='left').join(temperature_df, on=['person_id', 'visit_date'], how='left').join(blood_Glucose_df, on=['person_id', 'visit_date'], how='left').join(blood_Platelets_df, on=['person_id', 'visit_date'], how='left').join(blood_Hematocrit_df, on=['person_id', 'visit_date'], how='left').join(blood_Leukocytes_df, on=['person_id', 'visit_date'], how='left').join(blood_Bilirubin_df, on=['person_id', 'visit_date'], how='left').join(blood_Albumin_df, on=['person_id', 'visit_date'], how='left').join(blood_Troponin_df, on=['person_id', 'visit_date'], how='left').join(blood_Procalcitonin_df, on=['person_id', 'visit_date'], how='left')





    return df

def all_patients_visit_day_facts_table_de_id(everyone_conditions_of_interest, everyone_measurements_of_interest, everyone_procedures_of_interest, everyone_observations_of_interest, everyone_drugs_of_interest,  microvisits_to_macrovisits, everyone_vaccines_of_interest, person_top_nlp_symptom):

    macrovisits_df = microvisits_to_macrovisits
    vaccines_df = everyone_vaccines_of_interest
    procedures_df = everyone_procedures_of_interest
    observations_df = everyone_observations_of_interest
    conditions_df = everyone_conditions_of_interest
    drugs_df = everyone_drugs_of_interest
    measurements_df = everyone_measurements_of_interest
    # nlp_symptom_df = Person_top_nlp_symptom \
    #     .withColumnRenamed("note_date", "visit_date") \
    #     .withColumnRenamed("Palpitations", "PALPITATIONS_NOTE") \
    #     .drop("note_id") \
    #     .drop("visit_occurrence_id")

    df = macrovisits_df.select('person_id','visit_start_date').withColumnRenamed('visit_start_date','visit_date')
    df = df.join(vaccines_df, on=list(set(df.columns)&set(vaccines_df.columns)), how='outer')
    df = df.join(procedures_df, on=list(set(df.columns)&set(procedures_df.columns)), how='outer')
    df = df.join(observations_df, on=list(set(df.columns)&set(observations_df.columns)), how='outer')
    df = df.join(conditions_df, on=list(set(df.columns)&set(conditions_df.columns)), how='outer')
    df = df.join(drugs_df, on=list(set(df.columns)&set(drugs_df.columns)), how='outer')
    df = df.join(measurements_df, on=list(set(df.columns)&set(measurements_df.columns)), how='outer')
    # df = df.join(nlp_symptom_df, on=list(set(df.columns)&set(nlp_symptom_df.columns)), how='outer')

    # df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_rounded')])

    #add F.max of all indicator columns to collapse all cross-domain flags to unique person and visit rows
    #each visit_date represents the date of the event or fact being noted in the patient's medical record
    df = df.groupby('person_id', 'visit_date').agg(*[F.max(col).alias(col) for col in df.columns if col not in ('person_id','visit_date')])

    #create and join in flag that indicates whether the visit day was during a macrovisit (1) or not (0)
    #any conditions, observations, procedures, devices, drugs, measurements, and/or death flagged
    #with a (1) on that particular visit date would then be considered to have happened during a macrovisit
    macrovisits_df = macrovisits_df \
        .select('person_id', 'macrovisit_start_date', 'macrovisit_end_date') \
        .where(F.col('macrovisit_start_date').isNotNull() & F.col('macrovisit_end_date').isNotNull()) \
        .distinct()
    df_hosp = df.select('person_id', 'visit_date').join(macrovisits_df, on=['person_id'], how= 'outer')
    df_hosp = df_hosp.withColumn('during_macrovisit_hospitalization', F.when((F.datediff("macrovisit_end_date","visit_date")>=0) & (F.datediff("macrovisit_start_date","visit_date")<=0), 1).otherwise(0)) \
        .drop('macrovisit_start_date', 'macrovisit_end_date') \
        .where(F.col('during_macrovisit_hospitalization') == 1) \
        .distinct()
    df = df.join(df_hosp, on=['person_id','visit_date'], how="left")   

    # final fill of null in non-continuous variables with 0
    df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_rounded')])


    return df


def everyone_vaccines_of_interest(everyone_cohort_de_id, Vaccine_fact_de_identified, first_covid_positive):
    vaccine_fact_de_identified = Vaccine_fact_de_identified

    persons = everyone_cohort_de_id.select('person_id')
    vax_df = Vaccine_fact_de_identified.select('person_id', '1_vax_date', '2_vax_date', '3_vax_date', '4_vax_date') \
        .join(persons, 'person_id', 'inner')

    vax_switch = Vaccine_fact_de_identified.select('person_id', '1_vax_type', 'date_diff_1_2') \
        .withColumnRenamed('date_diff_1_2', 'DATE_DIFF_1_2') \
        .withColumn("1_VAX_JJ", F.when(F.col('1_vax_type') == 'janssen', 1).otherwise(0)) \
        .withColumn("1_VAX_PFIZER", F.when(F.col('1_vax_type') == 'pfizer', 1).otherwise(0)) \
        .withColumn("1_VAX_MODERNA", F.when(F.col('1_vax_type') == 'moderna', 1).otherwise(0)) \
        .drop(F.col('1_vax_type'))

    first_dose = vax_df.select('person_id', '1_vax_date') \
        .withColumnRenamed('1_vax_date', 'visit_date') \
        .withColumn('1_vax_dose', F.lit(1)) \
        .where(F.col('visit_date').isNotNull())
    second_dose = vax_df.select('person_id', '2_vax_date') \
        .withColumnRenamed('2_vax_date', 'visit_date') \
        .withColumn('2_vax_dose', F.lit(1)) \
        .where(F.col('visit_date').isNotNull())
    third_dose = vax_df.select('person_id', '3_vax_date') \
        .withColumnRenamed('3_vax_date', 'visit_date') \
        .withColumn('3_vax_dose', F.lit(1)) \
        .where(F.col('visit_date').isNotNull())
    fourth_dose = vax_df.select('person_id', '4_vax_date') \
        .withColumnRenamed('4_vax_date', 'visit_date') \
        .withColumn('4_vax_dose', F.lit(1)) \
        .where(F.col('visit_date').isNotNull())

    df = first_dose.join(second_dose, on=['person_id', 'visit_date'], how='outer') \
        .join(third_dose, on=['person_id', 'visit_date'], how='outer') \
        .join(fourth_dose, on=['person_id', 'visit_date'], how='outer') \
        .join(vax_switch, on=['person_id'], how='inner') \
        .distinct()

    df = df.withColumn('had_vaccine_administered', F.lit(1)) \
        .join(first_covid_positive, 'person_id', 'leftouter') \
        .withColumn('vax_before_FCP', F.when(F.datediff(F.col('visit_date'), F.col('first_covid_positive')) < 0, 1).otherwise(0)) \
        .withColumn('VAX_DAYS_SINCE_FCP', F.datediff(F.col('visit_date'), F.col('first_covid_positive'))) \
        .drop(F.col('first_covid_positive'))

    return df

def first_covid_positive(everyone_conditions_of_interest):
    w = Window.partitionBy('person_id').orderBy(F.asc('visit_date'))
    df = (everyone_conditions_of_interest \
        # .filter(F.col('LL_COVID_diagnosis') == 1) \
        .select('person_id', F.first('visit_date').over(w).alias('first_covid_positive')) \
        .distinct())

    return df


# ri.vector.main.execute.7641dae2-3118-4a2c-8a89-e4f646cbf18f
def Vaccine_fact_de_identified(aggregate_person, sql_pivot_vax_person):
    """
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
WHERE vaccine_txn < 5"""
    # Join the pivoted vaccine data with the aggregate person data
    Vaccine_fact_de_identified = sql_pivot_vax_person.join(aggregate_person, 'person_id', 'leftouter') \
        .where(F.col('vaccine_txn') < 5) \
        .distinct() \
        .withColumn('date_1_vax', F.datediff(F.col('1_vax_date'), F.lit('2020-01-01'))) \
        .withColumn('date_diff_1_2', F.datediff(F.col('2_vax_date'), F.col('1_vax_date'))) \
        .withColumn('date_diff_2_3', F.datediff(F.col('3_vax_date'), F.col('2_vax_date'))) \
        .withColumn('date_diff_3_4', F.datediff(F.col('4_vax_date'), F.col('3_vax_date'))) \
        .withColumn('switch_1_2', F.when((F.col('1_vax_type') != F.col('2_vax_type')) & (F.col('1_vax_date') != F.col('2_vax_date')), 1).otherwise(0)) \
        .withColumn('switch_2_3', F.when((F.col('2_vax_type') != F.col('3_vax_type')) & (F.col('2_vax_date') != F.col('3_vax_date')), 1).otherwise(0)) \
        .withColumn('switch_3_4', F.when((F.col('3_vax_type') != F.col('4_vax_type')) & (F.col('3_vax_date') != F.col('4_vax_date')), 1).otherwise(0)) \
        .select(
            sql_pivot_vax_person['person_id'],
            F.col('vaccine_txn'),
            'date_1_vax',
            'date_diff_1_2',
            'date_diff_2_3',
            'date_diff_3_4',
            'switch_1_2',
            'switch_2_3',
            'switch_3_4',
            '1_vax_date',
            '1_vax_type',
            '2_vax_date',
            '2_vax_type',
            '3_vax_date',
            '3_vax_type',
            '4_vax_date',
            '4_vax_type')

    return Vaccine_fact_de_identified

# ri.vector.main.execute.0c8d244b-83b0-4a73-9488-3db78097ac5a
def aggregate_person(deduplicated):
    """
SELECT person_id, data_partner_id, count(vax_date) as vaccine_txn
FROM deduplicated
group by person_id, data_partner_id"""
    # Aggregate the number of vaccine transactions per person
    aggregate_person = deduplicated.groupBy(
        'person_id'
    ).agg(
        F.count('vax_date').alias('vaccine_txn')
    )[[ 'person_id', 'vaccine_txn']]

    return aggregate_person


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
        F.count('person_id').over(w).alias('n')
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
        'lag_date', F.lag('vax_date', default='2000-01-01').over(w)
    ).withColumn(
        'date_diff', F.datediff('vax_date', 'lag_date')
    )

    # For site 406, filter if less than 14. For everyone else, filter if less than 5
    df = df.filter(df.date_diff >= 5)

    return df

# ri.vector.main.execute.8a3be0e3-a478-40ab-83b1-7289e3fc5136
def distinct_vax_person(baseline_vaccines, baseline_vaccines_from_proc_testing):
    """
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
FROM baseline_vaccines_from_proc"""
    # Create minimal long dataset of vaccine transactions to transpose
    distinct_vax_person = baseline_vaccines.select(
        'person_id', 'drug_exposure_start_date', 'vax_type'
    ).withColumnRenamed(
        'drug_exposure_start_date', 'vax_date'
    ).union(
        baseline_vaccines_from_proc_testing.select(
            'person_id', 'procedure_date', 'vax_type'
        ).withColumnRenamed(
            'procedure_date', 'vax_date'
        )
    ).distinct()

    return distinct_vax_person

# ri.vector.main.execute.20bfed79-2fea-446f-a920-88f0dbc22bc2
def baseline_vaccines(custom_concept_set_members, drug_exposure):
    """
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
    )"""
    # Create minimal long dataset of vaccine transactions to transpose
    baseline_vaccines = custom_concept_set_members.select(
        'concept_id'
    ).withColumnRenamed(
        'concept_id', 'drug_concept_id'
    ).join(
        drug_exposure, on='drug_concept_id', how='inner'
    # ).filter(
    #     F.col('codeset_id') == 600531961
    ).filter(
        F.col('drug_exposure_start_date').isNotNull()
    ).filter(
        F.col('drug_exposure_start_date') > '2018-12-20'
    # ).filter(
    #     F.col('drug_exposure_start_date') < (F.col('run_date') + 356*2)
    ).filter(
        (
            (~F.col('drug_type_concept_id').isin([38000177, 32833, 38000175, 32838, 32839])) #| (
                # (F.col('drug_type_concept_id') == 38000177) & (F.col('cdm_name') == 'ACT')
        )
    ).select(
        'person_id', 'drug_exposure_start_date', 'drug_concept_id'
    ).withColumn(
        'vax_type', F.when(
            F.col('drug_concept_id').isin([702677, 702678, 724907, 37003432, 37003435, 37003436]), 'pfizer'
        ).when(
            F.col('drug_concept_id').isin([724906, 37003518]), 'moderna'
        ).when(
            F.col('drug_concept_id').isin([702866, 739906]), 'janssen'
        ).when(
            F.col('drug_concept_id').isin([724905]), 'astrazeneca'
        ).otherwise(
            F.lit(None)
        )
    )

    return baseline_vaccines

# ri.vector.main.execute.41c51a4c-2331-41c3-acad-6b3c1be58064
def baseline_vaccines_from_proc_testing(procedure_occurrence):
    """
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
    and procedure_date is not null"""
    # Create minimal long dataset of vaccine transactions to transpose
    baseline_vaccines_from_proc = procedure_occurrence.filter(
        F.col('procedure_concept_id').isin([766238, 766239, 766241])
    ).filter(
        F.col('procedure_date').isNotNull()
    ).filter(
        F.col('procedure_date') > '2018-12-20'
    ).withColumn(
        'vax_type', F.when(
            F.col('procedure_concept_id') == 766238, 'pfizer'
        ).when(
            F.col('procedure_concept_id') == 766239, 'moderna'
        ).when(
            F.col('procedure_concept_id') == 766241, 'janssen'
        ).otherwise(
            F.lit(None)
        )
    ).select(
        'person_id', 'procedure_date', 'vax_type'
    )

    return baseline_vaccines_from_proc

# ri.vector.main.execute.c2687a32-aea0-4394-ae9d-b488d148563e
def sql_pivot_vax_person(deduplicated):
    """
select * from (
    select person_id, data_partner_id,
    row_number() over (partition by person_id order by person_id, vax_date) as number, vax_type, vax_date
    from deduplicated
) A

pivot (
    max(vax_date) as vax_date, max(vax_type) as vax_type
    for number in (1,2,3,4)
)"""
    # Pivot the long dataset to wide
    pivot_vax_person = deduplicated.select(
        'person_id', 'vax_date', 'vax_type'
    ).withColumn(
        'number', F.row_number().over(
            Window.partitionBy('person_id').orderBy('person_id', 'vax_date')
        )
    ).groupBy(
        'person_id'
    ).pivot(
        'number', [1, 2, 3, 4]
    ).agg(
        F.max('vax_date').alias('vax_date'), F.max('vax_type').alias('vax_type')
    )

    return pivot_vax_person

def all_patients_summary_fact_table_de_id(all_patients_visit_day_facts_table_de_id, everyone_cohort_de_id):

    #deaths_df = everyone_patient_deaths.select('person_id','patient_death')
    df = all_patients_visit_day_facts_table_de_id.drop('patient_death_at_visit', 'during_macrovisit_hospitalization')
    
    df2 = all_patients_visit_day_facts_table_de_id.select('person_id', 'visit_date', 'Oxygen_saturation').where(all_patients_visit_day_facts_table_de_id.Oxygen_saturation>0)    
    
    df3 = all_patients_visit_day_facts_table_de_id.select('person_id', 'visit_date', 'blood_sodium').where(all_patients_visit_day_facts_table_de_id.blood_sodium>0) 
    
    df4 = all_patients_visit_day_facts_table_de_id.select('person_id', 'visit_date', 'blood_hemoglobin').where(all_patients_visit_day_facts_table_de_id.blood_hemoglobin>0) 

    df5 = all_patients_visit_day_facts_table_de_id.select('person_id', 'visit_date', 'blood_Creatinine').where(all_patients_visit_day_facts_table_de_id.blood_Creatinine>0)

    df6 = all_patients_visit_day_facts_table_de_id.select('person_id', 'visit_date', 'blood_UreaNitrogen').where(all_patients_visit_day_facts_table_de_id.blood_UreaNitrogen>0)

    df = df.groupby('person_id').agg(
        F.max('BMI_rounded').alias('BMI_max_observed_or_calculated'),
        F.avg('respiratory_rate').alias('respiratory_rate'),
        *[F.max(col).alias(col + '_indicator') for col in df.columns if col not in ('person_id', 'BMI_rounded', 'visit_date', 'had_vaccine_administered', 'Oxygen_saturation', 'blood_sodium', 'blood_hemoglobin', 'respiratory_rate', 'blood_Creatinine', 'blood_UreaNitrogen')],
        F.sum('had_vaccine_administered').alias('total_number_of_COVID_vaccine_doses'))
    
    # df2 = df2.groupby('person_id').agg(
    #     F.min('Oxygen_saturation').alias('min_Oxygen_saturation'))
    # df3 = df3.groupby('person_id').agg(
    #     F.last('blood_sodium').alias('last_blood_sodium'))
    # df4 = df4.groupby('person_id').agg(
    #     F.last('blood_hemoglobin').alias('last_blood_hemoglobin'))
    # df5 = df5.groupby('person_id').agg(
    #     F.last('blood_Creatinine').alias('last_blood_Creatinine'))
    # df6 = df6.groupby('person_id').agg(
    #     F.last('blood_UreaNitrogen').alias('last_blood_UreaNitrogen'))
    
    # meas_df = df2.join(df3, on=['person_id'], how='left').join(df4, on=['person_id'], how='left').join(df5, on=['person_id'], how='left').join(df6, on=['person_id'], how='left')
    
    # df = df.join(meas_df, on=['person_id'], how='left')

    df = everyone_cohort_de_id.join(df, 'person_id','left')

    #final fill of null in non-continuous variables with 0
    # df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_max_observed_or_calculated', 'postal_code', 'age')])
    
    df = df.distinct()


    return df

def get_time_series_data(data_tables: dict, concept_tables:dict):
    person_table = data_tables["person"]
    measurement_table = data_tables["measurement"]
    drug_exposure_table = data_tables["drug_exposure"]
    procedure_occurrence_table = data_tables["procedure_occurrence"]
    observation_table = data_tables["observation"]
    micro_to_macro_table = data_tables["microvisits_to_macrovisits"]

    concept_set_members_table = concept_tables["concept_set_members"]
    condition_occurrence_table = data_tables["condition_occurrence"]
    LL_concept_sets_fusion_everyone_table = concept_tables["LL_concept_sets_fusion_everyone"]
    LL_DO_NOT_DELETE_REQUIRED_concept_sets_all_table = concept_tables["LL_DO_NOT_DELETE_REQUIRED_concept_sets_all"]

    custom_sets_table = pandas_to_spark(custom_sets(LL_concept_sets_fusion_everyone_table))
    customized_concept_set_input_table = customized_concept_set_input(LL_DO_NOT_DELETE_REQUIRED_concept_sets_all_table, custom_sets_table)
    custom_concept_set_members_table = custom_concept_set_members(concept_set_members_table)
    everyone_cohort_de_id_table = everyone_cohort_de_id(person_table, micro_to_macro_table, custom_concept_set_members_table)

    baseline_vaccines_from_proc_testing_table = baseline_vaccines_from_proc_testing(procedure_occurrence_table)
    baseline_vaccines_table = baseline_vaccines(custom_concept_set_members_table, drug_exposure_table)
    distinct_vax_person_table = distinct_vax_person(baseline_vaccines_table, baseline_vaccines_from_proc_testing_table)
    deduplicated_table = deduplicated(distinct_vax_person_table)
    aggregate_person_table = aggregate_person(deduplicated_table)
    sql_pivot_vax_person_table = sql_pivot_vax_person(deduplicated_table)
    Vaccine_fact_de_identified_table = Vaccine_fact_de_identified(aggregate_person_table, sql_pivot_vax_person_table)
    everyone_conditions_of_interest_table = everyone_conditions_of_interest(everyone_cohort_de_id_table, condition_occurrence_table, customized_concept_set_input_table, custom_concept_set_members_table)
    first_covid_positive_table = first_covid_positive(everyone_conditions_of_interest_table)

    everyone_conditions_of_interest_table = everyone_conditions_of_interest(everyone_cohort_de_id_table, condition_occurrence_table, customized_concept_set_input_table, custom_concept_set_members_table)
    everyone_observations_of_interest_table = everyone_observations_of_interest(observation_table,everyone_cohort_de_id_table, customized_concept_set_input_table, custom_concept_set_members_table)
    everyone_procedures_of_interest_table = everyone_procedures_of_interest(everyone_cohort_de_id_table, procedure_occurrence_table, customized_concept_set_input_table, custom_concept_set_members_table)
    everyone_drugs_of_interest_table = everyone_drugs_of_interest(drug_exposure_table, everyone_cohort_de_id_table, customized_concept_set_input_table, custom_concept_set_members_table)
    everyone_measurements_of_interest_table = everyone_measurements_of_interest(measurement_table, everyone_cohort_de_id_table, custom_concept_set_members_table)
    everyone_vaccines_of_interest_table = everyone_vaccines_of_interest(everyone_cohort_de_id_table, Vaccine_fact_de_identified_table, first_covid_positive_table)

    all_patients_visit_day_facts_table_de_id_table = all_patients_visit_day_facts_table_de_id(everyone_conditions_of_interest_table, everyone_measurements_of_interest_table, everyone_procedures_of_interest_table, everyone_observations_of_interest_table, everyone_drugs_of_interest_table, micro_to_macro_table, everyone_vaccines_of_interest_table, None)
    return everyone_cohort_de_id_table, all_patients_visit_day_facts_table_de_id_table

def get_static_from_time_series(everyone_cohort_de_id_table, all_patients_visit_day_facts_table_de_id_table):
    return all_patients_summary_fact_table_de_id(all_patients_visit_day_facts_table_de_id_table, everyone_cohort_de_id_table)

def get_top_k_data(everyone_cohort_de_id_table, data_tables):
    tables = {"procedure_occurrence":("procedure_concept_id",1000),"condition_occurrence":("condition_concept_id",1000), "drug_exposure":("drug_concept_id",1000), "observation":("observation_concept_id",1000), "measurement":("measurement_concept_id",1000)}
    feats = everyone_cohort_de_id_table.select(F.col("person_id"))
    for TABLE, (CONCEPT_ID_COL,k) in tables.items():
        TABLE = data_tables[TABLE].select(F.col("person_id"), F.col(CONCEPT_ID_COL)).withColumn(CONCEPT_ID_COL, F.regexp_replace(F.col(CONCEPT_ID_COL).cast(StringType()), ".", ""))               
        distinct = TABLE.groupBy(CONCEPT_ID_COL).count().orderBy("count", ascending=False).limit(k).select(F.col(CONCEPT_ID_COL)).toPandas()[CONCEPT_ID_COL].tolist()
        df = TABLE.filter(F.col(CONCEPT_ID_COL).isin(distinct))
        df= df.groupBy("person_id").pivot(CONCEPT_ID_COL).agg(F.lit(1)).na.fill(0)
        df = df.select([F.col(c).alias(CONCEPT_ID_COL[:3]+str(c)) if c != "person_id" else c for c in df.columns ])
        feats = feats.join(df, how="left",on="person_id")
    data = feats.na.fill(0)
    
    return data