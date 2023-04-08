from data_preparation_2 import *
from src.featurize import *
from src.sklearn_models import *
from src.mTan import train_sequential_model_3


import random

def train_valid_split( Long_COVID_Silver_Standard, num_recent_visits):
    all_person_ids_df = num_recent_visits[["person_id"]].join(Long_COVID_Silver_Standard, on="person_id", how="left")
    all_person_ids = list(all_person_ids_df.toPandas()["person_id"])

    train_valid_split = 9, 1
    train_ratio = train_valid_split[0] / (train_valid_split[0] + train_valid_split[1])
    num_train = int(train_ratio * len(all_person_ids))

    random.shuffle(all_person_ids)
    train_person_ids, valid_person_ids = all_person_ids[:num_train], all_person_ids[num_train:]
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    np_ls = [[person_id, "train"] for person_id in train_person_ids] + [[person_id, "valid"] for person_id in valid_person_ids]

    split_person_ids_df = spark.createDataFrame(np_ls, schema=["person_id", "split"]).sort("person_id")


    # split_person_ids_df = pd.DataFrame([[person_id, "train"] for person_id in train_person_ids] + [[person_id, "valid"] for person_id in valid_person_ids], columns=["person_id", "split"]).sort_values("person_id")

    return split_person_ids_df


def recent_visits_2(all_patients_visit_day_facts_table_de_id):
    # First sort the visits
    # all_patients_visit_day_facts_table_de_id = all_patients_visit_day_facts_table_de_id.sort_values(["person_id", "visit_date"])

    all_patients_visit_day_facts_table_de_id = all_patients_visit_day_facts_table_de_id.orderBy(all_patients_visit_day_facts_table_de_id.person_id, all_patients_visit_day_facts_table_de_id.visit_date)

    # Get the number of visits
    # num_visits = all_patients_visit_day_facts_table_de_id \
    #     .groupby('person_id')['visit_date'] \
    #     .nunique() \
    #     .reset_index() \
    #     .rename(columns={"visit_date": "num_visits"})

    # The maximum number of visits is around 1000
    # print(num_visits.max())

    # Get the last visit of each patient
    # print(all_patients_visit_day_facts_table_de_id.groupBy("person_id").agg(max_("visit_date")).show())
    last_visit = all_patients_visit_day_facts_table_de_id \
        .groupBy("person_id") \
        .agg(max_("visit_date")).withColumnRenamed("max(visit_date)", "last_visit_date") \
        # .reset_index("person_id") \
        # .rename(columns={"visit_date": "last_visit_date"})
    
    # Add a six-month before the last visit column to the dataframe
    # last_visit["six_month_before_last_visit"] = last_visit["last_visit_date"].map(lambda x: x - pd.Timedelta(days=180))
    last_visit = last_visit.withColumn("six_month_before_last_visit", F.date_sub(last_visit["last_visit_date"], 180))
    # print(last_visit.show())

    # Merge last_visit back
    # df = all_patients_visit_day_facts_table_de_id.merge(last_visit, on="person_id", how="left")
    df = all_patients_visit_day_facts_table_de_id.join(last_visit, on = "person_id", how = "left")

    # Find "recent visits" for each patient that are within six-month before their final visit
    # mask = df["visit_date"] > df["six_month_before_last_visit"]
    # print(mask)
    # recent_visits = df.loc[mask]


    recent_visits = df.where(datediff(df["visit_date"], df["six_month_before_last_visit"]) > 0)
    print(datediff(df["visit_date"], df["six_month_before_last_visit"]) > 0)
    # spark = SparkSession.builder.getOrCreate()
    # recent_visits = spark.createDataFrame(recent_visits)
    print("recent visit::", recent_visits.show())
    print(recent_visits)
    # Add diff_date feature: how many days have passed from the previous visit?
    # recent_visits["diff_date"] = recent_visits.groupby("person_id")["visit_date"].diff().fillna(0).map(lambda x: x if type(x) == int else x.days)
    min_person_visit_data = recent_visits.groupBy("person_id").agg(min_("visit_date"))
    print("here")

    # min_person_visit_data = min_person_visit_data.reset_index()
    min_person_visit_data = min_person_visit_data.withColumnRenamed("min(visit_date)", "min_visit_date")
    # min_person_visit_data.rename(columns={"min":"min_visit_date"}, inplace="True")

    # recent_visits = recent_visits.merge(min_person_visit_data)
    recent_visits = recent_visits.join(min_person_visit_data, on = "person_id")
    
    # recent_visits["diff_days"] = (recent_visits["visit_date"] - recent_visits["min_visit_date"]).dt.days.astype('int16')
    recent_visits = recent_visits.withColumn("diff_days", datediff(recent_visits["visit_date"], recent_visits["min_visit_date"]))
    print(recent_visits.show())
    
    # Rearrange columns
    cols = recent_visits.columns
    print(cols)
    cols = cols[0:2] + cols[-3:] + cols[2:-4]
    recent_visits = recent_visits.select(cols)

    # The maximum difference is 179
    # max_diff_date = recent_visits["diff_date"].max()
    # print(max_diff_date)

    # Make sure the data is sorted
    recent_visits = recent_visits.orderBy(recent_visits.person_id, recent_visits.visit_date)#.fillna(0)

    return recent_visits

def num_recent_visits(recent_visits_2):

    # Find how many recent visits are there
    # df = recent_visits_2 \
    #     .groupby('person_id')['visit_date'] \
    #     .nunique() \
    #     .reset_index() \
    #     .rename(columns={"visit_date": "num_recent_visits"})
    df = recent_visits_2.groupBy("person_id").agg(F.countDistinct("visit_date").alias("num_recent_visits")).select("person_id", "num_recent_visits").orderBy("person_id")

    return df

def recent_visits_w_nlp_notes_2(recent_visits_2, person_nlp_symptom):
    person_nlp_symptom = person_nlp_symptom
    person_nlp_symptom = person_nlp_symptom.join(recent_visits_2.select(["person_id", "six_month_before_last_visit"]).distinct(), on="person_id", how="left")
    person_nlp_symptom = person_nlp_symptom.where(person_nlp_symptom["note_date"] >= person_nlp_symptom["six_month_before_last_visit"])
    person_nlp_symptom = person_nlp_symptom.withColumnRenamed("note_date", "visit_date").drop(*["six_month_before_last_visit", "note_id", "visit_occurrence_id"])
    person_nlp_symptom = person_nlp_symptom.withColumn("has_nlp_note", lit(1.0))
    df = recent_visits_2.join(person_nlp_symptom, on = ["person_id", "visit_date"], how="left").orderBy(*["person_id", "visit_date"])#.fillna(0.0)


    # person_nlp_symptom = person_nlp_symptom.merge(recent_visits_2[["person_id", "six_month_before_last_visit"]].drop_duplicates(), on="person_id", how="left")
    # person_nlp_symptom = person_nlp_symptom.loc[person_nlp_symptom["note_date"] >= person_nlp_symptom["six_month_before_last_visit"]]
    # person_nlp_symptom = person_nlp_symptom.rename(columns={"note_date": "visit_date"}).drop(columns=["six_month_before_last_visit", "note_id", "visit_occurrence_id"])
    # person_nlp_symptom["has_nlp_note"] = 1.0

    # # Make sure type checks
    # df = recent_visits_2.merge(person_nlp_symptom, on=["person_id", "visit_date"], how="left").sort_values(["person_id", "visit_date"])#.fillna(0.0)

    return df

def person_information(everyone_cohort_de_id):
    df = everyone_cohort_de_id

    df_pandas = df.toPandas()
    # First add normalized age
    min_age = df_pandas["age"].min()
    max_age = df_pandas["age"].max()
    diff = max_age - min_age
    # df["normalized_age"] = df["age"].map(lambda a: (a - min_age) / diff).fillna(0.0)
    df = df.withColumn("normalized_age", (df["age"] - F.lit(float(min_age))) / F.lit(float(diff))).fillna(0.0)

    # Then add gender information
    # df["is_male"] = df["gender_concept_name"].map(lambda g: 1 if g == "MALE" else 0)
    # df = df.withColumn("is_male", F.when(F.col("gender_concept_name") == "MALE", 1).otherwise(0))
    # # df["is_female"] = df["gender_concept_name"].map(lambda g: 1 if g == "FEMALE" else 0)
    # df = df.withColumn("is_female", F.when(F.col("gender_concept_name") == "FEMALE", 1).otherwise(0))
    # # df["is_other_gender"] = df["gender_concept_name"].map(lambda g: 1 if g != "FEMALE" and g != "MALE" else 0)
    # df = df.withColumn("is_other_gender", F.when(F.col("gender_concept_name") != "FEMALE" & F.col("gender_concept_name") != "MALE", F.lit(1)).otherwise(F.lit(0)))

    # Only include necessary feature
    # df = df[["person_id", "age", "normalized_age", "is_male", "is_female", "is_other_gender"]]
    df = df.select("person_id", "age", "normalized_age")#, "is_male", "is_female", "is_other_gender")

    # Return
    return df

def select_subset_data(data_tables):
    new_data_tables = dict()
    for key in data_tables:
        new_data_tables[key] = data_tables[key].sort("person_id").limit(5000)
        
    return new_data_tables

def train_main():
    data_tables = get_training_data()
    concept_tables = get_concept_data()
    print("Beginning training data featurization...")
    data_tables = select_subset_data(data_tables)
    everyone_cohort_de_id_table, all_patients_visit_day_facts_table_de_id_table = get_time_series_data(data_tables, concept_tables)

    person_information_table = person_information(everyone_cohort_de_id_table)
    # static_data_table = get_static_from_time_series(everyone_cohort_de_id_table, all_patients_visit_day_facts_table_de_id_table)
    # top_k_data_table = get_top_k_data(everyone_cohort_de_id_table, data_tables)
    # print("Finished training data featurization!")
    # print("Beginning top-k training...")
    # train_top_k_models(top_k_data_table, data_tables["long_covid_silver_standard"])
    # print("Finished topk training!")
    # print("Beginning static training...")
    # train_static_models(static_data_table, data_tables["long_covid_silver_standard"])
    # print("Finished static training!")

    recent_visits_2_data = recent_visits_2(all_patients_visit_day_facts_table_de_id_table)

    num_recent_visits_data = num_recent_visits(recent_visits_2_data)

    train_valid_split_data = train_valid_split(data_tables["long_covid_silver_standard"], num_recent_visits_data)

    # recent_visits_w_nlp_notes_2_data = recent_visits_w_nlp_notes_2(recent_visits_2_data, person_nlp_symptom)

    rec, dec, classifier = train_sequential_model_3(train_valid_split_data, data_tables["long_covid_silver_standard"], person_information_table, recent_visits_2_data)

def test_main():
    data_tables = get_testing_data()
    concept_tables = get_concept_data()
    print("Beginning training data featurization...")
    everyone_cohort_de_id_table, all_patients_visit_day_facts_table_de_id_table = get_time_series_data(data_tables, concept_tables)    
    
    
    

if __name__ == "__main__":
    train_main()