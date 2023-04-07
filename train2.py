from data_preparation import *
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

if __name__ == "__main__":
    data_tables = get_training_data()
    concept_tables = get_concept_data()
    print("Beginning training data featurization...")
    everyone_cohort_de_id_table, all_patients_visit_day_facts_table_de_id_table = get_time_series_data(data_tables, concept_tables)
    static_data_table = get_static_from_time_series(everyone_cohort_de_id_table, all_patients_visit_day_facts_table_de_id_table)
    top_k_data_table = get_top_k_data(everyone_cohort_de_id_table, data_tables)
    print("Finished training data featurization!")
    print("Beginning top-k training...")
    train_top_k_models(top_k_data_table, data_tables["long_covid_silver_standard"])
    print("Finished topk training!")
    print("Beginning static training...")
    train_static_models(static_data_table, data_tables["long_covid_silver_standard"])
    print("Finished static training!")

    recent_visits_2_data = recent_visits_2(all_patients_visit_day_facts_table_de_id_table)

    num_recent_visits_data = num_recent_visits(recent_visits_2_data)

    train_valid_split_data = train_valid_split( data_tables["long_covid_silver_standard"], num_recent_visits_data)

    

    train_sequential_model_3(train_valid_split_data, data_tables["long_covid_silver_standard"], data_tables["person_info"], recent_visits_w_nlp_notes_2)