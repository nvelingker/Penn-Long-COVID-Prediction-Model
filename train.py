from data_preparation import *
from src.featurize import *
from src.sklearn_models import *
from src.mTan import train_sequential_model_3
from src.utils import *

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











def train_main():
    data_tables = get_training_data()
    concept_tables = get_concept_data()
    print("Beginning training data featurization...")
    # data_tables = select_subset_data(data_tables)
    everyone_cohort_de_id_table, all_patients_visit_day_facts_table_de_id_table = get_time_series_data(data_tables, concept_tables)

    person_information_table = person_information(everyone_cohort_de_id_table)
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

    train_valid_split_data = train_valid_split(data_tables["long_covid_silver_standard"], num_recent_visits_data)

    # recent_visits_w_nlp_notes_2_data = recent_visits_w_nlp_notes_2(recent_visits_2_data, person_nlp_symptom)

    train_sequential_model_3(train_valid_split_data, data_tables["long_covid_silver_standard"], person_information_table, recent_visits_2_data)

    

if __name__ == "__main__":
    train_main()