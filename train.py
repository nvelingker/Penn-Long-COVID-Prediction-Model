from data_preparation import *
from src.featurize import *
from src.sklearn_models import *


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