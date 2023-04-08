from data_preparation import *
from src.featurize import *
from src.sklearn_models import *


if __name__ == "__main__":
    data_tables = get_testing_data()
    concept_tables = get_concept_data()
    # data_tables = select_subset_data(data_tables)
    print("Beginning testing data featurization...")
    everyone_cohort_de_id_table, all_patients_visit_day_facts_table_de_id_table = get_time_series_data(data_tables, concept_tables)
    static_data_table = get_static_from_time_series(everyone_cohort_de_id_table, all_patients_visit_day_facts_table_de_id_table)
    top_k_data_table = get_top_k_data(everyone_cohort_de_id_table, data_tables)
    print("Finished testing data featurization!")
    print("Running model predictions...")
    sklearn_models_predict(top_k_data_table, static_data_table, everyone_cohort_de_id_table, all_patients_visit_day_facts_table_de_id_table)
    print("Finished running model predictions!")
    print("Predictions dumped to predictions.csv")