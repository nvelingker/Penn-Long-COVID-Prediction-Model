from data_preparation import *
from src.featurize import *

#for testing purposes
if __name__ == "__main__":
    data_tables = get_testing_data()
    concept_tables = get_concept_data()
    get_time_series_data(data_tables, concept_tables)