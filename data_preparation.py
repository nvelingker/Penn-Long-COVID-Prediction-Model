import pandas as pd

__TABLES = ["condition_occurrence", "death", "drug_exposure", "measurement", "observation_period", "observation", "person", "procedure_occurrence", "visit_occurrence"]

def __get_data(path):
    return {table: pd.read_csv(path + table + ".csv") for table in __TABLES}

def get_training_data():
    return __get_data("./synthetic_data/training/")

def get_testing_data():
    return __get_data("./synthetic_data/testing/")

if __name__ == "__main__":
    #for testing purposes
    get_training_data()
    get_testing_data()
