from pyspark.sql import SparkSession
import os

PERSON_TABLES = ["condition_occurrence", "death", "drug_exposure", "long_covid_silver_standard", "measurement", "microvisits_to_macrovisits", "observation_period", "observation", "person", "procedure_occurrence", "visit_occurrence"]
CONCEPT_TABLES = ["concept","concept_set_members","LL_concept_sets_fusion_everyone","LL_DO_NOT_DELETE_REQUIRED_concept_sets_all"]
SPARK = SparkSession.builder.appName('local[1]').config("spark.driver.memory", "500g").getOrCreate()
def __get_data(path, tables):
    return {table: SPARK.read.csv(path + table + ".csv", sep=',',
                         inferSchema=True, header=True) for table in tables}

def get_training_data():
    return __get_data(os.getcwd() + "/Penn-Long-COVID-Prediction-Model/synthetic_data/training/", PERSON_TABLES)

def get_testing_data():
    return __get_data(os.getcwd() + "/Penn-Long-COVID-Prediction-Model/synthetic_data/testing/", PERSON_TABLES)

def get_concept_data():
    return __get_data(os.getcwd() + "/Penn-Long-COVID-Prediction-Model/synthetic_data/", CONCEPT_TABLES)

if __name__ == "__main__":
    #for testing purposes
    get_training_data()
    get_testing_data()
    get_concept_data()
