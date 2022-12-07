from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql import Window
from pyspark.sql import SparkSession
import pandas as pd
import sklearn
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import numpy as np
from matplotlib import pyplot as plt
import time
# import torch

# # Set default dtype
# torch.set_default_dtype(torch.float32)

# # Evaluation Label Setups
# LABEL_SETUPS = [
#     "prior", # Only take `pasc_code_prior_four_weeks` as label
#     "after", # Only take `pasc_code_after_four_weeks` as label
#     "both", # Take the `or` of both `pasc_code_prior_four_weeks` and `pasc_code_after_four_weeks` as label
# ]

# # Visits Dataset for sequential
# class LongCOVIDVisitsDataset(torch.utils.data.Dataset):
#     def __init__(self, person_id, person_info, recent_visit, label, setup="both"):
#         self.person_ids = person_id
#         self.person_info = person_info.set_index("person_id")
#         self.recent_visit = recent_visit.set_index(["person_id", "visit_date"])
#         self.label = label.set_index("person_id")
#         self.setup = setup

#         assert len(self.person_ids) == len(self.label)
#         assert len(self.person_info) == len(self.label)

#     def __len__(self):
#         return len(self.person_info)

#     def __getitem__(self, idx):
#         person_id = self.person_ids.iloc[idx]["person_id"]

#         # Encode person_info into vector
#         person_info = self.person_info.loc[person_id]
#         person_info_tensor = torch.tensor([
#             person_info["normalized_age"], 
#             person_info["is_male"], 
#             person_info["is_female"], 
#             person_info["is_other_gender"]
#         ])

#         # Encode each visit into vector
#         visits = self.recent_visit.loc[person_id]
#         visit_tensors = []
#         for i in range(len(visits)):
#             visit = visits.iloc[i]
#             visit_tensor = torch.tensor([visit["diff_date"] / 180] + list(visit[5:]))
#             visit_tensors.append(visit_tensor)
#         visits_tensor = torch.stack(visit_tensors)

#         # Obtain the label
#         label_row = self.label.loc[person_id]
#         if self.setup == "prior":
#             label_tensor = torch.tensor(label_row["pasc_code_prior_four_weeks"])
#         elif self.setup == "after":
#             label_tensor = torch.tensor(label_row["pasc_code_after_four_weeks"])
#         elif self.setup == "both":
#             label_tensor = torch.tensor(max(label_row["pasc_code_after_four_weeks"], label_row["pasc_code_prior_four_weeks"]))
#         else:
#             raise Exception(f"Unknown setup `{self.setup}`")

#         return ((person_info_tensor, visits_tensor), label_tensor)

#     @staticmethod
#     def collate_fn(data):
#         batched_person_info_tensor = torch.stack([person_info for ((person_info, _), _) in data]).to(dtype=torch.float32)
#         batched_visits_tensor = torch.nn.utils.rnn.pad_sequence([visits for ((_, visits), _) in data]).to(dtype=torch.float32)
#         batched_label_tensor = torch.stack([label for (_, label) in data]).to(dtype=torch.float32)
#         return ((batched_person_info_tensor, batched_visits_tensor), batched_label_tensor)

# class LongCOVIDVisitsLSTMModel(torch.nn.Module):
#     def __init__(
#         self, 
#         person_info_dim=4,
#         visit_dim=72,
#         latent_dim=128, 
#         encoder_num_layers=1,
#         decoder_num_layers=1,
#     ):
#         super(LongCOVIDVisitsLSTMModel, self).__init__()

#         # Configurations
#         self.person_info_dim = person_info_dim
#         self.visit_dim = visit_dim
#         self.latent_dim = latent_dim
#         self.encoder_num_layers = encoder_num_layers
#         self.decoder_num_layers = decoder_num_layers

#         # Person info encoder
#         encoder_layers = [torch.nn.Linear(self.person_info_dim, self.latent_dim).to(dtype=torch.float32)]
#         for _ in range(encoder_num_layers):
#             encoder_layers += [torch.nn.ReLU(), torch.nn.Linear(self.latent_dim, self.latent_dim)]
#         self.person_info_encoder = torch.nn.Sequential(*encoder_layers).to(dtype=torch.float32)

#         # Visits encoder
#         self.rnn = torch.nn.LSTM(self.visit_dim, self.latent_dim, 1).to(dtype=torch.float32)
#         self.rnn_c0 = torch.nn.Embedding(1, self.latent_dim).to(dtype=torch.float32)

#         # Final predictor
#         predictor_layers = []
#         for _ in range(self.decoder_num_layers):
#             predictor_layers += [torch.nn.Linear(self.latent_dim, self.latent_dim), torch.nn.ReLU()]
#         predictor_layers += [torch.nn.Linear(self.latent_dim, 1), torch.nn.Sigmoid()]
#         self.predictor = torch.nn.Sequential(*predictor_layers).to(dtype=torch.float32)

#     def forward(self, person_info, visits):
#         batch_size, _ = person_info.shape
#         h0 = self.person_info_encoder(person_info).view(1, batch_size, -1)
#         c0 = self.rnn_c0(torch.tensor([0] * batch_size, dtype=torch.long)).view(1, batch_size, -1)
#         _, (hn, cn) = self.rnn(visits, (h0, c0))
#         y_pred = self.predictor(cn).view(batch_size)
#         return y_pred

# class Trainer:
#     def __init__(self, train_loader, test_loader, model, lr=0.0001, num_epochs=5):
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.model = model
#         self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#         self.loss_fn = torch.nn.MSELoss()
#         self.num_epochs = num_epochs

#     def train_epoch(self, epoch):
#         self.model.train()
#         total_loss = 0
#         num_batches = 0
#         for (x, y) in self.train_loader:
#             self.optimizer.zero_grad()
#             y_pred = self.model(*x)
#             loss = self.loss_fn(y_pred, y)
#             total_loss += loss.item()
#             num_batches += 1
#             loss.backward()
#             self.optimizer.step()
#         print(f"[Train epoch {epoch}] Avg Loss: {total_loss / num_batches}")
            
#     def test_epoch(self, epoch):
#         self.model.eval()
#         num_items = len(self.test_loader.dataset)
#         total_loss = 0
#         num_batches = 0
#         num_correct = 0
#         num_total = 0
#         num_tp, num_fp, num_tn, num_fn = 0, 0, 0, 0

#         for (x, y) in self.test_loader:
#             y_pred = self.model(*x)
#             loss = self.loss_fn(y_pred, y)
#             total_loss += loss.item()
#             num_batches += 1
#             batch_size = len(y)
#             for i in range(batch_size):
#                 gt = y[i].item()
#                 pred = 1 if y_pred[i].item() > 0.5 else 0
#                 correct = gt == pred
#                 if correct: num_correct += 1
#                 if gt == 1 and pred == 1: num_tp += 1
#                 if gt == 1 and pred == 0: num_fn += 1
#                 if gt == 0 and pred == 1: num_fp += 1
#                 if gt == 0 and pred == 0: num_tn += 1
#                 num_total += 1

#         precision = num_tp / (num_tp + num_fp)
#         recall = num_tp / (num_tp + num_fn)

#         print(f"[Test epoch {epoch}] Avg Loss: {total_loss / num_batches}, Accuracy: {num_correct / num_total}, Precision: {precision}, Recall: {recall}")

#     def train(self):
#         self.test_epoch(0)
#         for epoch in range(1, self.num_epochs + 1):
#             self.train_epoch(epoch)
#             self.test_epoch(epoch)
#         return self.model

            

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.324a6115-7c17-4d4d-94da-a2df11a87fa6"),
    all_patients_visit_day_facts_table_de_id=Input(rid="ri.foundry.main.dataset.ace57213-685a-4f18-a157-2b02b41086be"),
    everyone_cohort_de_id=Input(rid="ri.foundry.main.dataset.120adc97-2986-4b7d-9f96-42d8b5d5bedf")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 7/8/22
#Description - The final step is to aggregate information to create a data frame that contains a single row of data for each patient in the cohort.  This node aggregates all information from the cohort_all_facts_table and summarizes each patient's facts in a single row.

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
    
    df2 = df2.groupby('person_id').agg(
        F.min('Oxygen_saturation').alias('min_Oxygen_saturation'))
    df3 = df3.groupby('person_id').agg(
        F.last('blood_sodium').alias('last_blood_sodium'))
    df4 = df4.groupby('person_id').agg(
        F.last('blood_hemoglobin').alias('last_blood_hemoglobin'))
    df5 = df5.groupby('person_id').agg(
        F.last('blood_Creatinine').alias('last_blood_Creatinine'))
    df6 = df6.groupby('person_id').agg(
        F.last('blood_UreaNitrogen').alias('last_blood_UreaNitrogen'))
    
    df = df.join(df2, on=['person_id'], how='left').join(df3, on=['person_id'], how='left').join(df4, on=['person_id'], how='left').join(df5, on=['person_id'], how='left').join(df6, on=['person_id'], how='left')

    #columns to indicate whether a patient belongs in confirmed or possible subcohorts
    df = df.withColumn('confirmed_covid_patient', 
        F.when((F.col('LL_COVID_diagnosis_indicator') == 1) | (F.col('PCR_AG_Pos_indicator') == 1), 1).otherwise(0))

    df = df.withColumn('possible_covid_patient', 
        F.when(F.col('confirmed_covid_patient') == 1, 0)
        .when(F.col('Antibody_Pos_indicator') == 1, 1)
        .when(F.col('LL_Long_COVID_clinic_visit_indicator') == 1, 1)
        .when(F.col('LL_PNEUMONIADUETOCOVID_indicator') == 1, 1)
        .when(F.col('LL_MISC_indicator') == 1, 1)
        .otherwise(0))     
    #.when(F.col('LL_Long_COVID_diagnosis_indicator') == 1, 1) removed above since it seems this was removed from the conditions table
    #join above tables on patient ID  
    #df = df.join(deaths_df, 'person_id', 'left').withColumnRenamed('patient_death', 'patient_death_indicator')
    df = everyone_cohort_de_id.join(df, 'person_id','left')

    #final fill of null in non-continuous variables with 0
    df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_max_observed_or_calculated', 'postal_code', 'age')])
    
    df = df.distinct()

    return df
        
#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4b4d2bc0-b43f-4d63-abc6-ed115f0cd117"),
    all_patients_visit_day_facts_table_de_id_testing=Input(rid="ri.foundry.main.dataset.7ace5232-cf55-4095-bb84-35ae2f2350ab"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 7/8/22
#Description - The final step is to aggregate information to create a data frame that contains a single row of data for each patient in the cohort.  This node aggregates all information from the cohort_all_facts_table and summarizes each patient's facts in a single row.

def all_patients_summary_fact_table_de_id_testing(all_patients_visit_day_facts_table_de_id_testing, everyone_cohort_de_id_testing):
    everyone_cohort_de_id = everyone_cohort_de_id_testing
    all_patients_visit_day_facts_table_de_id = all_patients_visit_day_facts_table_de_id_testing

    #deaths_df = everyone_patient_deaths.select('person_id','patient_death')
    df = all_patients_visit_day_facts_table_de_id.drop('patient_death_at_visit', 'during_macrovisit_hospitalization')
  
    df = df.groupby('person_id').agg(
        F.max('BMI_rounded').alias('BMI_max_observed_or_calculated'),
        *[F.max(col).alias(col + '_indicator') for col in df.columns if col not in ('person_id', 'BMI_rounded', 'visit_date', 'had_vaccine_administered')],
        F.sum('had_vaccine_administered').alias('total_number_of_COVID_vaccine_doses'))

    #columns to indicate whether a patient belongs in confirmed or possible subcohorts
    df = df.withColumn('confirmed_covid_patient', 
        F.when((F.col('LL_COVID_diagnosis_indicator') == 1) | (F.col('PCR_AG_Pos_indicator') == 1), 1).otherwise(0))

    df = df.withColumn('possible_covid_patient', 
        F.when(F.col('confirmed_covid_patient') == 1, 0)
        .when(F.col('Antibody_Pos_indicator') == 1, 1)
        .when(F.col('LL_Long_COVID_diagnosis_indicator') == 1, 1)
        .when(F.col('LL_Long_COVID_clinic_visit_indicator') == 1, 1)
        .when(F.col('LL_PNEUMONIADUETOCOVID_indicator') == 1, 1)
        .when(F.col('LL_MISC_indicator') == 1, 1)
        .otherwise(0))     
    
    #join above tables on patient ID  
    #df = df.join(deaths_df, 'person_id', 'left').withColumnRenamed('patient_death', 'patient_death_indicator')
    df = everyone_cohort_de_id.join(df, 'person_id','left')

    #final fill of null in non-continuous variables with 0
    df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_max_observed_or_calculated', 'postal_code', 'age')])

    df = df.distinct()
    
    return df
        
#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ace57213-685a-4f18-a157-2b02b41086be"),
    Person_top_nlp_symptom=Input(rid="ri.foundry.main.dataset.74827f05-58fb-4281-aed4-4da6f728b448"),
    everyone_conditions_of_interest=Input(rid="ri.foundry.main.dataset.514f3fe8-7565-4701-8982-174b43937006"),
    everyone_devices_of_interest=Input(rid="ri.foundry.main.dataset.15ddf371-0d59-4397-9bee-866c880620cf"),
    everyone_drugs_of_interest=Input(rid="ri.foundry.main.dataset.32bad30b-9322-4e6d-8a88-ab5133e98543"),
    everyone_measurements_of_interest=Input(rid="ri.foundry.main.dataset.99e1cf7c-8848-4a3c-8f26-5cc7499311da"),
    everyone_observations_of_interest=Input(rid="ri.foundry.main.dataset.d2eefa83-105e-404c-9e21-5475e1e1110c"),
    everyone_procedures_of_interest=Input(rid="ri.foundry.main.dataset.ff38921a-cc27-4c35-9a09-9a7ccced1ad6"),
    everyone_vaccines_of_interest=Input(rid="ri.foundry.main.dataset.202ec093-e569-4af8-897a-ab8d2c4325c0"),
    microvisits_to_macrovisits=Input(rid="ri.foundry.main.dataset.d77a701f-34df-48a1-a71c-b28112a07ffa")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 7/8/22
#Description - All facts collected in the previous steps are combined in this cohort_all_facts_table on the basis of unique visit days for each patient. Indicators are created for the presence or absence of events, medications, conditions, measurements, device exposures, observations, procedures, and outcomes.  It also creates an indicator for whether the visit date where a fact was noted occurred during any hospitalization. This table is useful if the analyst needs to use actual dates of events as it provides more detail than the final patient-level table.  Use the max and min functions to find the first and last occurrences of any events.

def all_patients_visit_day_facts_table_de_id(everyone_conditions_of_interest, everyone_measurements_of_interest, everyone_procedures_of_interest, everyone_observations_of_interest, everyone_drugs_of_interest, everyone_devices_of_interest, microvisits_to_macrovisits, everyone_vaccines_of_interest, Person_top_nlp_symptom):

    macrovisits_df = microvisits_to_macrovisits
    vaccines_df = everyone_vaccines_of_interest
    procedures_df = everyone_procedures_of_interest
    devices_df = everyone_devices_of_interest
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
    df = df.join(devices_df, on=list(set(df.columns)&set(devices_df.columns)), how='outer')
    df = df.join(observations_df, on=list(set(df.columns)&set(observations_df.columns)), how='outer')
    df = df.join(conditions_df, on=list(set(df.columns)&set(conditions_df.columns)), how='outer')
    df = df.join(drugs_df, on=list(set(df.columns)&set(drugs_df.columns)), how='outer')
    df = df.join(measurements_df, on=list(set(df.columns)&set(measurements_df.columns)), how='outer')
    # df = df.join(nlp_symptom_df, on=list(set(df.columns)&set(nlp_symptom_df.columns)), how='outer')
    
    df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_rounded')])
   
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

    #final fill of null in non-continuous variables with 0
    df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_rounded')])

    for col in sorted(df.columns):
        print(col)

    return df
    
#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7ace5232-cf55-4095-bb84-35ae2f2350ab"),
    everyone_conditions_of_interest_testing=Input(rid="ri.foundry.main.dataset.ae4f0220-6939-4f61-a97a-ff78d29df156"),
    everyone_devices_of_interest_testing=Input(rid="ri.foundry.main.dataset.f423414f-5fc1-4b38-8019-a2176fd99de5"),
    everyone_drugs_of_interest_testing=Input(rid="ri.foundry.main.dataset.c467232f-7ce8-493a-9c58-19438b8bae42"),
    everyone_measurements_of_interest_testing=Input(rid="ri.foundry.main.dataset.947ff73f-4427-404f-b65b-2e709cdcbddd"),
    everyone_observations_of_interest_testing=Input(rid="ri.foundry.main.dataset.746705a9-da68-43c5-8ad9-dad8ab4ab3cf"),
    everyone_procedures_of_interest_testing=Input(rid="ri.foundry.main.dataset.a53998dc-abce-48c9-a390-b0cbf8b4a0a2"),
    everyone_vaccines_of_interest_testing=Input(rid="ri.foundry.main.dataset.97cdf176-e012-49e9-8eff-6667e5f67e1a"),
    microvisits_to_macrovisits_testing=Input(rid="ri.foundry.main.dataset.f5008fa4-e736-4244-88e1-1da7a68efcdb")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 7/8/22
#Description - All facts collected in the previous steps are combined in this cohort_all_facts_table on the basis of unique visit days for each patient. Indicators are created for the presence or absence of events, medications, conditions, measurements, device exposures, observations, procedures, and outcomes.  It also creates an indicator for whether the visit date where a fact was noted occurred during any hospitalization. This table is useful if the analyst needs to use actual dates of events as it provides more detail than the final patient-level table.  Use the max and min functions to find the first and last occurrences of any events.

def all_patients_visit_day_facts_table_de_id_testing(everyone_conditions_of_interest_testing, everyone_measurements_of_interest_testing, everyone_procedures_of_interest_testing, everyone_observations_of_interest_testing, everyone_drugs_of_interest_testing, everyone_devices_of_interest_testing, everyone_vaccines_of_interest_testing, microvisits_to_macrovisits_testing):
    everyone_drugs_of_interest = everyone_drugs_of_interest_testing
    everyone_procedures_of_interest = everyone_procedures_of_interest_testing
    everyone_observations_of_interest = everyone_observations_of_interest_testing
    everyone_measurements_of_interest = everyone_measurements_of_interest_testing
    everyone_conditions_of_interest = everyone_conditions_of_interest_testing
    everyone_vaccines_of_interest = everyone_vaccines_of_interest_testing
    everyone_devices_of_interest = everyone_devices_of_interest_testing

    macrovisits_df = microvisits_to_macrovisits_testing
    vaccines_df = everyone_vaccines_of_interest
    procedures_df = everyone_procedures_of_interest
    devices_df = everyone_devices_of_interest
    observations_df = everyone_observations_of_interest
    conditions_df = everyone_conditions_of_interest
    drugs_df = everyone_drugs_of_interest
    measurements_df = everyone_measurements_of_interest

    df = macrovisits_df.select('person_id','visit_start_date').withColumnRenamed('visit_start_date','visit_date')
    df = df.join(vaccines_df, on=list(set(df.columns)&set(vaccines_df.columns)), how='outer')
    df = df.join(procedures_df, on=list(set(df.columns)&set(procedures_df.columns)), how='outer')
    df = df.join(devices_df, on=list(set(df.columns)&set(devices_df.columns)), how='outer')
    df = df.join(observations_df, on=list(set(df.columns)&set(observations_df.columns)), how='outer')
    df = df.join(conditions_df, on=list(set(df.columns)&set(conditions_df.columns)), how='outer')
    df = df.join(drugs_df, on=list(set(df.columns)&set(drugs_df.columns)), how='outer')
    df = df.join(measurements_df, on=list(set(df.columns)&set(measurements_df.columns)), how='outer')
    
    df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_rounded')])
   
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

    #final fill of null in non-continuous variables with 0
    df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_rounded')])

    return df
    
#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bcbf4137-1508-42b5-bb05-631492b8d3b9"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2")
)
def condition_table_analysis(condition_occurrence, Long_COVID_Silver_Standard):
    TABLE = condition_occurrence
    CONCEPT_NAME_COL = "condition_concept_name"
    l, h = 0, 1000

    label = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"]))
    TABLE = TABLE.join(label, "person_id").select(F.col("person_id"), F.col(CONCEPT_NAME_COL), F.col("outcome")).filter(F.col(CONCEPT_NAME_COL) != "No matching concept")
    distinct = TABLE.groupBy(CONCEPT_NAME_COL).count().orderBy("count", ascending=False).limit(h).select(F.col(CONCEPT_NAME_COL)).toPandas()[CONCEPT_NAME_COL]
    
    pos, count = [], []
    cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
    print(len(distinct))
    t = time.time()
    for cname in distinct[l:]:
        f = TABLE.agg(
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname)),
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1))
        ).collect()
        one_count = f[0][1]
        size = f[0][0]
        pos.append(one_count/size)
        count.append(size)
    print(time.time() - t)
    r = pd.DataFrame(list(zip(distinct,pos, count)), columns=[CONCEPT_NAME_COL,"pos", "count"])
    r['neg'] = r.apply(lambda row: 1-row.pos, axis = 1)
    r['max'] = r.apply(lambda row: max(row.pos, 1-row.pos), axis=1)
    r =r[[CONCEPT_NAME_COL,'pos','neg','max','count']]
    
    return r

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6")
)
def custom_concept_set_members(concept_set_members):
    df = concept_set_members
    max_codeset_id = df.agg({"codeset_id":"max"}).collect()[0][0]
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
        ["43018325", "Performance of Urinary Filtration, Continuous, Greater than 18 hours Per Day", "urinary_filtration_penn"],
        ["3661408", "Pneumonia caused by SARS-CoV-2", "pneumonia_penn"],
        ["42538827", "Uses contraception", "contraception_penn"],
        ["43018325", "Performance of Urinary Filtration, Continuous, Greater than 18 hours Per Day", "urinary_filtration_penn"],
        ["21494995", "Pain assessment [Interpretation]", "pain_assessment_penn"],
        ["3035482", "Pain duration - Reported", "pain_duration_penn"],
        ["4271661", "Characteristic of pain", "characteristic_pain_penn"],
        ["1367500", "losartan", "losartan_penn"],
        ["19011093", "tenofovir", "tenofovir_penn"],
        ["903963", "triamcinolone", "triamcinolone_penn"],
        ["1593467", "dupilumab", "dupilumab_penn"],
        ["1336926", "tadalafil", "tadalafil_penn"],
        ["19014878", "azathioprine", "azathioprine_penn"],
        ["1367571", "heparin", "heparin_penn"],
        ["1112921", "ipratropium", "ipratropium_penn"],
        ["798875", "clonazepam 0.5 MG Oral Tablet", "clonazepam_penn"],
        ["42538117", "Transplanted heart present", "transplanted_heart_penn"],
        ["713823", "ropinirole", "ropinirole_penn"],
        ["19045045", "ergocalciferol", "ergocalciferol_penn"],
    ]
    #
    #codeset_id, concept_id, concept_set_name, is_most_recent (true),version (1), concept_name, archived (false)
    new_sets = {}
    for concept_id, concept_name, concept_set_name in data:
        if concept_set_name not in new_sets:
            max_codeset_id += 1
            new_sets[concept_set_name] = max_codeset_id
        more.loc[len(more.index)] = [new_sets[concept_set_name], concept_id, concept_set_name, True, 1, concept_name, False]
    more = more.iloc[1: , :]
    spark = SparkSession.builder.master("local[1]").appName("Penn").getOrCreate()
    more = spark.createDataFrame(more)
    mems = more.union(df)
    return mems

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2a1d8398-c54a-4732-8f23-073ced750426"),
    LL_concept_sets_fusion_everyone=Input(rid="ri.foundry.main.dataset.b36c87be-4e43-4f55-a1b2-fc48b0576a77")
)
def custom_sets(LL_concept_sets_fusion_everyone):
    df = LL_concept_sets_fusion_everyone
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
    df.loc[len(df.index)] = ['fall_risk_assessment_penn', 'FALL_RISK', 'procedure']
    df.loc[len(df.index)] = ['urinary_filtration_penn', 'URINARY_FILTRATION', 'procedure']
    df.loc[len(df.index)] = ['pneumonia_penn', 'pneumonia', 'condition']
    df.loc[len(df.index)] = ['contraception_penn', 'contraception', 'condition']
    
    df.loc[len(df.index)] = ['pain_assessment_penn', 'PAIN_ASSESSMENT', 'observation']
    df.loc[len(df.index)] = ["pain_duration_penn", "PAIN_DURATION", "observation"]
    df.loc[len(df.index)] = ["characteristic_pain_penn", "PAIN_CHARACTERISTIC", "observation"]
    df.loc[len(df.index)] = ["losartan_penn", "LOSARTAN", "drug"]
    df.loc[len(df.index)] = ["tenofovir_penn", "TENOFOVIR", "drug"]
    df.loc[len(df.index)] = ["triamcinolone_penn", "TRIAMCINOLONE", "drug"]
    df.loc[len(df.index)] = ["dupilumab_penn", "DUPILUMAB", "drug"]
    df.loc[len(df.index)] = ["tadalafil_penn", "TADALAFIL", "drug"]
    df.loc[len(df.index)] = ["azathioprine_penn", "AZATHIOPRINE", "drug"]
    df.loc[len(df.index)] = ["heparin_penn", "HEPARIN", "drug"]
    df.loc[len(df.index)] = ["ipratropium_penn", "IPRATROPIUM", "drug"]
    df.loc[len(df.index)] = ["clonazepam_penn", "CLONEAZEPAM", "drug"]
    df.loc[len(df.index)] = ["transplanted_heart_penn", "TRANSPLANTED_HEART", "condition"]
    df.loc[len(df.index)] = ["ropinirole_penn", "ROPINIROLE", "drug"]
    df.loc[len(df.index)] = ["ergocalciferol_penn", "ERGOCALCIFEROL", "drug"]

    print(df)
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7881151d-1d96-4301-a385-5d663cc22d56"),
    LL_DO_NOT_DELETE_REQUIRED_concept_sets_all=Input(rid="ri.foundry.main.dataset.029aa987-cfef-48fc-bf45-cffd3792cd93"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    custom_sets=Input(rid="ri.foundry.main.dataset.2a1d8398-c54a-4732-8f23-073ced750426")
)
#The purpose of this node is to optimize the user's experience connecting a customized concept set "fusion sheet" input data frame to replace LL_concept_sets_fusion_everyone.

def customized_concept_set_input( LL_DO_NOT_DELETE_REQUIRED_concept_sets_all, custom_sets, custom_concept_set_members):
    concept_set_members = custom_concept_set_members
    PM_LL_DO_NOT_DELETE_REQUIRED_concept_sets_all = LL_DO_NOT_DELETE_REQUIRED_concept_sets_all

    required = LL_DO_NOT_DELETE_REQUIRED_concept_sets_all
    customizable = custom_sets
    
    df = required.join(customizable, on = required.columns, how = 'outer')

    
    return df

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.842d6169-dd15-44de-9955-c978ffb1c801"),
    LL_DO_NOT_DELETE_REQUIRED_concept_sets_all=Input(rid="ri.foundry.main.dataset.029aa987-cfef-48fc-bf45-cffd3792cd93"),
    custom_sets=Input(rid="ri.foundry.main.dataset.2a1d8398-c54a-4732-8f23-073ced750426")
)
#The purpose of this node is to optimize the user's experience connecting a customized concept set "fusion sheet" input data frame to replace LL_concept_sets_fusion_everyone.

def customized_concept_set_input_testing( LL_DO_NOT_DELETE_REQUIRED_concept_sets_all, custom_sets):
    PM_LL_DO_NOT_DELETE_REQUIRED_concept_sets_all = LL_DO_NOT_DELETE_REQUIRED_concept_sets_all

    required = LL_DO_NOT_DELETE_REQUIRED_concept_sets_all
    customizable = custom_sets
    
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
    Output(rid="ri.foundry.main.dataset.ffc3d120-eaa8-4a04-8bcb-69b6dcb16ad8"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    device_exposure=Input(rid="ri.foundry.main.dataset.c1fd6d67-fc80-4747-89ca-8eb04efcb874")
)
def device_table_analysis_1(device_exposure, Long_COVID_Silver_Standard):
    TABLE = device_exposure
    CONCEPT_NAME_COL = "device_concept_name"
    l, h = 0, 1000

    label = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"]))
    TABLE = TABLE.join(label, "person_id").select(F.col("person_id"), F.col(CONCEPT_NAME_COL), F.col("outcome")).filter(F.col(CONCEPT_NAME_COL) != "No matching concept")
    distinct = TABLE.groupBy(CONCEPT_NAME_COL).count().orderBy("count", ascending=False).limit(h).select(F.col(CONCEPT_NAME_COL)).toPandas()[CONCEPT_NAME_COL]
    
    pos, count = [], []
    cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
    print(len(distinct))
    t = time.time()
    for cname in distinct[l:]:
        f = TABLE.agg(
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname)),
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1))
        ).collect()
        one_count = f[0][1]
        size = f[0][0]
        pos.append(one_count/size)
        count.append(size)
    print(time.time() - t)
    r = pd.DataFrame(list(zip(distinct,pos, count)), columns=[CONCEPT_NAME_COL,"pos", "count"])
    r['neg'] = r.apply(lambda row: 1-row.pos, axis = 1)
    r['max'] = r.apply(lambda row: max(row.pos, 1-row.pos), axis=1)
    r =r[[CONCEPT_NAME_COL,'pos','neg','max','count']]
    
    return r

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2a5480ef-7699-4f0c-bf5c-2a0f8401224d"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5")
)
def drug_table_analysis_1(drug_exposure, Long_COVID_Silver_Standard):
    TABLE = drug_exposure
    CONCEPT_NAME_COL = "drug_concept_name"
    l, h = 0, 1000

    label = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"]))
    TABLE = TABLE.join(label, "person_id").select(F.col("person_id"), F.col(CONCEPT_NAME_COL), F.col("outcome")).filter(F.col(CONCEPT_NAME_COL) != "No matching concept")
    distinct = TABLE.groupBy(CONCEPT_NAME_COL).count().orderBy("count", ascending=False).limit(h).select(F.col(CONCEPT_NAME_COL)).toPandas()[CONCEPT_NAME_COL]
    
    pos, count = [], []
    cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
    print(len(distinct))
    t = time.time()
    for cname in distinct[l:]:
        f = TABLE.agg(
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname)),
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1))
        ).collect()
        one_count = f[0][1]
        size = f[0][0]
        pos.append(one_count/size)
        count.append(size)
    print(time.time() - t)
    r = pd.DataFrame(list(zip(distinct,pos, count)), columns=[CONCEPT_NAME_COL,"pos", "count"])
    r['neg'] = r.apply(lambda row: 1-row.pos, axis = 1)
    r['max'] = r.apply(lambda row: max(row.pos, 1-row.pos), axis=1)
    r =r[[CONCEPT_NAME_COL,'pos','neg','max','count']]
    
    return r

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.120adc97-2986-4b7d-9f96-42d8b5d5bedf"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    location=Input(rid="ri.foundry.main.dataset.4805affe-3a77-4260-8da5-4f9ff77f51ab"),
    manifest_safe_harbor=Input(rid="ri.foundry.main.dataset.b4407989-1851-4e07-a13f-0539fae10f26"),
    microvisits_to_macrovisits=Input(rid="ri.foundry.main.dataset.d77a701f-34df-48a1-a71c-b28112a07ffa"),
    person=Input(rid="ri.foundry.main.dataset.f71ffe18-6969-4a24-b81c-0e06a1ae9316")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave. More information can be found in the README linked here (https://unite.nih.gov/workspace/report/ri.report.main.report.855e1f58-bf44-4343-9721-8b4c878154fe).
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node gathers some commonly used facts about these patients from the "person" and "location" tables, as well as some facts about the patient's institution (from the "manifest" table).  Available age, race, and locations data (including SDOH variables for L3 only) is gathered at this node.  The patient’s total number of visits as well as the number of days in their observation period is calculated from the “microvisits_to_macrovisits” table in this node.  These facts will eventually be joined with the final patient-level table in the final node.

def everyone_cohort_de_id( person, location, manifest_safe_harbor, microvisits_to_macrovisits, custom_concept_set_members):
    concept_set_members = custom_concept_set_members
        
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
    Output(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    location_testing=Input(rid="ri.foundry.main.dataset.06b728e0-0262-4a7a-b9b7-fe91c3f7da34"),
    manifest_safe_harbor_testing=Input(rid="ri.foundry.main.dataset.7a5c5585-1c69-4bf5-9757-3fd0d0a209a2"),
    microvisits_to_macrovisits_testing=Input(rid="ri.foundry.main.dataset.f5008fa4-e736-4244-88e1-1da7a68efcdb"),
    person_testing=Input(rid="ri.foundry.main.dataset.06629068-25fc-4802-9b31-ead4ed515da4")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave. More information can be found in the README linked here (https://unite.nih.gov/workspace/report/ri.report.main.report.855e1f58-bf44-4343-9721-8b4c878154fe).
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node gathers some commonly used facts about these patients from the "person" and "location" tables, as well as some facts about the patient's institution (from the "manifest" table).  Available age, race, and locations data (including SDOH variables for L3 only) is gathered at this node.  The patient’s total number of visits as well as the number of days in their observation period is calculated from the “microvisits_to_macrovisits” table in this node.  These facts will eventually be joined with the final patient-level table in the final node.

def everyone_cohort_de_id_testing( person_testing, location_testing, manifest_safe_harbor_testing, microvisits_to_macrovisits_testing, custom_concept_set_members):
    concept_set_members = custom_concept_set_members
        
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
    Output(rid="ri.foundry.main.dataset.514f3fe8-7565-4701-8982-174b43937006"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    customized_concept_set_input=Input(rid="ri.foundry.main.dataset.7881151d-1d96-4301-a385-5d663cc22d56"),
    everyone_cohort_de_id=Input(rid="ri.foundry.main.dataset.120adc97-2986-4b7d-9f96-42d8b5d5bedf")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node filters the condition_eras table for rows that have a condition_concept_id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these conditions are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

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
    

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ae4f0220-6939-4f61-a97a-ff78d29df156"),
    condition_occurrence_testing=Input(rid="ri.foundry.main.dataset.3e01546f-f110-4c67-a6db-9063d2939a74"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    customized_concept_set_input_testing=Input(rid="ri.foundry.main.dataset.842d6169-dd15-44de-9955-c978ffb1c801"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node filters the condition_eras table for rows that have a condition_concept_id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these conditions are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_conditions_of_interest_testing(everyone_cohort_de_id_testing, condition_occurrence_testing, customized_concept_set_input_testing, custom_concept_set_members):
    concept_set_members = custom_concept_set_members
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
    Output(rid="ri.foundry.main.dataset.15ddf371-0d59-4397-9bee-866c880620cf"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    customized_concept_set_input=Input(rid="ri.foundry.main.dataset.7881151d-1d96-4301-a385-5d663cc22d56"),
    device_exposure=Input(rid="ri.foundry.main.dataset.c1fd6d67-fc80-4747-89ca-8eb04efcb874"),
    everyone_cohort_de_id=Input(rid="ri.foundry.main.dataset.120adc97-2986-4b7d-9f96-42d8b5d5bedf")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

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
    

#################################################
## Global imports and functions included below ##
#################################################

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f423414f-5fc1-4b38-8019-a2176fd99de5"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    customized_concept_set_input_testing=Input(rid="ri.foundry.main.dataset.842d6169-dd15-44de-9955-c978ffb1c801"),
    device_exposure_testing=Input(rid="ri.foundry.main.dataset.7e24a101-2206-45d9-bcaa-b9d84bd2f990"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_devices_of_interest_testing(device_exposure_testing, everyone_cohort_de_id_testing, customized_concept_set_input_testing, custom_concept_set_members):
    concept_set_members = custom_concept_set_members
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
    Output(rid="ri.foundry.main.dataset.32bad30b-9322-4e6d-8a88-ab5133e98543"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    customized_concept_set_input=Input(rid="ri.foundry.main.dataset.7881151d-1d96-4301-a385-5d663cc22d56"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5"),
    everyone_cohort_de_id=Input(rid="ri.foundry.main.dataset.120adc97-2986-4b7d-9f96-42d8b5d5bedf"),
    first_covid_positive=Input(rid="ri.vector.main.execute.5fe4fba8-de72-489d-8a93-4e3398220f66")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

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
        .drop(F.col('first_covid_positive'))

    return df
    

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c467232f-7ce8-493a-9c58-19438b8bae42"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    customized_concept_set_input_testing=Input(rid="ri.foundry.main.dataset.842d6169-dd15-44de-9955-c978ffb1c801"),
    drug_exposure_testing=Input(rid="ri.foundry.main.dataset.26a51cab-0279-45a6-bbc0-f44a12b52f9c"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4"),
    first_covid_positive_testing=Input(rid="ri.vector.main.execute.9c7ebde3-44ed-4e96-85e3-010f458651be")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_drugs_of_interest_testing( drug_exposure_testing, everyone_cohort_de_id_testing, customized_concept_set_input_testing, first_covid_positive_testing, custom_concept_set_members):
    concept_set_members = custom_concept_set_members
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
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0) \
        .join(first_covid_positive_testing, 'person_id', 'leftouter') \
        .withColumn('BEFORE_FCP', F.when(F.datediff(F.col('visit_date'), F.col('first_covid_positive')) < 0, 1).otherwise(0)) \
        .drop(F.col('first_covid_positive'))

    return df
    

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.99e1cf7c-8848-4a3c-8f26-5cc7499311da"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    everyone_cohort_de_id=Input(rid="ri.foundry.main.dataset.120adc97-2986-4b7d-9f96-42d8b5d5bedf"),
    measurement=Input(rid="ri.foundry.main.dataset.5c8b84fb-814b-4ee5-a89a-9525f4a617c7")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node filters the measurements table for rows that have a measurement_concept_id associated with one of the concept sets described in the data dictionary in the README.  Indicator names for a positive COVID PCR or AG test, negative COVID PCR or AG test, positive COVID antibody test, and negative COVID antibody test are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date. It also finds the harmonized value as a number for BMI measurements and collapses these values to unique instances on the basis of patient and visit date.  Measurement BMI cutoffs included are intended for adults. Analyses focused on pediatric measurements should use different bounds for BMI measurements. 

def everyone_measurements_of_interest(measurement, everyone_cohort_de_id, custom_concept_set_members):
    concept_set_members = custom_concept_set_members
    
    #bring in only cohort patient ids
    persons = everyone_cohort_de_id.select('person_id', 'gender_concept_name')
    #filter procedure occurrence table to only cohort patients    
    df = measurement \
        .select('person_id','measurement_date','measurement_concept_id','harmonized_value_as_number','value_as_concept_id','value_as_number') \
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

    blood_oxygen_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('Oxygen_saturation', F.when(df.measurement_concept_id.isin(blood_oxygen_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_oxygen, highest_blood_oxygen), df.harmonized_value_as_number).otherwise(0))
    
    blood_sodium_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_sodium', F.when(df.measurement_concept_id.isin(blood_sodium_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_sodium, highest_blood_sodium), df.harmonized_value_as_number).otherwise(0))

   
    blood_hemoglobin_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_hemoglobin', F.when(df.measurement_concept_id.isin(blood_hemoglobin_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_hemoglobin, highest_blood_hemoglobin), df.harmonized_value_as_number).otherwise(0))
    
    respiratory_rate_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('respiratory_rate', F.when(df.measurement_concept_id.isin(respiratory_rate_codeset_id) & df.harmonized_value_as_number.between(lowest_respiratory_rate, highest_respiratory_rate), df.harmonized_value_as_number).otherwise(0))
 
    blood_Creatinine_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Creatinine', F.when(df.measurement_concept_id.isin(blood_Creatinine_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Creatinine, highest_blood_Creatinine), df.harmonized_value_as_number).otherwise(0))

    blood_UreaNitrogen_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_UreaNitrogen', F.when(df.measurement_concept_id.isin(blood_UreaNitrogen_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_UreaNitrogen, highest_blood_UreaNitrogen), df.harmonized_value_as_number).otherwise(0))
    
    blood_Potassium_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Potassium', F.when(df.measurement_concept_id.isin(blood_Potassium_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Potassium, highest_blood_Potassium), df.harmonized_value_as_number).otherwise(0))
    
    blood_Chloride_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Chloride', F.when(df.measurement_concept_id.isin(blood_Chloride_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Chloride, highest_blood_Chloride), df.harmonized_value_as_number).otherwise(0))
    
    blood_Calcium_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Calcium', F.when(df.measurement_concept_id.isin(blood_Calcium_codeset_id) & df.value_as_number.between(lowest_blood_Calcium, highest_blood_Calcium), df.value_as_number).otherwise(0))
    
    MCV_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('MCV', F.when(df.measurement_concept_id.isin(MCV_codeset_id) & df.value_as_number.between(lowest_MCV, highest_MCV), df.value_as_number).otherwise(0))
    
    Erythrocytes_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('Erythrocytes', F.when(df.measurement_concept_id.isin(Erythrocytes_codeset_id) & df.value_as_number.between(lowest_Erythrocytes, highest_Erythrocytes), df.value_as_number).otherwise(0))
    
    MCHC_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('MCHC', F.when(df.measurement_concept_id.isin(MCHC_codeset_id) & df.value_as_number.between(lowest_MCHC, highest_MCHC), df.value_as_number).otherwise(0))
    
    ####
    Systolic_blood_pressure_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('Systolic_blood_pressure', F.when(df.measurement_concept_id.isin(Systolic_blood_pressure_codeset_id) & df.harmonized_value_as_number.between(lowest_Systolic_blood_pressure, highest_Systolic_blood_pressure), df.harmonized_value_as_number).otherwise(0))
    
    Diastolic_blood_pressure_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('Diastolic_blood_pressure', F.when(df.measurement_concept_id.isin(Diastolic_blood_pressure_codeset_id) & df.harmonized_value_as_number.between(lowest_Diastolic_blood_pressure, highest_Diastolic_blood_pressure), df.harmonized_value_as_number).otherwise(0))

    heart_rate_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('heart_rate', F.when(df.measurement_concept_id.isin(heart_rate_codeset_id) & df.harmonized_value_as_number.between(lowest_heart_rate, highest_heart_rate), df.harmonized_value_as_number).otherwise(0))

    temperature_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('temperature', F.when(df.measurement_concept_id.isin(temperature_codeset_id) & df.harmonized_value_as_number.between(lowest_temperature, highest_temperature), df.harmonized_value_as_number).otherwise(0))
    
    blood_Glucose_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Glucose', F.when(df.measurement_concept_id.isin(blood_Glucose_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Glucose, highest_blood_Glucose), df.harmonized_value_as_number).otherwise(0))

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
    ###
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
    df = labs_df.join(BMI_df, on=['person_id', 'visit_date'], how='left').join(blood_oxygen_df, on=['person_id', 'visit_date'], how='left').join(blood_sodium_df, on=['person_id', 'visit_date'], how='left').join(blood_hemoglobin_df, on=['person_id', 'visit_date'], how='left').join(respiratory_rate_df, on=['person_id', 'visit_date'], how='left').join(blood_Creatinine_df, on=['person_id', 'visit_date'], how='left').join(blood_UreaNitrogen_df, on=['person_id', 'visit_date'], how='left').join(blood_Potassium_df, on=['person_id', 'visit_date'], how='left').join(blood_Chloride_df, on=['person_id', 'visit_date'], how='left').join(blood_Calcium_df, on=['person_id', 'visit_date'], how='left').join(MCV_df, on=['person_id', 'visit_date'], how='left').join(Erythrocytes_df, on=['person_id', 'visit_date'], how='left').join(MCHC_df, on=['person_id', 'visit_date'], how='left').join(Systolic_blood_pressure_df, on=['person_id', 'visit_date'], how='left').join(Diastolic_blood_pressure_df, on=['person_id', 'visit_date'], how='left').join(heart_rate_df, on=['person_id', 'visit_date'], how='left').join(temperature_df, on=['person_id', 'visit_date'], how='left').join(blood_Glucose_df, on=['person_id', 'visit_date'], how='left')

    return df

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.947ff73f-4427-404f-b65b-2e709cdcbddd"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4"),
    measurement_testing=Input(rid="ri.foundry.main.dataset.b7749e49-cf01-4d0a-a154-2f00eecab21e")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node filters the measurements table for rows that have a measurement_concept_id associated with one of the concept sets described in the data dictionary in the README.  Indicator names for a positive COVID PCR or AG test, negative COVID PCR or AG test, positive COVID antibody test, and negative COVID antibody test are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date. It also finds the harmonized value as a number for BMI measurements and collapses these values to unique instances on the basis of patient and visit date.  Measurement BMI cutoffs included are intended for adults. Analyses focused on pediatric measurements should use different bounds for BMI measurements. 

def everyone_measurements_of_interest_testing(measurement_testing, everyone_cohort_de_id_testing, custom_concept_set_members):
    concept_set_members = custom_concept_set_members
    
    #bring in only cohort patient ids
    persons = everyone_cohort_de_id_testing.select('person_id', 'gender_concept_name')
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
    Output(rid="ri.foundry.main.dataset.d2eefa83-105e-404c-9e21-5475e1e1110c"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    customized_concept_set_input=Input(rid="ri.foundry.main.dataset.7881151d-1d96-4301-a385-5d663cc22d56"),
    everyone_cohort_de_id=Input(rid="ri.foundry.main.dataset.120adc97-2986-4b7d-9f96-42d8b5d5bedf"),
    observation=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.746705a9-da68-43c5-8ad9-dad8ab4ab3cf"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    customized_concept_set_input_testing=Input(rid="ri.foundry.main.dataset.842d6169-dd15-44de-9955-c978ffb1c801"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4"),
    observation_testing=Input(rid="ri.foundry.main.dataset.fc1ce22e-9cf6-4335-8ca7-aa8c733d506d")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_observations_of_interest_testing(observation_testing, everyone_cohort_de_id_testing, customized_concept_set_input_testing, custom_concept_set_members):
    concept_set_members = custom_concept_set_members
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
    Output(rid="ri.foundry.main.dataset.ff38921a-cc27-4c35-9a09-9a7ccced1ad6"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    customized_concept_set_input=Input(rid="ri.foundry.main.dataset.7881151d-1d96-4301-a385-5d663cc22d56"),
    everyone_cohort_de_id=Input(rid="ri.foundry.main.dataset.120adc97-2986-4b7d-9f96-42d8b5d5bedf"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

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
    

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a53998dc-abce-48c9-a390-b0cbf8b4a0a2"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    customized_concept_set_input_testing=Input(rid="ri.foundry.main.dataset.842d6169-dd15-44de-9955-c978ffb1c801"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4"),
    procedure_occurrence_testing=Input(rid="ri.foundry.main.dataset.88523aaa-75c3-4b55-a79a-ebe27e40ba4f")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_procedures_of_interest_testing(everyone_cohort_de_id_testing, procedure_occurrence_testing, customized_concept_set_input_testing, custom_concept_set_members):
    concept_set_members = custom_concept_set_members
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.202ec093-e569-4af8-897a-ab8d2c4325c0"),
    Vaccine_fact_de_identified=Input(rid="ri.vector.main.execute.7641dae2-3118-4a2c-8a89-e4f646cbf18f"),
    everyone_cohort_de_id=Input(rid="ri.foundry.main.dataset.120adc97-2986-4b7d-9f96-42d8b5d5bedf"),
    first_covid_positive=Input(rid="ri.vector.main.execute.5fe4fba8-de72-489d-8a93-4e3398220f66")
)
def everyone_vaccines_of_interest(everyone_cohort_de_id, Vaccine_fact_de_identified, first_covid_positive):
    vaccine_fact_de_identified = Vaccine_fact_de_identified
    
    persons = everyone_cohort_de_id.select('person_id')
    vax_df = Vaccine_fact_de_identified.select('person_id', 'vaccine_txn', '1_vax_date', '2_vax_date', '3_vax_date', '4_vax_date') \
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
        .drop(F.col('first_covid_positive'))

    return df

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.97cdf176-e012-49e9-8eff-6667e5f67e1a"),
    Vaccine_fact_de_identified_testing=Input(rid="ri.foundry.main.dataset.9392c81b-bbbf-4e66-a366-a2e7e4f9db7b"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4"),
    first_covid_positive_testing=Input(rid="ri.vector.main.execute.9c7ebde3-44ed-4e96-85e3-010f458651be")
)
def everyone_vaccines_of_interest_testing(everyone_cohort_de_id_testing, Vaccine_fact_de_identified_testing, first_covid_positive_testing):
    everyone_cohort_de_id = everyone_cohort_de_id_testing
    vaccine_fact_de_identified = Vaccine_fact_de_identified_testing
    
    persons = everyone_cohort_de_id_testing.select('person_id')
    vax_df = Vaccine_fact_de_identified_testing.select('person_id', 'vaccine_txn', '1_vax_date', '2_vax_date', '3_vax_date', '4_vax_date') \
        .join(persons, 'person_id', 'inner')
        
    vax_switch = Vaccine_fact_de_identified.select('person_id', '1_vax_type', 'date_diff_1_2') \
        .withColumnRenamed('date_diff_1_2', 'DATE_DIFF_1_2') \
        .withColumn("1_VAX_JJ", F.when(F.col('1_vax_type') == 'janssen', 1).otherwise(0)) \
        .withColumn("1_VAX_PFIZER", F.when(F.col('1_vax_type') == 'pfizer', 1).otherwise(0)) \
        .withColumn("1_VAX_MODERNA", F.when(F.col('1_vax_type') == 'moderna', 1).otherwise(0)) \
        .drop(F.col('1_vax_type'))

    first_dose = vax_df.select('person_id', '1_vax_date') \
        .withColumnRenamed('1_vax_date', 'visit_date') \
        .where(F.col('visit_date').isNotNull())
    second_dose = vax_df.select('person_id', '2_vax_date') \
        .withColumnRenamed('2_vax_date', 'visit_date') \
        .where(F.col('visit_date').isNotNull())        
    third_dose = vax_df.select('person_id', '3_vax_date') \
        .withColumnRenamed('3_vax_date', 'visit_date') \
        .where(F.col('visit_date').isNotNull())
    fourth_dose = vax_df.select('person_id', '4_vax_date') \
        .withColumnRenamed('4_vax_date', 'visit_date') \
        .where(F.col('visit_date').isNotNull())

    df = first_dose.join(second_dose, on=['person_id', 'visit_date'], how='outer') \
        .join(third_dose, on=['person_id', 'visit_date'], how='outer') \
        .join(fourth_dose, on=['person_id', 'visit_date'], how='outer') \
        .join(vax_switch, on=['person_id'], how='inner') \
        .distinct()

    df = df.withColumn('had_vaccine_administered', F.lit(1)) \
        .join(first_covid_positive_testing, 'person_id', 'leftouter') \
        .withColumn('vax_before_FCP', F.when(F.datediff(F.col('visit_date'), F.col('first_covid_positive')) < 0, 1).otherwise(0)) \
        .drop(F.col('first_covid_positive'))

    return df

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.vector.main.execute.511e026f-ef9b-4f50-8f0a-d4c0855a2390"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    all_patients_summary_fact_table_de_id=Input(rid="ri.foundry.main.dataset.324a6115-7c17-4d4d-94da-a2df11a87fa6"),
    all_patients_visit_day_facts_table_de_id=Input(rid="ri.foundry.main.dataset.ace57213-685a-4f18-a157-2b02b41086be")
)
def feature_analysis_tool(all_patients_visit_day_facts_table_de_id, all_patients_summary_fact_table_de_id, Long_COVID_Silver_Standard):

    #specify table (either time-series or summary) and feature name
    TABLE = all_patients_summary_fact_table_de_id
    FEATURE_NAME = "OBESITY_indicator"
    IS_CONTINUOUS = False

    Long_COVID_Silver_Standard = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"]))
    labels_df = Long_COVID_Silver_Standard.select(F.col("person_id"), F.col("outcome"))
    data = TABLE.join(labels_df, "person_id", "outer")
    data = data.select(F.col(FEATURE_NAME), F.col("outcome"))
    #OPTIONAL aggregation, choose one or both
    # data = data.groupby('person_id','date').agg(F.min(FEATURE_NAME).alias(FEATURE_NAME), F.max('outcome').alias('outcome'))
    # data = data.groupby('person_id').agg(F.avg(FEATURE_NAME).alias(FEATURE_NAME), F.max('outcome').alias('outcome'))
    data = data.toPandas()
    if IS_CONTINUOUS:
        zipped = list(zip(data[FEATURE_NAME],data.outcome))
        neg = np.asarray([v for v,o in zipped if o==0]).reshape(-1,1)
        pos = np.asarray([v for v,o in zipped if o==1]).reshape(-1,1)
        neg = np.hstack((neg,np.zeros(neg.shape)))
        pos = np.hstack((pos,np.zeros(pos.shape)))
        model = sklearn.svm.SVC(kernel='linear', C=4)
        X, y = data[FEATURE_NAME].to_numpy().reshape(-1, 1), data['outcome']
        model.fit(X,y)
        pred = model.predict(data[FEATURE_NAME].to_numpy().reshape(-1, 1))
        
        w = model.coef_[0]
        x_0 = -model.intercept_[0]/w[0]
        margin = w[0]

        plt.figure()
        x_min, x_max = min(data[FEATURE_NAME]), max(data[FEATURE_NAME])
        y_min, y_max = -3, 3
        yy = np.linspace(y_min, y_max)
        XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
        Z = model.predict(np.c_[XX.ravel()]).reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)
        plt.plot(x_0*np.ones(shape=yy.shape), yy, 'k-')
        plt.plot(x_0*np.ones(shape=yy.shape) - margin, yy, 'k--')
        plt.plot(x_0*np.ones(shape=yy.shape) + margin, yy, 'k--')
        plt.scatter(pos, np.random.rand()*np.ones(shape=pos.shape), s=10, marker='o', facecolors='C1')
        plt.scatter(neg, np.random.rand()*-1*np.ones(shape=neg.shape), s=10, marker='^', facecolors='C2')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

        print("SVM Classification Report:\n{}".format(classification_report(data["outcome"], pred)))
    else:
        grouped = data.groupby([FEATURE_NAME,"outcome"]).size().to_frame('size').reset_index().rename({"size":"num"}, axis=1)
        s = grouped["num"].sum()
        grouped["ratio"] = grouped.apply(lambda x: x.num/s, axis=1)
        print(grouped)
        print("Pearson coefficient: {}".format(scipy.stats.pearsonr(data[FEATURE_NAME].to_numpy(),data["outcome"].to_numpy())))
    return data

@transform_pandas(
    Output(rid="ri.vector.main.execute.5fe4fba8-de72-489d-8a93-4e3398220f66"),
    everyone_conditions_of_interest=Input(rid="ri.foundry.main.dataset.514f3fe8-7565-4701-8982-174b43937006")
)
def first_covid_positive(everyone_conditions_of_interest):
    w = Window.partitionBy('person_id').orderBy(F.asc('visit_date'))
    df = everyone_conditions_of_interest \
        .filter(F.col('LL_COVID_diagnosis') == 1) \
        .select('person_id', F.first('visit_date').over(w).alias('first_covid_positive')) \
        .distinct()

    return df

@transform_pandas(
    Output(rid="ri.vector.main.execute.9c7ebde3-44ed-4e96-85e3-010f458651be"),
    everyone_conditions_of_interest_testing=Input(rid="ri.foundry.main.dataset.ae4f0220-6939-4f61-a97a-ff78d29df156")
)
def first_covid_positive_testing(everyone_conditions_of_interest_testing):
    w = Window.partitionBy('person_id').orderBy(F.asc('visit_date'))
    df = everyone_conditions_of_interest_testing \
        .filter(F.col('LL_COVID_diagnosis') == 1) \
        .select('person_id', F.first('visit_date').over(w).alias('first_covid_positive')) \
        .distinct()

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a7fb5734-565b-4647-9945-a44ff8ae62db"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    measurement=Input(rid="ri.foundry.main.dataset.5c8b84fb-814b-4ee5-a89a-9525f4a617c7")
)
def measurement_analysis_tool(measurement, Long_COVID_Silver_Standard):
    #specify the measurement concept id, the lower and upper bound (for noise purposes)
    MCID, LOW, HIGH = 40762499, 60, 100
    Long_COVID_Silver_Standard = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"]))
    labels_df = Long_COVID_Silver_Standard.select(F.col("person_id"), F.col("outcome"))
    measurement = measurement.filter((F.col("measurement_concept_id") == MCID) & F.col("harmonized_value_as_number").between(LOW,HIGH))
    data = measurement.join(labels_df, "person_id", "outer")
    data = data.filter(F.col("harmonized_value_as_number").isNotNull()).select(F.col("person_id"), F.col("measurement_date"), F.col("harmonized_value_as_number"), F.col("outcome")).withColumnRenamed("measurement_date","date").withColumnRenamed("harmonized_value_as_number","value")
    #OPTIONAL aggregation, choose one or both
    data = data.groupby('person_id','date').agg(F.min('value').alias('value'), F.max('outcome').alias('outcome'))
    data = data.groupby('person_id').agg(F.avg('value').alias('value'), F.max('outcome').alias('outcome'))
    data = data.toPandas()
    zipped = list(zip(data.value,data.outcome))
    neg = np.asarray([v for v,o in zipped if o==0]).reshape(-1,1)
    pos = np.asarray([v for v,o in zipped if o==1]).reshape(-1,1)
    neg = np.hstack((neg,np.zeros(neg.shape)))
    pos = np.hstack((pos,np.zeros(pos.shape)))
    model = sklearn.svm.SVC(kernel='linear', C=4)
    X, y = data['value'].to_numpy().reshape(-1, 1), data['outcome']
    model.fit(X,y)
    pred = model.predict(data['value'].to_numpy().reshape(-1, 1))
    
    w = model.coef_[0]
    x_0 = -model.intercept_[0]/w[0]
    margin = w[0]

    plt.figure()
    x_min, x_max = LOW, HIGH
    y_min, y_max = -3, 3
    yy = np.linspace(y_min, y_max)
    XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
    Z = model.predict(np.c_[XX.ravel()]).reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)
    plt.plot(x_0*np.ones(shape=yy.shape), yy, 'k-')
    plt.plot(x_0*np.ones(shape=yy.shape) - margin, yy, 'k--')
    plt.plot(x_0*np.ones(shape=yy.shape) + margin, yy, 'k--')
    plt.scatter(pos, np.ones(shape=pos.shape), s=10, marker='o', facecolors='C1')
    plt.scatter(neg, -1*np.ones(shape=neg.shape), s=10, marker='^', facecolors='C2')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

    print("SVM Classification Report:\n{}".format(classification_report(data["outcome"], pred)))
    return data

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d39564f3-817f-4b8a-a8b6-81d4f8fd6bf1"),
    recent_visits=Input(rid="ri.foundry.main.dataset.d42a9b47-a06f-4d43-b113-7414a1bdb9b6")
)
def num_recent_visits(recent_visits):

    # Find how many recent visits are there
    df = recent_visits \
        .groupby('person_id')['visit_date'] \
        .nunique() \
        .reset_index() \
        .rename(columns={"visit_date": "num_recent_visits"})

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d049152c-00c4-4584-aa28-c0d4a4177b22"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    observation=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516")
)
def observation_table_analysis_1(observation, Long_COVID_Silver_Standard):
    TABLE = observation
    CONCEPT_NAME_COL = "observation_concept_name"
    l, h = 0, 1000

    label = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"]))
    TABLE = TABLE.join(label, "person_id").select(F.col("person_id"), F.col(CONCEPT_NAME_COL), F.col("outcome")).filter(F.col(CONCEPT_NAME_COL) != "No matching concept")
    distinct = TABLE.groupBy(CONCEPT_NAME_COL).count().orderBy("count", ascending=False).limit(h).select(F.col(CONCEPT_NAME_COL)).toPandas()[CONCEPT_NAME_COL]
    
    pos, count = [], []
    cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
    print(len(distinct))
    t = time.time()
    for cname in distinct[l:]:
        f = TABLE.agg(
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname)),
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1))
        ).collect()
        one_count = f[0][1]
        size = f[0][0]
        pos.append(one_count/size)
        count.append(size)
    print(time.time() - t)
    r = pd.DataFrame(list(zip(distinct,pos, count)), columns=[CONCEPT_NAME_COL,"pos", "count"])
    r['neg'] = r.apply(lambda row: 1-row.pos, axis = 1)
    r['max'] = r.apply(lambda row: max(row.pos, 1-row.pos), axis=1)
    r =r[[CONCEPT_NAME_COL,'pos','neg','max','count']]
    
    return r

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2f6ebf73-3a2d-43dc-ace9-da56da4b1743"),
    everyone_cohort_de_id=Input(rid="ri.foundry.main.dataset.120adc97-2986-4b7d-9f96-42d8b5d5bedf")
)
def person_information(everyone_cohort_de_id):
    df = everyone_cohort_de_id

    # First add normalized age
    min_age = df["age"].min()
    max_age = df["age"].max()
    diff = max_age - min_age
    df["normalized_age"] = df["age"].map(lambda a: (a - min_age) / diff).fillna(0.0)

    # Then add gender information
    df["is_male"] = df["gender_concept_name"].map(lambda g: 1 if g == "MALE" else 0)
    df["is_female"] = df["gender_concept_name"].map(lambda g: 1 if g == "FEMALE" else 0)
    df["is_other_gender"] = df["gender_concept_name"].map(lambda g: 1 if g != "FEMALE" and g != "MALE" else 0)

    # Only include necessary feature
    df = df[["person_id", "age", "normalized_age", "is_male", "is_female", "is_other_gender"]]

    # Return
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.166746be-24ec-4f37-84e0-141c8e56706b"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3")
)
def procedure_table_analysis_1(procedure_occurrence, Long_COVID_Silver_Standard):
    TABLE = procedure_occurrence
    CONCEPT_NAME_COL = "procedure_concept_name"
    l, h = 0, 1000

    label = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"]))
    TABLE = TABLE.join(label, "person_id").select(F.col("person_id"), F.col(CONCEPT_NAME_COL), F.col("outcome")).filter(F.col(CONCEPT_NAME_COL) != "No matching concept")
    distinct = TABLE.groupBy(CONCEPT_NAME_COL).count().orderBy("count", ascending=False).limit(h).select(F.col(CONCEPT_NAME_COL)).toPandas()[CONCEPT_NAME_COL]
    
    pos, count = [], []
    cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
    print(len(distinct))
    t = time.time()
    for cname in distinct[l:]:
        f = TABLE.agg(
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname)),
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1))
        ).collect()
        one_count = f[0][1]
        size = f[0][0]
        pos.append(one_count/size)
        count.append(size)
    print(time.time() - t)
    r = pd.DataFrame(list(zip(distinct,pos, count)), columns=[CONCEPT_NAME_COL,"pos", "count"])
    r['neg'] = r.apply(lambda row: 1-row.pos, axis = 1)
    r['max'] = r.apply(lambda row: max(row.pos, 1-row.pos), axis=1)
    r =r[[CONCEPT_NAME_COL,'pos','neg','max','count']]
    
    return r

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d42a9b47-a06f-4d43-b113-7414a1bdb9b6"),
    all_patients_visit_day_facts_table_de_id=Input(rid="ri.foundry.main.dataset.ace57213-685a-4f18-a157-2b02b41086be")
)
def recent_visits(all_patients_visit_day_facts_table_de_id):
    # First sort the visits
    all_patients_visit_day_facts_table_de_id = all_patients_visit_day_facts_table_de_id.sort_values(["person_id", "visit_date"])

    # Get the number of visits
    # num_visits = all_patients_visit_day_facts_table_de_id \
    #     .groupby('person_id')['visit_date'] \
    #     .nunique() \
    #     .reset_index() \
    #     .rename(columns={"visit_date": "num_visits"})

    # The maximum number of visits is around 1000
    # print(num_visits.max())

    # Get the last visit of each patient
    last_visit = all_patients_visit_day_facts_table_de_id \
        .groupby("person_id")["visit_date"] \
        .max() \
        .reset_index("person_id") \
        .rename(columns={"visit_date": "last_visit_date"})
    
    # Add a six-month before the last visit column to the dataframe
    last_visit["six_month_before_last_visit"] = last_visit["last_visit_date"].map(lambda x: x - pd.Timedelta(days=180))

    # Merge last_visit back
    df = all_patients_visit_day_facts_table_de_id.merge(last_visit, on="person_id", how="left")

    # Find "recent visits" for each patient that are within six-month before their final visit
    mask = df["visit_date"] > df["six_month_before_last_visit"]
    recent_visits = df.loc[mask]

    # Add diff_date feature: how many days have passed from the previous visit?
    recent_visits["diff_date"] = recent_visits.groupby("person_id")["visit_date"].diff().fillna(0).map(lambda x: x if type(x) == int else x.days)

    # Rearrange columns
    cols = recent_visits.columns.tolist()
    cols = cols[0:2] + cols[-3:] + cols[2:-3]
    recent_visits = recent_visits[cols]

    # The maximum difference is 179
    # max_diff_date = recent_visits["diff_date"].max()
    # print(max_diff_date)

    # Make sure the data is sorted
    recent_visits = recent_visits.sort_values(["person_id", "visit_date"]).fillna(0)

    return recent_visits
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f3ea4ea5-6240-4bd5-a633-b5ea0628226e"),
    person_nlp_symptom=Input(rid="ri.foundry.main.dataset.48a30545-e47e-4ec1-8811-2f15ba55b045"),
    recent_visits=Input(rid="ri.foundry.main.dataset.d42a9b47-a06f-4d43-b113-7414a1bdb9b6")
)
def recent_visits_w_nlp_notes(recent_visits, person_nlp_symptom):
    person_nlp_symptom = person_nlp_symptom.merge(recent_visits[["person_id", "six_month_before_last_visit"]].drop_duplicates(), on="person_id", how="left")
    person_nlp_symptom = person_nlp_symptom.loc[person_nlp_symptom["note_date"] >= person_nlp_symptom["six_month_before_last_visit"]]
    person_nlp_symptom = person_nlp_symptom.rename(columns={"note_date": "visit_date"}).drop(columns=["six_month_before_last_visit", "note_id", "visit_occurrence_id"])
    person_nlp_symptom["has_nlp_note"] = 1.0

    # Make sure type checks
    df = recent_visits.merge(person_nlp_symptom, on=["person_id", "visit_date"], how="left").sort_values(["person_id", "visit_date"]).fillna(0.0)

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.235be874-5669-4c42-9ae3-3e6d37b645e1"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2"),
    device_exposure=Input(rid="ri.foundry.main.dataset.c1fd6d67-fc80-4747-89ca-8eb04efcb874"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5"),
    observation=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3"),
    train_test_model=Input(rid="ri.foundry.main.dataset.ea6c836a-9d51-4402-b1b7-0e30fb514fc8")
)
def study_misclassified(train_test_model, procedure_occurrence, condition_occurrence, drug_exposure, observation, device_exposure):
    tables = {procedure_occurrence:"procedure_concept_name",condition_occurrence:"condition_concept_name", drug_exposure:"drug_concept_name", observation:"observation_concept_name", device_exposure:"device_concept_name"}
    top_k = 100
    
    ret_tables = []
    for TABLE, CONCEPT_NAME_COL in tables.items():
        TABLE = TABLE.join(train_test_model, "person_id").filter((F.col(CONCEPT_NAME_COL) != "No matching concept") & (F.col("ens_outcome") != F.col("outcome"))).select(F.col("person_id"), F.col(CONCEPT_NAME_COL), F.col("outcome"))
        distinct = TABLE.groupBy(CONCEPT_NAME_COL).count().orderBy("count", ascending=False).limit(top_k).select(F.col(CONCEPT_NAME_COL)).toPandas()[CONCEPT_NAME_COL]
        
        pos, count = [], []
        cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
        print(len(distinct))
        t = time.time()
        for cname in distinct:
            f = TABLE.agg(
                cnt_cond((F.col(CONCEPT_NAME_COL) == cname)),
                cnt_cond((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1))
            ).collect()
            one_count = f[0][1]
            size = f[0][0]
            pos.append(one_count/size)
            count.append(size)
        print(time.time() - t)
        r = pd.DataFrame(list(zip(distinct,pos, count)), columns=["concept_name","pos", "count"])
        r['neg'] = r.apply(lambda row: 1-row.pos, axis = 1)
        r['maximum'] = r.apply(lambda row: max(row.pos, 1-row.pos), axis=1)
        r['domain'] = CONCEPT_NAME_COL
        r["heuristic"] = r.apply(lambda row: row["maximum"] * row["count"] if row["maximum"] > 0.7 else 0, axis=1)
        r =r[["concept_name", 'domain', 'pos','neg','maximum','count','heuristic']]
        ret_tables.append(r)
    ret =  pd.concat(ret_tables)
    return ret
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.133d515d-3a27-46b1-acbe-749a21a788e9"),
    condition_table_analysis=Input(rid="ri.foundry.main.dataset.bcbf4137-1508-42b5-bb05-631492b8d3b9"),
    device_table_analysis_1=Input(rid="ri.foundry.main.dataset.ffc3d120-eaa8-4a04-8bcb-69b6dcb16ad8"),
    drug_table_analysis_1=Input(rid="ri.foundry.main.dataset.2a5480ef-7699-4f0c-bf5c-2a0f8401224d"),
    observation_table_analysis_1=Input(rid="ri.foundry.main.dataset.d049152c-00c4-4584-aa28-c0d4a4177b22"),
    procedure_table_analysis_1=Input(rid="ri.foundry.main.dataset.166746be-24ec-4f37-84e0-141c8e56706b")
)
def top_concept_ids(condition_table_analysis, device_table_analysis_1, drug_table_analysis_1, procedure_table_analysis_1, observation_table_analysis_1):
    condition_table_analysis = condition_table_analysis.withColumn("domain", F.lit("condition"))
    device_table_analysis_1 = device_table_analysis_1.withColumn("domain", F.lit("device"))
    drug_table_analysis_1 = drug_table_analysis_1.withColumn("domain", F.lit("drug_concept_name"))
    procedure_table_analysis_1 = procedure_table_analysis_1.withColumn("domain", F.lit("procedure"))
    observation_table_analysis_1 = observation_table_analysis_1.withColumn("domain", F.lit("observation"))
    r = observation_table_analysis_1.union(procedure_table_analysis_1).union(drug_table_analysis_1).union(device_table_analysis_1).union(condition_table_analysis)
    r = r.withColumn("scale_above_.8", F.when((F.col("max") > 0.8), F.col("max")*F.col("count")).otherwise(F.lit(0)))
    return r

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e870e250-353c-4263-add9-98ce1858f0c6"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    person_information=Input(rid="ri.foundry.main.dataset.2f6ebf73-3a2d-43dc-ace9-da56da4b1743"),
    recent_visits_w_nlp_notes=Input(rid="ri.foundry.main.dataset.f3ea4ea5-6240-4bd5-a633-b5ea0628226e"),
    train_valid_split=Input(rid="ri.foundry.main.dataset.9d1a79f6-7627-4ee4-abc0-d6d6179c2f26")
)
def train_sequential_model(train_valid_split, Long_COVID_Silver_Standard, person_information, recent_visits_w_nlp_notes):
    # First get the splitted person ids
    train_person_ids = train_valid_split.loc[train_valid_split["split"] == "train"]
    valid_person_ids = train_valid_split.loc[train_valid_split["split"] == "valid"]

    # Use it to split the data into training x/y and validation x/y
    train_recent_visits = train_person_ids.merge(recent_visits_w_nlp_notes, on="person_id")
    valid_recent_visits = valid_person_ids.merge(recent_visits_w_nlp_notes, on="person_id")
    train_labels = train_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")
    valid_labels = valid_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")

    # Basic person information
    train_person_info = train_person_ids.merge(person_information, on="person_id")
    valid_person_info = valid_person_ids.merge(person_information, on="person_id")

    # Construct the two datasets
    train_dataset = LongCOVIDVisitsDataset(train_person_ids, train_person_info, train_recent_visits, train_labels)
    valid_dataset = LongCOVIDVisitsDataset(valid_person_ids, valid_person_info, valid_recent_visits, valid_labels)

    # Construct dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=LongCOVIDVisitsDataset.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True, collate_fn=LongCOVIDVisitsDataset.collate_fn)

    # Construct model
    model = LongCOVIDVisitsLSTMModel()

    # # Training loop
    trainer = Trainer(train_loader, valid_loader, model)
    result_model = trainer.train()

    return train_person_ids

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ea6c836a-9d51-4402-b1b7-0e30fb514fc8"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    all_patients_summary_fact_table_de_id=Input(rid="ri.foundry.main.dataset.324a6115-7c17-4d4d-94da-a2df11a87fa6"),
    all_patients_summary_fact_table_de_id_testing=Input(rid="ri.foundry.main.dataset.4b4d2bc0-b43f-4d63-abc6-ed115f0cd117")
)
def train_test_model(all_patients_summary_fact_table_de_id, all_patients_summary_fact_table_de_id_testing, Long_COVID_Silver_Standard):
    
    static_cols = ['person_id','total_visits', 'age']

    cols = static_cols + [col for col in all_patients_summary_fact_table_de_id.columns if 'indicator' in col]
    
    ## get outcome column
    Long_COVID_Silver_Standard["outcome"] = Long_COVID_Silver_Standard.apply(lambda x: max([x["pasc_code_after_four_weeks"], x["pasc_code_prior_four_weeks"]]), axis=1)
    Outcome_df = all_patients_summary_fact_table_de_id[["person_id"]].merge(Long_COVID_Silver_Standard, on="person_id", how="left")
    Outcome_df = Outcome_df[["person_id", "outcome"]].sort_values('person_id')

    Outcome = list(Outcome_df["outcome"])

    Training_and_Holdout = all_patients_summary_fact_table_de_id[cols].fillna(0.0).sort_values('person_id')
    #Testing = all_patients_summary_fact_table_de_id_testing[cols].fillna(0.0)
    X_train_no_ind, X_test_no_ind, y_train, y_test = train_test_split(Training_and_Holdout, Outcome, train_size=0.9, random_state=1)
    X_train, X_test = X_train_no_ind.set_index("person_id"), X_test_no_ind.set_index("person_id")

    lrc = LogisticRegression(penalty='l2', solver='liblinear', random_state=0, max_iter=500).fit(X_train, y_train)
    lrc2 = LogisticRegression(penalty='l2', class_weight='balanced', solver='liblinear', random_state=0, max_iter=500).fit(X_train, y_train)
    rfc = RandomForestClassifier().fit(X_train, y_train)
    gbc = GradientBoostingClassifier().fit(X_train, y_train)

    lrc_sort_features = np.argsort(lrc.coef_.flatten())[-20:]
    rfc_sort_features = np.argsort(rfc.feature_importances_.flatten())[-20:]
    plt.bar(np.arange(20), rfc.feature_importances_.flatten()[rfc_sort_features])
    plt.xticks(np.arange(20), [cols[1:][i] for i in rfc_sort_features], rotation='vertical')
    plt.tight_layout()
    plt.show()

    print("lrc important features:", [cols[1:][int(i)] for i in lrc_sort_features])
    print("rfc important features:", [cols[1:][int(i)] for i in rfc_sort_features])

    nn_scaler = StandardScaler().fit(X_train)
    nnc = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 10), random_state=1).fit(nn_scaler.transform(X_train), y_train)

    #preds = clf.predict_proba(Testing)[:,1]

    lr_test_preds = lrc.predict(X_test)
    lr2_test_preds = lrc2.predict(X_test)
    rf_test_preds = rfc.predict(X_test)
    gb_test_preds = gbc.predict(X_test)
    nnc_test_preds = nnc.predict(nn_scaler.transform(X_test))

    #test_df = 
    test_predictions = pd.DataFrame.from_dict({
        'person_id': list(X_test_no_ind["person_id"]),
        'lr_outcome': lr_test_preds.tolist(),
        'lr2_outcome': lr2_test_preds.tolist(),
        'rf_outcome': rf_test_preds.tolist(),
        'gb_outcome': gb_test_preds.tolist(),
        'nn_outcome': nnc_test_preds.tolist(),
    }, orient='columns')
    
    test_predictions = test_predictions.merge(Outcome_df, on="person_id", how="left")
    outcomes = ['lr_outcome', 'lr2_outcome', 'rf_outcome', 'gb_outcome', 'nn_outcome']
    test_predictions['ens_outcome'] = test_predictions.apply(lambda row: 1 if sum([row[c] for c in outcomes])/len(outcomes) >=0.5 else 0, axis=1)

    # predictions = pd.DataFrame.from_dict({
    #     'person_id': list(all_patients_summary_fact_table_de_id_testing["person_id"]),
    #     'outcome_likelihood': preds.tolist()
    # }, orient='columns')

    return test_predictions

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9d1a79f6-7627-4ee4-abc0-d6d6179c2f26"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    num_recent_visits=Input(rid="ri.foundry.main.dataset.d39564f3-817f-4b8a-a8b6-81d4f8fd6bf1")
)
import random

def train_valid_split( Long_COVID_Silver_Standard, num_recent_visits):
    all_person_ids_df = num_recent_visits[["person_id"]].merge(Long_COVID_Silver_Standard, on="person_id", how="left")
    all_person_ids = list(all_person_ids_df["person_id"])

    train_valid_split = 9, 1
    train_ratio = train_valid_split[0] / (train_valid_split[0] + train_valid_split[1])
    num_train = int(train_ratio * len(all_person_ids))

    random.shuffle(all_person_ids)
    train_person_ids, valid_person_ids = all_person_ids[:num_train], all_person_ids[num_train:]
    split_person_ids_df = pd.DataFrame([[person_id, "train"] for person_id in train_person_ids] + [[person_id, "valid"] for person_id in valid_person_ids], columns=["person_id", "split"]).sort_values("person_id")

    return split_person_ids_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.def6f994-533b-46b8-95ab-3708d867119c"),
    train_test_model=Input(rid="ri.foundry.main.dataset.ea6c836a-9d51-4402-b1b7-0e30fb514fc8")
)
def validation_metrics( train_test_model):
    df = train_test_model
    print("LR Classification Report:\n{}".format(classification_report(df["outcome"], df["lr_outcome"])))
    print("LR2 Classification Report:\n{}".format(classification_report(df["outcome"], df["lr2_outcome"])))
    print("RF Classification Report:\n{}".format(classification_report(df["outcome"], df["rf_outcome"])))
    print("GB Classification Report:\n{}".format(classification_report(df["outcome"], df["gb_outcome"])))
    print("NN Classification Report:\n{}".format(classification_report(df["outcome"], df["nn_outcome"])))
    print("Ensemble Classification Report:\n{}".format(classification_report(df["outcome"], df["ens_outcome"])))
    return df

