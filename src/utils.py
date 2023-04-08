import pickle
import torch
import numpy as np
import pyspark.pandas as ps
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, max as max_, min as min_
from pyspark.sql.functions import datediff
import os


selected_cols = ['1_vax_dose', 'HOSPITALIZED', 'Diastolic_blood_pressure', 'blood_Chloride', 'DEMENTIA', 'SOLIDORGANORBLOODSTEMCELLTRANSPLANT', 'DIABETESCOMPLICATED', 'THALASSEMIA', 'blood_Platelets', 'VAX_DAYS_SINCE_FCP', 'HYDROCORTISONE', 'temperature', 'RESPIRATORY', 'blood_UreaNitrogen', 'HYPERTENSION', '1_VAX_PFIZER', 'DIABETESUNCOMPLICATED', 'SUBSTANCEABUSE', 'LANCET', 'respiratory_rate', 'PCR_AG_Pos', 'DEPRESSION', 'INSOMNIA', 'BLOODPRESSURE', 'TRIAMCINOLONE', 'blood_Potassium', 'RHEUMATOLOGICDISEASE', 'MCV', 'heart_rate', 'CHRONICLUNGDISEASE', 'PREGNANCY', 'DATE_DIFF_1_2', 'Oxygen_saturation', 'Systolic_blood_pressure', 'blood_Albumin', 'vax_before_FCP', 'LUNGDISORDER', 'Antibody_Neg', '1_VAX_MODERNA', 'MODERATESEVERELIVERDISEASE', 'CONGESTIVEHEARTFAILURE', 'blood_Glucose', 'MENTAL', 'PEPTICULCER', 'blood_Procalcitonin', 'CEREBROVASCULARDISEASE', 'blood_Bilirubin', 'had_vaccine_administered', 'blood_Troponin', 'DYSPNEA', '4_vax_dose', 'HEMIPLEGIAORPARAPLEGIA', 'MILDLIVERDISEASE', 'blood_Creatinine', 'BMI_rounded', 'blood_Hematocrit', '1_VAX_JJ', 'MYOCARDIALINFARCTION', 'CORONARYARTERYDISEASE', 'PERIPHERALVASCULARDISEASE', 'blood_Leukocytes', 'Erythrocytes', 'ANTIBIOTICS', 'KIDNEYDISEASE', 'ANXIETY', 'HEARTFAILURE', '2_vax_dose', 'PREDNISONE', 'CLONEAZEPAM', 'blood_Calcium', 'MALIGNANTCANCER', 'SYSTEMICCORTICOSTEROIDS', 'HEPARIN', 'blood_hemoglobin', 'VENOUSIMPLANT', 'METASTATICSOLIDTUMORCANCERS', 'blood_sodium', '3_vax_dose', 'STEROIDS', 'PCR_AG_Neg', 'MCHC', 'Antibody_Pos', 'PSYCHOSIS', 'OTHERIMMUNOCOMPROMISED']

def remove_empty_columns_with_non_empty_cls(values, masks, non_empty_column_ids):

    # full_mask = 0

    # for idx in range(len(values)):
        
    #     mask = masks[idx]

    #     full_mask += torch.sum(mask, dim=0)

    # non_empty_column_ids = (full_mask > 0).view(-1)

    updated_values = []

    updated_masks = []

    for idx in range(len(values)):
        
        mask = masks[idx]
        val = values[idx]

        updated_values.append(val[:, non_empty_column_ids])
        updated_masks.append(mask[:, non_empty_column_ids])

    return updated_values, updated_masks

def remove_empty_columns(values, masks):

    full_mask = 0

    for idx in range(len(values)):
        
        mask = masks[idx]

        full_mask += torch.sum(mask, dim=0)

    non_empty_column_ids = (full_mask > 0).view(-1)

    empty_column_ids = torch.nonzero(full_mask < 0).view(-1)

    print("empty column ids count::", len(empty_column_ids))

    updated_values = []

    updated_masks = []

    for idx in range(len(values)):
        
        mask = masks[idx]
        val = values[idx]

        updated_values.append(val[:, non_empty_column_ids])
        updated_masks.append(mask[:, non_empty_column_ids])

    return updated_values, updated_masks, non_empty_column_ids

def get_data_min_max(values, masks):
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0]#.to(device)

    for idx in range(len(values)):
        
        vals = values[idx]
        mask = masks[idx]
        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:,i][mask[:,i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(torch.min(non_missing_vals))
                batch_max.append(torch.max(non_missing_vals))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    return data_min, data_max

def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max

def variable_time_collate_fn(tt_ls, val_ls, mask_ls, labels_ls, device=torch.device("cpu"), classify=False, activity=False,
                             data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = val_ls[0].shape[1]
    # number of labels
    if labels_ls is not None:
        N = labels_ls[0].shape[1] if activity else 1
    len_tt = [ex.size(0) for ex in val_ls]
    
    # print(len(val_ls), D)
    maxlen = np.max(len_tt)
    # print("max_len::", maxlen)
    enc_combined_tt = torch.zeros([len(val_ls), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(val_ls), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(val_ls), maxlen, D]).to(device)
    # if classify:
    if labels_ls is not None:
        if activity:
            combined_labels = torch.zeros([len(val_ls), maxlen, N]).to(device)
        else:
            combined_labels = torch.zeros([len(val_ls), N]).to(device)

    # for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
    for b in range(len(tt_ls)):
        if labels_ls is not None:
            tt, vals, mask, labels = tt_ls[b], val_ls[b], mask_ls[b], labels_ls[b]
        else:
            tt, vals, mask, labels = tt_ls[b], val_ls[b], mask_ls[b], None
        vals[mask==0] = 0
        currlen = tt.size(0)
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals.to(device)
        enc_combined_mask[b, :currlen] = mask.to(device)
        # if classify:
        if labels_ls is not None:
            if activity:
                combined_labels[b, :currlen] = labels.to(device)
            else:
                combined_labels[b] = labels.to(device)

    if not activity:
        enc_combined_vals, _, _ = normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                        att_min=data_min, att_max=data_max)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)
    # if classify:
    if labels_ls is not None:
        return combined_data, combined_labels
    
    else:
        return combined_data, None

class LongCOVIDVisitsDataset2(torch.utils.data.Dataset):
    def __init__(self, visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max, setup="both"):
        self.setup = setup
        self.visit_tensor_ls = visit_tensor_ls
        self.mask_ls = mask_ls
        self.time_step_ls = time_step_ls
        self.person_info_ls = person_info_ls
        self.label_tensor_ls = label_tensor_ls
        self.data_min = data_min
        self.data_max = data_max

        # self.data_tensor, self.label_tensor = variable_time_collate_fn(time_step_ls, visit_tensor_ls, mask_ls, label_tensor_ls, device=torch.device("cpu"), data_min=data_min, data_max=data_max)
        if self.person_info_ls is not None:
            if self.label_tensor_ls is not None:
                assert len(self.label_tensor_ls) == len(self.person_info_ls)
        
            assert len(self.person_info_ls) == len(self.time_step_ls)
        if self.label_tensor_ls is not None:
            assert len(self.visit_tensor_ls) == len(self.label_tensor_ls)

    def __len__(self):
        return len(self.time_step_ls)

    
    def __getitem__(self, idx):
        # return self.data_tensor[idx], self.label_tensor[idx]
        if self.label_tensor_ls is not None:
            labels = self.label_tensor_ls[idx]
        else:
            labels = None

        if self.person_info_ls is not None:
            return self.visit_tensor_ls[idx], self.mask_ls[idx], self.time_step_ls[idx], self.person_info_ls[idx], labels, self.data_min, self.data_max, idx
        else:
            return self.visit_tensor_ls[idx], self.mask_ls[idx], self.time_step_ls[idx], None, labels, self.data_min, self.data_max, idx

    @staticmethod
    def collate_fn(data):
        time_step_ls = [item[2] for item in data]
        visit_tensor_ls = [item[0] for item in data]
        mask_ls = [item[1] for item in data]
        
        if data[0][4] is not None:
            label_tensor_ls = [item[4] for item in data]
        else:
            label_tensor_ls = None
        if data[0][3] is not None:
            person_info_ls = [item[3].view(-1) for item in data]
        else:
            person_info_ls = None
        data_min = [item[5] for item in data][0]
        data_max = [item[6] for item in data][0]
        idx_ls = [item[7] for item in data]
        batched_data_tensor, batched_label_tensor = variable_time_collate_fn(time_step_ls, visit_tensor_ls, mask_ls, label_tensor_ls, device=torch.device("cpu"), data_min=data_min, data_max=data_max)
        # batched_person_=[]
        if person_info_ls is not None:
            batched_person_info = torch.stack(person_info_ls)
        else:
            batched_person_info = None
        idx_ls_tensor = torch.tensor(idx_ls)
        # print("selected_ids::",idx_ls_tensor)
        # batched_data_tensor = torch.stack([item[0] for item in data])
        # batched_label_tensor = torch.stack([item[1] for item in data])
        return batched_data_tensor, batched_label_tensor, batched_person_info


def write_to_pickle(data, output_filename):
    output = Transforms.get_output()
    output_fs = output.filesystem()

    with output_fs.open(output_filename + '.pickle', 'wb') as f:
        pickle.dump(data, f)

def read_from_pickle(transform_input, filename):
    with transform_input.filesystem().open(filename, 'rb') as f:
        data = pickle.load(f)

    return data

def pre_processing_visits(person_ids, all_person_info, recent_visit, label, setup="both", start_col_id = 5, end_col_id=-1, label_col_name = None, return_person_ids = False):
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if label is not None and "person_id" in label.columns:
        label = label.set_index("person_id")
    if type(person_ids) is list:
        all_person_ids = person_ids
        all_person_ids.sort()
    else:         
        # all_person_ids = list(label.index.unique())
        all_person_ids = list(recent_visit["person_id"].unique())
        all_person_ids.sort()
    if all_person_info is not None:
        all_person_info = all_person_info.set_index("person_id")
        print(all_person_info)
    recent_visit = recent_visit.set_index(["person_id", "visit_date"])
    
    print("first 10 person ids::", all_person_ids[0:10])
    # all_person_ids = list(all_person_info.index.unique())
    visit_tensor_ls = []
    mask_ls= []
    time_step_ls=[]
    if all_person_info is None:
        person_info_ls = None
    else:
        person_info_ls = []
    
    if label is None:
        label_tensor_ls = None
    else:
        label_tensor_ls = []
    person_count=0
    print(len(recent_visit.columns))
    print(recent_visit.columns)
    
    # if selected_cols is None:
    #     selected_cols = list(recent_visit.columns[start_col_id:end_col_id])
    
    # torch.save(selected_cols, os.path.join(root_dir, "selected_cols"))
    
    for person_id in all_person_ids:
        if all_person_info is not None:
            person_info = all_person_info.loc[person_id]
            person_info_tensor = torch.tensor([
                person_info["normalized_age"]
                # person_info["is_male"], 
                # person_info["is_female"], 
                # person_info["is_other_gender"]
            ])
        visits = recent_visit.loc[person_id]
        visit_tensors = []
        time_steps = []
        
        # visits_tensor2 = torch.from_numpy(np.array(visits.iloc[:,start_col_id:end_col_id].values.tolist()))
        # time_steps2 = torch.from_numpy(np.array(visits["diff_days"].values.tolist()))
        
        for i in range(len(visits)):
            visit = visits.iloc[i]
            # visit_tensor = torch.tensor([visit["diff_date"] / 180] + list(visit[5:]))
            visit_tensor = list(visit[selected_cols])# list(visit[start_col_id:end_col_id])
            time_steps.append(visit["diff_days"])
            visit_tensors.append(torch.tensor(visit_tensor))
        visits_tensor = torch.stack(visit_tensors)

        mask = (~torch.isnan(visits_tensor)).float()

        time_steps = torch.tensor(time_steps)

        # Obtain the label
        if label is not None:
            label_row = label.loc[person_id]
            # print("label row::", label_row)
            if label_col_name is None:
                if setup == "prior":
                    label_tensor = torch.tensor(label_row["pasc_code_prior_four_weeks"])
                elif setup == "after":
                    label_tensor = torch.tensor(label_row["pasc_code_after_four_weeks"])
                elif setup == "both":
                    label_tensor = torch.tensor(max(label_row["pasc_code_after_four_weeks"], label_row["pasc_code_prior_four_weeks"]))
                else:
                    raise Exception(f"Unknown setup `{setup}`")
            else:
                # print("label:", label_row[label_col_name])
                label_tensor = torch.tensor(label_row[label_col_name])
            label_tensor_ls.append(label_tensor)
        visit_tensor_ls.append(visits_tensor)
        mask_ls.append(mask)
        time_step_ls.append(time_steps)
        if person_info_ls is not None:
            person_info_ls.append(person_info_tensor)
        person_count +=1
        if person_count %100 == 0:
            print("person count::", person_count)
            
    
    if return_person_ids:
        return visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, all_person_ids
    else:
        return visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls
    

def spark_to_pandas(df):
    return df.to_pandas()

def pandas_to_spark(df):
    spark = SparkSession.builder \
    .master("local[1]").config("spark.driver.memory", "500g").getOrCreate()
    return spark.createDataFrame(df)

def select_subset_data(data_tables):
    new_data_tables = dict()
    for key in data_tables:
        new_data_tables[key] = data_tables[key].sort("person_id").limit(10000)
        
    return new_data_tables

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
