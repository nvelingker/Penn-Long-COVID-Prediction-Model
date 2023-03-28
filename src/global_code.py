from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql import Window
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,lit
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from matplotlib import pyplot as plt
import time
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

import random
import pyspark
from sklearn.model_selection import ShuffleSplit

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,roc_auc_score,recall_score, precision_score, brier_score_loss, average_precision_score, mean_absolute_error
from sklearn.decomposition import PCA
#load train or test to test copies
LOAD_TEST = 1
#turn merge_label to 0 before submission
MERGE_LABEL = 0

# #load train or test to test copies
# LOAD_TEST = 0
# #turn merge_label to 0 before submission
# MERGE_LABEL = 1

import torch
import torch.nn as nn
import numpy as np
import math
# import shap

import scipy.stats

from pyspark.sql.functions import datediff
from pyspark.sql.functions import col, max as max_, min as min_

# Set default dtype
torch.set_default_dtype(torch.float32)

# Evaluation Label Setups
LABEL_SETUPS = [
    "prior", # Only take `pasc_code_prior_four_weeks` as label
    "after", # Only take `pasc_code_after_four_weeks` as label
    "both", # Take the `or` of both `pasc_code_prior_four_weeks` and `pasc_code_after_four_weeks` as label
]

import pickle, io

class mTan_model(nn.Module):
    def __init__(self, rec, dec, classifier, latent_dim, k_iwae, device):
        super(mTan_model, self).__init__()
        self.rec = rec
        self.dec = dec
        self.classifier = classifier
        self.device = device
        self.latent_dim = latent_dim
        self.k_iwae = k_iwae

    def forward(self, *input):
        observed_data, observed_mask, observed_tp, person_info_batch = input
        batch_len  = observed_data.shape[0]
        # observed_data, observed_mask, observed_tp \
        #     = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
        out = self.rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
        qz0_mean, qz0_logvar = out[:, :, :self.latent_dim], out[:, :, self.latent_dim:]
        epsilon = torch.randn(self.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(self.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
        pred_y = self.classifier(z0, person_info_batch)
        # pred_x = self.dec(
        #     z0, observed_tp[None, :, :].repeat(k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
        # pred_x = pred_x.view(k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2]) #nsample, batch, 
        return pred_y

def convert_type(df, all_types):
    column_names = df.columns
    for idx in range(len(column_names)):
        if all_types[idx] == "integer" or all_types[idx] == "short":
            df[column_names[idx]] = df[column_names[idx]].astype('int32')
        elif all_types[idx] == "date":
            df[column_names[idx]] = pd.to_datetime(df[column_names[idx]])
        elif all_types[idx] == "double":
            df[column_names[idx]] = pd.to_numeric(df[column_names[idx]])

    return df

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

def pre_processing_data():

    all_types = ["string","date","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","integer","double","integer"]

    all_patients_visit_day_facts_table_de_id = pd.read_csv("/home/wuyinjun/all_patients_visit_day_facts_table_de_id.csv")

    all_patients_visit_day_facts_table_de_id = all_patients_visit_day_facts_table_de_id.sort_values(["person_id", "visit_date"])

    column_names = all_patients_visit_day_facts_table_de_id.columns

    for idx in range(len(column_names)):
        if all_types[idx] == "integer":
            all_patients_visit_day_facts_table_de_id[column_names[idx]] = all_patients_visit_day_facts_table_de_id[column_names[idx]].astype('int32')
        elif all_types[idx] == "date":
            all_patients_visit_day_facts_table_de_id[column_names[idx]] = pd.to_datetime(all_patients_visit_day_facts_table_de_id[column_names[idx]])
        elif all_types[idx] == "double":
            all_patients_visit_day_facts_table_de_id[column_names[idx]] = pd.to_numeric(all_patients_visit_day_facts_table_de_id[column_names[idx]])

    print()

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

    print(recent_visits)

    # Add diff_date feature: how many days have passed from the previous visit?

    min_person_visit_data = recent_visits.groupby("person_id")["visit_date"].agg(["min"])

    min_person_visit_data = min_person_visit_data.reset_index()

    min_person_visit_data.rename(columns={"min":"min_visit_date"}, inplace="True")

    recent_visits = recent_visits.merge(min_person_visit_data)

    recent_visits["diff_days"] = (recent_visits["visit_date"] - recent_visits["min_visit_date"]).dt.days.astype('int16')

    recent_visits.drop("min_visit_date", inplace=True, axis=1)

    # recent_visits["diff_date"] = recent_visits.groupby("person_id")["visit_date"].diff().fillna(0).map(lambda x: x if type(x) == int else x.days)

    # Rearrange columns
    cols = recent_visits.columns.tolist()
    cols = cols[0:2] + cols[-3:] + cols[2:-3]
    recent_visits = recent_visits[cols]

    # The maximum difference is 179
    # max_diff_date = recent_visits["diff_date"].max()
    # print(max_diff_date)

    # Make sure the data is sorted
    recent_visits = recent_visits.sort_values(["person_id", "visit_date"]).fillna(0)

    print()

    return recent_visits

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
    # else:
    #     return combined_data

def pre_processing_visits(person_ids, all_person_info, recent_visit, label, setup="both", start_col_id = 5, end_col_id=-1, label_col_name = None, return_person_ids = False):
    if label is not None:
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
    for person_id in all_person_ids:
        if all_person_info is not None:
            person_info = all_person_info.loc[person_id]
            person_info_tensor = torch.tensor([
                person_info["normalized_age"], 
                person_info["is_male"], 
                person_info["is_female"], 
                person_info["is_other_gender"]
            ])
        visits = recent_visit.loc[person_id]
        visit_tensors = []
        time_steps = []
        
        visits_tensor2 = torch.from_numpy(np.array(visits.iloc[:,start_col_id:end_col_id].values.tolist()))
        time_steps2 = torch.from_numpy(np.array(visits["diff_days"].values.tolist()))
        for i in range(len(visits)):
            visit = visits.iloc[i]
            # visit_tensor = torch.tensor([visit["diff_date"] / 180] + list(visit[5:]))
            visit_tensor = list(visit[start_col_id:end_col_id])
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

def pre_processing_visits2(person_ids, all_person_info, recent_visit, label, setup="both", start_col_id = 5, end_col_id=-1, label_col_name = None):
    label = label.set_index("person_id")
    # if type(person_ids) is list:
    #     all_person_ids = person_ids
    #     all_person_ids.sort()
    # else:         
    all_person_ids = list(label.index.unique())
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
    label_tensor_ls = []
    person_count=0
    for person_id in all_person_ids:
        if person_id in all_person_info.index:
            person_info = all_person_info.loc[person_id]
            # print("person info::", person_info)
            # print("person info values::", person_info.values.tolist())
            person_info_tensor = torch.tensor([
                person_info.values.tolist()[1:-1]
            ])
        else:
            person_info_tensor = torch.zeros(len(all_person_info.columns)-2)
        visits = recent_visit.loc[person_id]
        visit_tensors = []
        time_steps = []
        # print(visits)
        visits_tensor2 = torch.from_numpy(np.array(visits.iloc[:,start_col_id:end_col_id].values.tolist()))
        time_steps2 = torch.from_numpy(np.array(visits["diff_days"].values.tolist()))
        for i in range(len(visits)):
            visit = visits.iloc[i]
            # visit_tensor = torch.tensor([visit["diff_date"] / 180] + list(visit[5:]))
            visit_tensor = list(visit[start_col_id:end_col_id])
            time_steps.append(visit["diff_days"])
            visit_tensors.append(torch.tensor(visit_tensor))
        visits_tensor = torch.stack(visit_tensors)

        mask = (~torch.isnan(visits_tensor)).float()

        time_steps = torch.tensor(time_steps)

        # Obtain the label
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
        if person_count %10000 == 0:
            print("person count::", person_count)

    return visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls

# def pre_processing_visits(person_ids, all_person_info, recent_visit, label, setup="both"):
#     # all_person_ids = person_ids["person_id"].unique()
    
    
#     # all_person_info = all_person_info.set_index("person_id")
#     # recent_visit = recent_visit.set_index(["person_id", "visit_date"])
#     # label = label.set_index("person_id")
#     # all_person_ids = list(all_person_info.index)

#     all_person_ids = list(all_person_info.select("person_id").distinct().toPandas()["person_id"])

#     visit_tensor_ls = []
#     mask_ls= []
#     time_step_ls=[]
#     person_info_ls = []
#     label_tensor_ls = []

#     print("person id count::", len(all_person_ids))

#     person_count = 0

#     for person_id in all_person_ids:
        
#         # person_info = all_person_info.loc[person_id]
#         person_info = all_person_info.where(all_person_info["person_id"]==person_id)
#         person_info_tensor = torch.tensor([
#             list(person_info.select("normalized_age").toPandas()["normalized_age"]), 
#             list(person_info.select("is_male").toPandas()["is_male"]),
#             list(person_info.select("is_female").toPandas()["is_female"]),
#             list(person_info.select("is_other_gender").toPandas()["is_other_gender"])
#             # person_info["is_male"], 
#             # person_info["is_female"], 
#             # person_info["is_other_gender"]
#         ])
#         # print("recent_visit shape:", recent_visit.shape)
#         # visits = recent_visit.loc[person_id]
#         visits = recent_visit.where(recent_visit["person_id"]==person_id)
#         visit_tensors = []
#         time_steps = []
#         # print("person_id::", person_id)
#         # print(visits)
#         # for i in range(len(visits)):
#         #     visit = visits.iloc[i]
#         #     # visit_tensor = torch.tensor([visit["diff_date"] / 180] + list(visit[5:]))
#         #     visit_tensor = list(visit[5:-1])
#         #     # print(visit)
#         #     time_steps.append(visit["diff_days"])
#         #     visit_tensors.append(torch.tensor(visit_tensor))
#         # visits_tensor = torch.stack(visit_tensors)

#         # mask = (~torch.isnan(visits_tensor)).float()

#         # time_steps = torch.tensor(time_steps)
#         selected_column_names = visits.columns[6:-1]
#         # print("column types::", visits.dtypes[6:-1])
#         # print("selected_column names::", selected_column_names)
#         # print(visits.select(selected_column_names).toPandas())
#         numpy_arr = np.array((visits.select(selected_column_names).toPandas().values.tolist()))
#         # print("numpy_arr::", numpy_arr)
#         visits_tensor = torch.from_numpy(numpy_arr)
#         # print("tensor shape::", visits_tensor.shape)
#         time_steps = torch.from_numpy(np.array(list(visits.select("diff_days").toPandas()["diff_days"])))
#         # visits_tensor = torch.from_numpy(np.array(visits.iloc[:,5:-1].values.tolist()))
#         # time_steps = torch.from_numpy(np.array(visits["diff_days"].values.tolist()))
#         mask = (~torch.isnan(visits_tensor)).float()

#         # Obtain the label
#         # label_row = label.loc[person_id]
#         label_row = label.where(label["person_id"] == person_id)
#         if setup == "prior":
#             label_tensor = torch.tensor(list(label_row.select("pasc_code_prior_four_weeks").toPandas()["pasc_code_prior_four_weeks"]))
#         elif setup == "after":
#             label_tensor = torch.tensor(list(label_row.select("pasc_code_after_four_weeks").toPandas()["pasc_code_after_four_weeks"]))
#         elif setup == "both":
#             label1 = list(label_row.select("pasc_code_prior_four_weeks").toPandas()["pasc_code_prior_four_weeks"])
#             label2 = list(label_row.select("pasc_code_after_four_weeks").toPandas()["pasc_code_after_four_weeks"])
#             label_ls = list(map(max, zip(label1, label2)))
#             label_tensor = torch.tensor(label_ls)
#             # label_tensor = torch.tensor(max(label_row.select("pasc_code_after_four_weeks"), label_row.select("pasc_code_prior_four_weeks")))
#         else:
#             raise Exception(f"Unknown setup `{setup}`")
#         label_tensor_ls.append(label_tensor)
#         visit_tensor_ls.append(visits_tensor)
#         mask_ls.append(mask)
#         time_step_ls.append(time_steps)
#         person_info_ls.append(person_info_tensor)

#         person_count += 1
#         if person_count %10000 == 0:
#             print("person count::", person_count)
#     print("pre_processing person done!!!")
#     return visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls

class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = torch.nn.functional.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
    
    
    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        return self.linears[-1](x)

# Visits Dataset for sequential
class LongCOVIDVisitsDataset(torch.utils.data.Dataset):
    def __init__(self, person_id, person_info, recent_visit, label, setup="both"):
        self.person_ids = person_id
        self.person_info = person_info.set_index("person_id")
        self.recent_visit = recent_visit.set_index(["person_id", "visit_date"])
        self.label = label.set_index("person_id")
        self.setup = setup

        assert len(self.person_ids) == len(self.label)
        assert len(self.person_info) == len(self.label)

    def __len__(self):
        return len(self.person_info)

    def __getitem__(self, idx):
        person_id = self.person_ids.iloc[idx]["person_id"]

        # Encode person_info into vector
        person_info = self.person_info.loc[person_id]
        person_info_tensor = torch.tensor([
            person_info["normalized_age"], 
            person_info["is_male"], 
            person_info["is_female"], 
            person_info["is_other_gender"]
        ])

        # Encode each visit into vector
        visits = self.recent_visit.loc[person_id]
        visit_tensors = []
        for i in range(len(visits)):
            visit = visits.iloc[i]
            visit_tensor = torch.tensor([visit["diff_date"] / 180] + list(visit[5:]))
            visit_tensors.append(visit_tensor)
        visits_tensor = torch.stack(visit_tensors)

        # Obtain the label
        label_row = self.label.loc[person_id]
        if self.setup == "prior":
            label_tensor = torch.tensor(label_row["pasc_code_prior_four_weeks"])
        elif self.setup == "after":
            label_tensor = torch.tensor(label_row["pasc_code_after_four_weeks"])
        elif self.setup == "both":
            label_tensor = torch.tensor(max(label_row["pasc_code_after_four_weeks"], label_row["pasc_code_prior_four_weeks"]))
        else:
            raise Exception(f"Unknown setup `{self.setup}`")

        return ((person_info_tensor, visits_tensor), label_tensor)

    @staticmethod
    def collate_fn(data):
        batched_person_info_tensor = torch.stack([person_info for ((person_info, _), _) in data]).to(dtype=torch.float32)
        batched_visits_tensor = torch.nn.utils.rnn.pad_sequence([visits for ((_, visits), _) in data]).to(dtype=torch.float32)
        batched_label_tensor = torch.stack([label for (_, label) in data]).to(dtype=torch.float32)
        return ((batched_person_info_tensor, batched_visits_tensor), batched_label_tensor)

# Visits Dataset for sequential

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

    # @staticmethod
    # def collate_fn2(data):
    #     time_step_ls = [item[2] for item in data]
    #     visit_tensor_ls = [item[0] for item in data]
    #     mask_ls = [item[1] for item in data]
    #     label_tensor_ls = [item[4] for item in data]
    #     person_info_ls = [item[3].view(-1) for item in data]
    #     data_min = [item[5] for item in data][0]
    #     data_max = [item[6] for item in data][0]
    #     idx_ls = [item[7] for item in data]
    #     batched_data_tensor, batched_label_tensor = variable_time_collate_fn(time_step_ls, visit_tensor_ls, mask_ls, label_tensor_ls, device=torch.device("cpu"), data_min=data_min, data_max=data_max)
    #     # batched_person_=[]
    #     batched_person_info = torch.stack(person_info_ls)
    #     idx_ls_tensor = torch.tensor(idx_ls)
    #     print("selected_ids::",idx_ls_tensor)
    #     # batched_data_tensor = torch.stack([item[0] for item in data])
    #     # batched_label_tensor = torch.stack([item[1] for item in data])
    #     return batched_data_tensor, batched_label_tensor, batched_person_info

    # @staticmethod
    # def collate_fn(data):
    #     time_step_ls = [item[2] for item in data]
    #     visit_tensor_ls = [item[0] for item in data]
    #     mask_ls = [item[1] for item in data]
    #     label_tensor_ls = [item[4] for item in data]
    #     person_info_ls = [item[3] for item in data]
    #     data_min = [item[5] for item in data][0]
    #     data_max = [item[6] for item in data][0]
    #     batched_data_tensor, batched_label_tensor = variable_time_collate_fn(time_step_ls, visit_tensor_ls, mask_ls, label_tensor_ls, device=torch.device("cpu"), data_min=data_min, data_max=data_max)
    #     # batched_data_tensor = torch.stack([item[0] for item in data])
    #     # batched_label_tensor = torch.stack([item[1] for item in data])
    #     return batched_data_tensor, batched_label_tensor
# class LongCOVIDVisitsDataset2(torch.utils.data.Dataset):
#     def __init__(self, visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max, setup="both"):
#         self.setup = setup
#         self.data_tensor, self.label_tensor = variable_time_collate_fn(time_step_ls, visit_tensor_ls, mask_ls, label_tensor_ls, device=torch.device("cpu"), data_min=data_min, data_max=data_max)
#         assert len(self.data_tensor) == len(self.label_tensor)
#         # assert len(self.person_info) == len(self.label)

#     def __len__(self):
#         return len(self.person_info)

    
#     def __getitem__(self, idx):
#         return self.data_tensor[idx], self.label_tensor[idx]

#     @staticmethod
#     def collate_fn(data):
#         batched_data_tensor = torch.stack([item[0] for item in data])
#         batched_label_tensor = torch.stack([item[1] for item in data])
#         return batched_data_tensor, batched_label_tensor

class LongCOVIDVisitsLSTMModel(torch.nn.Module):
    def __init__(
        self, 
        person_info_dim=4,
        visit_dim=49,
        latent_dim=128, 
        encoder_num_layers=1,
        decoder_num_layers=1,
    ):
        super(LongCOVIDVisitsLSTMModel, self).__init__()

        # Configurations
        self.person_info_dim = person_info_dim
        self.visit_dim = visit_dim
        self.latent_dim = latent_dim
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers

        # Person info encoder
        encoder_layers = [torch.nn.Linear(self.person_info_dim, self.latent_dim).to(dtype=torch.float32)]
        for _ in range(encoder_num_layers):
            encoder_layers += [torch.nn.ReLU(), torch.nn.Linear(self.latent_dim, self.latent_dim)]
        self.person_info_encoder = torch.nn.Sequential(*encoder_layers).to(dtype=torch.float32)

        # Visits encoder
        self.rnn = torch.nn.LSTM(self.visit_dim, self.latent_dim, 1).to(dtype=torch.float32)
        self.rnn_c0 = torch.nn.Embedding(1, self.latent_dim).to(dtype=torch.float32)

        # Final predictor
        predictor_layers = []
        for _ in range(self.decoder_num_layers):
            predictor_layers += [torch.nn.Linear(self.latent_dim, self.latent_dim), torch.nn.ReLU()]
        predictor_layers += [torch.nn.Linear(self.latent_dim, 1), torch.nn.Sigmoid()]
        self.predictor = torch.nn.Sequential(*predictor_layers).to(dtype=torch.float32)

    def forward(self, person_info, visits):
        batch_size, _ = person_info.shape
        h0 = self.person_info_encoder(person_info).view(1, batch_size, -1)
        c0 = self.rnn_c0(torch.tensor([0] * batch_size, dtype=torch.long)).view(1, batch_size, -1)
        _, (hn, cn) = self.rnn(visits, (h0, c0))
        y_pred = self.predictor(cn)
        return y_pred

class create_classifier(nn.Module):
 
    def __init__(self, latent_dim, nhidden=16, N=2, has_static=False, static_input_dim=0):
        super(create_classifier, self).__init__()
        self.gru_rnn = nn.GRU(latent_dim, nhidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, N))

        if has_static:
            self.static_feat = nn.Sequential(
                nn.Linear(static_input_dim, 20),
                nn.ReLU(),
                nn.Linear(20, nhidden))
            self.classifier = nn.Sequential(
                nn.Linear(2*nhidden, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, N))
        
       
    def forward(self, z, static_x=None):
        _, out = self.gru_rnn(z)
        # if static_x is not None:
        if static_x is not None:
            static_feat = self.static_feat(static_x)
            return self.classifier(torch.cat([out.squeeze(0), static_feat], dim = -1))
        else:
            return self.classifier(out.squeeze(0))

class enc_mtan_rnn(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(enc_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim * 2))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def fixed_time_embedding(self, pos):
        d_model=self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, x, time_steps):
        # time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.fixed_time_embedding(time_steps).to(self.device)
            query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out

class dec_mtan_rnn(nn.Module):
 
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(dec_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*nhidden, 2*nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)    
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        
        
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
        
        
    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        # time_steps = time_steps.cpu()
        if self.learn_emb:
            query = self.learn_time_embedding(time_steps).to(self.device)
            key = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            query = self.fixed_time_embedding(time_steps).to(self.device)
            key = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out = self.att(query, key, out)
        out = self.z0_to_obs(out)
        return out        

def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask

def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

def compute_losses(dim, observed_data, observed_mask, qz0_mean, qz0_logvar, pred_x, device, norm=True, std=1):
    # observed_data, observed_mask \
    #     = dec_train_batch[:, :, :dim], dec_train_batch[:, :, dim:2*dim]

    noise_std = std  # default 0.1
    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(observed_data, pred_x, noise_logvar,
                           observed_mask).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1).sum(-1)
    if norm:
        logpx /= observed_mask.sum(-1).sum(-1)
        analytic_kl /= observed_mask.sum(-1).sum(-1)
    return logpx, analytic_kl

def mean_squared_error0(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()

def evaluate_classifier(model, test_loader, dec=None, latent_dim=None, classify_pertp=True, classifier=None,dim=41, device='cuda', reconst=False, num_sample=1):
    pred = []
    true = []
    test_loss = 0
    for item in test_loader:
        test_batch, label, person_info_batch = item
        # train_batch, label, person_info_batch = item
        if person_info_batch is not None:
            test_batch, label, person_info_batch = test_batch.float().to(device), label.to(device), person_info_batch.float().to(device)
        else:
            test_batch, label = test_batch.float().to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
        # observed_data = observed_data.float()
        # observed_mask = observed_mask.float()
        # observed_tp = observed_tp.float()
        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                           :latent_dim], out[:, :, latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                if classify_pertp:
                    pred_x = dec(z0, observed_tp[None, :, :].repeat(
                        num_sample, 1, 1).view(-1, observed_tp.shape[1]))
                    #pred_x = pred_x.view(num_sample, batch_len, pred_x.shape[1], pred_x.shape[2])
                    out = classifier(pred_x, person_info_batch)
                else:
                    out = classifier(z0, person_info_batch)
            if classify_pertp:
                N = label.size(-1)
                out = out.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * 50.
            else:
                label = label.unsqueeze(0).repeat_interleave(
                    num_sample, 0).view(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * num_sample
        pred.append(out.cpu())
        true.append(label.cpu())

    
    pred = torch.cat(pred, 0)
    true = torch.cat(true, 0)
    pred_scores = torch.sigmoid(pred[:, 1])

    pred = pred.numpy()
    true = true.numpy()
    pred_scores = pred_scores.numpy()
    print("True labels::", true.reshape(-1))
    print("Predicated labels::", pred_scores.reshape(-1))
    acc = np.mean(pred.argmax(1) == true)
    auc = roc_auc_score(
        true, pred_scores) if not classify_pertp else 0.
    true = true.reshape(-1)
    pred_labels = (pred_scores > 0.5).reshape(-1).astype(int)
    recall = recall_score(true.astype(int), pred_labels)
    precision = precision_score(true.astype(int), pred_labels)
    print("validation classification Report:\n{}".format(classification_report(true.astype(int), pred_labels)))
    return test_loss/pred.shape[0], acc, auc, recall, precision, true, pred_labels

def evaluate_classifier_final(model, test_loader, dec=None, latent_dim=None, classify_pertp=True, classifier=None,dim=41, device='cuda', reconst=False, num_sample=1):
    pred = []
    true = []
    test_loss = 0
    for item in test_loader:
        test_batch, label, person_info_batch = item
        # train_batch, label, person_info_batch = item
        if person_info_batch is not None:
            test_batch, label, person_info_batch = test_batch.float().to(device), label.to(device), person_info_batch.float().to(device)
        else:
            test_batch, label = test_batch.float().to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
        # observed_data = observed_data.float()
        # observed_mask = observed_mask.float()
        # observed_tp = observed_tp.float()
        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                           :latent_dim], out[:, :, latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                if classify_pertp:
                    pred_x = dec(z0, observed_tp[None, :, :].repeat(
                        num_sample, 1, 1).view(-1, observed_tp.shape[1]))
                    #pred_x = pred_x.view(num_sample, batch_len, pred_x.shape[1], pred_x.shape[2])
                    out = classifier(pred_x, person_info_batch)
                else:
                    out = classifier(z0, person_info_batch)
            if classify_pertp:
                N = label.size(-1)
                out = out.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * 50.
            else:
                label = label.unsqueeze(0).repeat_interleave(
                    num_sample, 0).view(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * num_sample
        pred.append(out.cpu())
        true.append(label.cpu())

    
    pred = torch.cat(pred, 0)
    true = torch.cat(true, 0)
    pred_scores = torch.sigmoid(pred[:, 1])

    pred = pred.numpy()
    true = true.numpy()
    pred_scores = pred_scores.numpy()
    print("True labels::", true.reshape(-1))
    print("Predicated labels::", pred_scores.reshape(-1))
    acc = np.mean(pred.argmax(1) == true)
    auc = roc_auc_score(
        true, pred_scores) if not classify_pertp else 0.
    true = true.reshape(-1)
    pred_labels = (pred_scores > 0.5).reshape(-1).astype(int)
    recall = recall_score(true.astype(int), pred_labels)
    precision = precision_score(true.astype(int), pred_labels)
    print("validation classification Report:\n{}".format(classification_report(true.astype(int), pred_labels)))
    return test_loss/pred.shape[0], acc, auc, recall, precision, true, pred_labels, pred_scores.reshape(-1)

def test_classifier(model, test_loader, dec=None, latent_dim=None, classify_pertp=True, classifier=None,dim=41, device='cuda', reconst=False, num_sample=1):
    pred = []
    true = []
    test_loss = 0
    for item in test_loader:
        test_batch, label, person_info_batch = item
        # train_batch, label, person_info_batch = item
        if person_info_batch is not None:
            test_batch, person_info_batch = test_batch.float().to(device), person_info_batch.float().to(device)
        else:
            test_batch = test_batch.float().to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
        # observed_data = observed_data.float()
        # observed_mask = observed_mask.float()
        # observed_tp = observed_tp.float()
        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                           :latent_dim], out[:, :, latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                if classify_pertp:
                    pred_x = dec(z0, observed_tp[None, :, :].repeat(
                        num_sample, 1, 1).view(-1, observed_tp.shape[1]))
                    #pred_x = pred_x.view(num_sample, batch_len, pred_x.shape[1], pred_x.shape[2])
                    out = classifier(pred_x, person_info_batch)
                else:
                    out = classifier(z0, person_info_batch)
            # if classify_pertp:
            #     N = label.size(-1)
            #     out = out.view(-1, N)
            #     label = label.view(-1, N)
            #     _, label = label.max(-1)
            #     test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * 50.
            # else:
            #     label = label.unsqueeze(0).repeat_interleave(
            #         num_sample, 0).view(-1)
            #     test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * num_sample
        pred.append(out.cpu())
        # true.append(label.cpu())
    pred = torch.cat(pred, 0)
    pred_scores = torch.sigmoid(pred[:, 1])
    pred_scores = pred_scores.numpy()
    pred_labels = (pred_scores > 0.5).reshape(-1).astype(int)
    return pred_labels, pred_scores

    # 
    # true = torch.cat(true, 0)
    

    # pred = pred.numpy()
    # true = true.numpy()
    
    # print("True labels::", true.reshape(-1))
    # print("Predicated labels::", pred_scores.reshape(-1))
    # acc = np.mean(pred.argmax(1) == true)
    # auc = roc_auc_score(
    #     true, pred_scores) if not classify_pertp else 0.
    # true = true.reshape(-1)
    
    # recall = recall_score(true.astype(int), pred_labels)
    # precision = precision_score(true.astype(int), pred_labels)
    # print("validation classification Report:\n{}".format(classification_report(true.astype(int), pred_labels)))
    # return test_loss/pred.shape[0], acc, auc, recall, precision, true, pred_labels

def train_mTans(lr, norm, std, alpha, k_iwae, dim, latent_dim, rec, dec, classifier, epochs, train_loader, val_loader, is_kl=True):
    best_val_loss = float('inf')
    params = (list(rec.parameters()) + list(dec.parameters()) + list(classifier.parameters()))
    # print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier))
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    val_loss, val_acc, val_auc, val_recall, val_precision,_,_ =         evaluate_classifier(rec, val_loader,latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)
    itr=0
    print("validation performance at epoch::", itr)
    print("validation loss::", val_loss)
    print("validation accuracy::", val_acc)
    print("validation auc score::", val_auc)
    print("validation recall::", val_recall)
    print("validation precision score::", val_precision)
    for itr in range(1,  epochs+ 1):
        print("epoch count::", itr)
        train_recon_loss, train_ce_loss = 0, 0
        mse = 0
        train_n = 0
        train_acc = 0
        #avg_reconst, avg_kl, mse = 0, 0, 0
        if is_kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1-0.99** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1
        local_iter = 0
        for item in train_loader:
            # ((person_info,observed_data, observed_mask, observed_tp), label) = item
            local_iter += 1
            if local_iter % 200 == 0:
                print("local iter::", local_iter, len(train_loader))

            train_batch, label, person_info_batch = item
            if person_info_batch is not None:
                person_info_batch = person_info_batch.float().to(device)
            train_batch = train_batch.float()
            observed_data, observed_mask, observed_tp = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
            observed_data, observed_mask, observed_tp = observed_data.to(device), observed_mask.to(device), observed_tp.to(device)
            # observed_data, observed_mask, observed_tp = observed_data.float(), observed_mask.float(), observed_tp.float()
            label = label.to(device)
            # train_batch, label = train_batch.to(device), label.to(device)
            batch_len  = observed_data.shape[0]
            # observed_data, observed_mask, observed_tp \
            #     = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
            out = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
            qz0_mean, qz0_logvar = out[:, :, :latent_dim], out[:, :, latent_dim:]
            epsilon = torch.randn(k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            pred_y = classifier(z0, person_info_batch)
            pred_x = dec(
                z0, observed_tp[None, :, :].repeat(k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
            pred_x = pred_x.view(k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2]) #nsample, batch, seqlen, dim
            # compute loss
            # compute_losses(dim, dec_train_batch, qz0_mean, qz0_logvar, pred_x, device, norm=True, std=1)
            logpx, analytic_kl = compute_losses(
                dim, observed_data, observed_mask, qz0_mean, qz0_logvar, pred_x, device, norm=norm, std=std)
            recon_loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(k_iwae))
            label = label.unsqueeze(0).repeat_interleave(k_iwae, 0).view(-1)
            # print(pred_y.shape, label)
            ce_loss = criterion(pred_y, label.long())
            loss = recon_loss + alpha*ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # train_ce_loss += ce_loss.item() * batch_len
            # train_recon_loss += recon_loss.item() * batch_len
            # train_acc += (pred_y.argmax(1) == label).sum().item()/k_iwae
            # train_n += batch_len
            # mse += mean_squared_error0(observed_data, pred_x.mean(0), 
            #                           observed_mask) * batch_len
        # total_time += time.time() - start_time
        # evaluate_classifier(model, test_loader, dec=None, latent_dim=None, classify_pertp=True, classifier=None,dim=41, device='cuda', reconst=False, num_sample=1)
        train_loader.shuffle = False
        train_loss, train_acc, train_auc, train_recall, train_precision,train_true, train_pred_labels = evaluate_classifier(
            rec, train_loader,latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)
        train_loader.shuffle = True
        val_loss, val_acc, val_auc, val_recall, val_precision,true, pred_labels = evaluate_classifier(
            rec, val_loader,latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)

        print("validation performance at epoch::", itr)
        print("validation loss::", val_loss)
        print("validation accuracy::", val_acc)
        print("validation auc score::", val_auc)
        print("validation recall::", val_recall)
        print("validation precision score::", val_precision)
        if val_loss <= best_val_loss:
            best_val_loss = min(best_val_loss, val_loss)
            rec_state_dict = rec.state_dict()
            dec_state_dict = dec.state_dict()
            classifier_state_dict = classifier.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            best_true, best_pred_labels = true.copy(), pred_labels.copy()
            best_train_true, best_train_pred_labels = train_true.copy(), train_pred_labels.copy()
            write_to_pickle(rec_state_dict, "mTans_rec")
            write_to_pickle(dec_state_dict, "mTans_dec")
            write_to_pickle(classifier_state_dict, "mTans_classifier")

    return best_true, best_pred_labels, best_train_true, best_train_pred_labels, rec_state_dict, dec_state_dict, classifier_state_dict
        # test_loss, test_acc, test_auc = evaluate_classifier(
        #     rec, test_loader, latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim)

class Trainer:
    def __init__(self, train_loader, test_loader, model, lr=0.0001, num_epochs=5):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()
        self.num_epochs = num_epochs

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        for (x, y) in self.train_loader:
            self.optimizer.zero_grad()
            y_pred = self.model(*x)
            loss = self.loss_fn(y_pred, y)
            total_loss += loss.item()
            num_batches += 1
            loss.backward()
            self.optimizer.step()
        print(f"[Train epoch {epoch}] Avg Loss: {total_loss / num_batches}")
            
    def test_epoch(self, epoch):
        self.model.eval()
        num_items = len(self.test_loader.dataset)
        total_loss = 0
        num_batches = 0
        num_correct = 0
        for (x, y) in self.test_loader:
            y_pred = self.model(*x)
            loss = self.loss_fn(y_pred, y)
            total_loss += loss.item()
            num_batches += 1
            batch_size = len(y)
            print(y, y_pred)
            for i in range(batch_size):
                gt = y[i].item()
                pred = 1 if y_pred[i] > 0.5 else 0
                print(gt, pred)
            return

    def train(self):
        self.test_epoch(0)
        return
        for epoch in range(1, self.num_epochs + 1):
            self.train_epoch(epoch)
            self.test_epoch(epoch)
        return self.model

            
#P-LSTM
class GradMod(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input, other):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        result = torch.fmod(input, other)
        ctx.save_for_backward(input, other)        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        x, y = ctx.saved_variables
        return grad_output * 1, grad_output * torch.neg(torch.floor_divide(x, y))

OFF_SLOPE=1e-3

class PLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.Periods = nn.Parameter(torch.Tensor(hidden_sz, 1))
        self.Shifts = nn.Parameter(torch.Tensor(hidden_sz, 1))
        self.On_End = nn.Parameter(torch.Tensor(hidden_sz, 1))
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        # Phased LSTM
        # -----------------------------------------------------
        nn.init.constant_(self.On_End, 0.05) # Set to be 5% "open"
        nn.init.uniform_(self.Shifts, 0, 100) # Have a wide spread of shifts
        # Uniformly distribute periods in log space between exp(1, 3)
        self.Periods.data.copy_(torch.exp((3 - 1) *
            torch.rand(self.Periods.shape) + 1))
        # -----------------------------------------------------
         
    def forward(self, x, ts,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        # PHASED LSTM
        # -----------------------------------------------------
        # Precalculate some useful vars
        shift_broadcast = self.Shifts.view(1, -1)
        period_broadcast = abs(self.Periods.view(1, -1))
        on_mid_broadcast = abs(self.On_End.view(1, -1)) * 0.5 * period_broadcast
        on_end_broadcast = abs(self.On_End.view(1, -1)) * period_broadcast                       
        
        def calc_time_gate(time_input_n):
            # Broadcast the time across all units
            t_broadcast = time_input_n.unsqueeze(-1)
            # Get the time within the period
            in_cycle_time = GradMod.apply(t_broadcast + shift_broadcast, period_broadcast)            

            # Find the phase
            is_up_phase = torch.le(in_cycle_time, on_mid_broadcast)
            is_down_phase = torch.gt(in_cycle_time, on_mid_broadcast)*torch.le(in_cycle_time, on_end_broadcast)

            # Set the mask
            sleep_wake_mask = torch.where(is_up_phase, in_cycle_time/on_mid_broadcast,
                                torch.where(is_down_phase,
                                    (on_end_broadcast-in_cycle_time)/on_mid_broadcast,
                                        OFF_SLOPE*(in_cycle_time/period_broadcast)))
            return sleep_wake_mask
        # -----------------------------------------------------

        HS = self.hidden_size
        for t in range(seq_sz):
            old_c_t = c_t
            old_h_t = h_t
            x_t = x[:, t, :]
            t_t = ts[:, t]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            # PHASED LSTM
            # -----------------------------------------------------
            # Get time gate openness
            sleep_wake_mask = calc_time_gate(t_t)
            # Sleep if off, otherwise stay a bit on
            c_t = sleep_wake_mask*c_t + (1. - sleep_wake_mask)*old_c_t
            h_t = sleep_wake_mask*h_t + (1. - sleep_wake_mask)*old_h_t
            # -----------------------------------------------------
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

class PLSTM_Net(nn.Module):

    def __init__(self, inp_dim=1, hidden_dim=20, use_lstm=False):
        super(PLSTM_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm

        # if use_lstm:
        #     pass
        #     # One extra vector for time            
        #     self.rnn = LSTMRaw(inp_dim + 1, hidden_dim)
        # else:
        self.rnn = PLSTM(inp_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, points, times):         
        if self.use_lstm:
            combined_input = torch.cat((points, torch.unsqueeze(times, dim=-1)), -1)      
            lstm_out, _ = self.rnn(combined_input)
        else:
            lstm_out, _ = self.rnn(points, times)
        linear_out = self.linear(lstm_out)
        final_logits = linear_out[:, -1, :]
        classes = F.log_softmax(final_logits, dim=1)
        return classes

def test_plstm(model, device, test_loader, dim):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    pred = []
    true = []
    with torch.no_grad():
        # for b_idx, (bX, bXmask, bT, bY) in enumerate(test_loader):
        for item in test_loader:
            test_batch, label = item
            observed_data, observed_mask, observed_tp = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
            observed_data, observed_mask, observed_tp = observed_data.to(device), observed_mask.to(device), observed_tp.to(device)
            observed_data, observed_mask, observed_tp = observed_data.float(), observed_mask.float(), observed_tp.float()
            total += int(bXmask[:,-1].sum())
            bX, bXmask = bX.to(device), bXmask.to(device)
            bT, bY = bT.to(device), bY.to(device)
            output = model(bX, bT)
            test_loss += F.nll_loss(output, bY, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(bY.view_as(pred)).sum().item()

    pred = torch.cat(pred, 0)
    true = torch.cat(true, 0)
    pred_scores = torch.sigmoid(pred[:, 1])

    pred = pred.numpy()
    true = true.numpy()
    pred_scores = pred_scores.numpy()
    print("True labels::", true.reshape(-1))
    print("Predicated labels::", pred_scores.reshape(-1))
    acc = np.mean(pred.argmax(1) == true)
    auc = roc_auc_score(
        true, pred_scores) if not classify_pertp else 0.
    true = true.reshape(-1)
    pred_labels = (pred_scores > 0.5).reshape(-1).astype(int)
    recall = recall_score(true.astype(int), pred_labels)
    precision = precision_score(true.astype(int), pred_labels)
    print("validation classification Report:\n{}".format(classification_report(true.astype(int), pred_labels)))

    test_loss /= total

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))
    return correct / total

def transform_and_draw_figures_for_shapley_values(image_idx, observed_data, observed_mask, observed_tp, shap_values1, shap_values_ls, shap_feature_value_ls):
    

    shap_value_label_mappings = {}

    shap_feat_value_label_mappings = {}

    for label in range(2):

        

        observed_data_feature_shapley_value = np.sum(shap_values1[label][0], axis = 1)

        observed_mask_feature_shapley_value = np.sum(shap_values1[label][1], axis = 1)

        observed_tp_feature_shapley_value = np.sum(shap_values1[label][2], axis = 1)

        shap_value_label_mappings[label] = shap_values1[label]

        shap_feat_value_label_mappings[label] = [observed_data_feature_shapley_value, observed_mask_feature_shapley_value, observed_tp_feature_shapley_value, shap_values1[label][3]]

        feature_names = ["feature " + str(k) + " at time " + str(r) for r in range(observed_data.shape[1]) for k in range(observed_data.shape[2])]
        
        fig_handle = plt.figure(4*image_idx + 2*label)
        print("start drawing figures")
        shap.summary_plot(shap_values1[label][0].reshape(1,-1), observed_data.reshape(1,-1), feature_names=feature_names)
        print("end drawing figures")
        plt.show()
        # write_to_pickle(fig_handle, "label_" + str(label) + "_image_" + str(image_idx) + "_data")

        print("end saving figures")

        fig_handle = plt.figure(4*image_idx + 2*label + 1)

        feature_names = ["feature " + str(k) for k in range(observed_data.shape[2])]

        shap.summary_plot(observed_data_feature_shapley_value.reshape(1,-1), torch.sum(observed_data, dim=1).reshape(1,-1), feature_names=feature_names)

        plt.show()
        # write_to_pickle(fig_handle, "label_" + str(label) + "_feat_image_" + str(image_idx) + "_data")

    shap_values_ls.append(shap_value_label_mappings)

    shap_feature_value_ls.append(shap_feat_value_label_mappings)

def evaluate_shapley_value(data_loader, rec, dec, classifier, latent_dim, k_iwae, dim, device):

    shap_values_ls = []

    image_idx = 0
    shap_feature_value_ls = []

    for item in data_loader:

        train_batch, label, person_info_batch = item
        person_info_batch = person_info_batch.float().to(device)
        train_batch = train_batch.float()
        observed_data, observed_mask, observed_tp = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
        observed_data, observed_mask, observed_tp = observed_data.to(device), observed_mask.to(device), observed_tp.to(device)
        base_observed_data, base_observed_mask, base_observed_tp, base_person_info = torch.zeros_like(observed_data), torch.zeros_like(observed_mask), torch.zeros_like(observed_tp), torch.zeros_like(person_info_batch)

        model = mTan_model(rec, dec, classifier, latent_dim, k_iwae, device)
        

        e = shap.DeepExplainer(model, [base_observed_data, base_observed_mask, base_observed_tp, base_person_info])

        model.classifier.gru_rnn.train()
        model.rec.gru_rnn.train()
        
        shap_values1 = e.shap_values([observed_data, observed_mask, observed_tp, person_info_batch])
        print("here")

        transform_and_draw_figures_for_shapley_values(image_idx, observed_data, observed_mask, observed_tp, shap_values1, shap_values_ls, shap_feature_value_ls)

        # explainer1 = shap.GradientExplainer(model, [observed_data, observed_mask, observed_tp, person_info_batch])

        # # pred_x, pred_y, qz0_mean, qz0_logvar = model((observed_data, observed_mask, observed_tp, person_info_batch))
        # model.train()
        # shap_values1 = explainer1.shap_values([observed_data, observed_mask, observed_tp, person_info_batch])
        # shap_values_ls.append(shap_values1)
        # print(person_info_batch)
        # print("shapley value::", shap_values1[0][-1])
        # print("unique shapley value::", np.unique(shap_values1[0][0]))
        del model
        image_idx += 1

    print("finish computing shapley values!!")
    return shap_values_ls, shap_feature_value_ls

def write_to_pickle(data, output_filename):
    output = Transforms.get_output()
    output_fs = output.filesystem()

    with output_fs.open(output_filename + '.pickle', 'wb') as f:
        pickle.dump(data, f)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def read_model_from_pickle(transform_input, filename):
    with transform_input.filesystem().open(filename, 'rb') as f:
        data = CPU_Unpickler(f).load()
        # data = pickle.load(f, map_location=torch.device('cpu'))

    return data

def read_from_pickle(transform_input, filename):
    with transform_input.filesystem().open(filename, 'rb') as f:
        data = pickle.load(f)

    return data

def obtain_latent_sequence(observation, condition_occurrence, drug_exposure, procedure_occurrence, Long_COVID_Silver_Standard, measurement, device_exposure, k = 200, start_id=0):
    

    procedure_occurrence = procedure_occurrence.withColumnRenamed('procedure_date','visit_date')
    condition_occurrence = condition_occurrence.withColumnRenamed('condition_start_date','visit_date')
    drug_exposure = drug_exposure.withColumnRenamed('drug_exposure_start_date','visit_date')
    observation = observation.withColumnRenamed('observation_date','visit_date')
    measurement = measurement.withColumnRenamed('measurement_date','visit_date')
    device_exposure = device_exposure.withColumnRenamed('device_exposure_start_date','visit_date')
    tables = {procedure_occurrence:"procedure_concept_id",condition_occurrence:"condition_concept_id", drug_exposure:"drug_concept_id", observation:"observation_concept_id", measurement:"measurement_concept_id", device_exposure:"device_concept_id"}
    labels_df = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"])).select(F.col("person_id"), F.col("outcome"))

    feats = Long_COVID_Silver_Standard.select(F.col("person_id"))
    table_id = 0

    reduced_tables = {}

    union_table = None

    for TABLE, CONCEPT_ID_COL in tables.items():
        TABLE = TABLE.select(F.col("person_id"), F.col("visit_date"))
        if union_table is None:
            union_table = TABLE
        else:
            union_table = union_table.union(TABLE)

    # last_visit = all_patients_visit_day_facts_table_de_id \
    #     .groupby("person_id")["visit_date"] \
    #     .max() \
    #     .reset_index("person_id") \
    #     .rename(columns={"visit_date": "last_visit_date"})
    
    # # Add a six-month before the last visit column to the dataframe
    # last_visit["six_month_before_last_visit"] = last_visit["last_visit_date"].map(lambda x: x - pd.Timedelta(days=180))

    last_visit = union_table \
        .groupBy("person_id") \
        .agg(max_("visit_date")).withColumnRenamed("max(visit_date)", "last_visit_date")
    last_visit = last_visit.withColumn("six_month_before_last_visit", F.date_sub(last_visit["last_visit_date"], 180))

    for TABLE, CONCEPT_ID_COL in tables.items():
        
        print("original table size::", TABLE.count())

        df = TABLE.join(last_visit, on = "person_id", how = "left")

        df = df.where(datediff(df["visit_date"], df["six_month_before_last_visit"]) > 0)
        reduced_tables[df] = CONCEPT_ID_COL
        print(df)
        print("reduced table size::", df.count())

    for TABLE, CONCEPT_ID_COL in reduced_tables.items():
        print(TABLE.show())
        TABLE = TABLE.select(F.col("person_id"), F.col("visit_date"), F.col(CONCEPT_ID_COL))             
        # distinct = TABLE.groupBy(CONCEPT_ID_COL).count().orderBy("count", ascending=False).limit(k).select(F.col(CONCEPT_ID_COL)).toPandas()[CONCEPT_ID_COL].tolist()
        distinct = TABLE.groupBy(CONCEPT_ID_COL).count().orderBy("count", ascending=False).select(F.col(CONCEPT_ID_COL)).toPandas()[CONCEPT_ID_COL].tolist()
        print("top 10 distinct concepts before::", distinct[0:10])
        distinct = distinct[start_id*k:(start_id + 1)*k]
        print("top 10 distinct concepts after::", distinct[0:10])
        df = TABLE.filter(F.col(CONCEPT_ID_COL).isin(distinct))
        df= df.groupBy('person_id', 'visit_date').pivot(CONCEPT_ID_COL).agg(F.lit(1)).na.fill(0)
        df = df.select([F.col(c).alias(CONCEPT_ID_COL[:3]+c) if c != "person_id" and c != "visit_date" else c for c in df.columns ])

        print("df columns::", len(df.columns), df.columns)
        print("feats columns::", len(feats.columns), feats.columns)
        print("df row count::", df.count())
        print("feats row count::", feats.count())
        print("common columns::", list(set(df.columns)&set(feats.columns)))
        if table_id == 0:
            feats = feats.join(df, on=list(set(df.columns)&set(feats.columns)), how = "left")
        else:
            feats = feats.join(df, on=list(set(df.columns)&set(feats.columns)), how = "outer")
        table_id += 1
        print(feats.show())
        print()
        print()

    unique_person_ids = list(feats.select(F.col('person_id')).distinct().toPandas()['person_id'])

    print("total person count:", len(unique_person_ids))

    empty_person_ids = list(feats.filter(F.col("visit_date").isNull()).select(F.col("person_id")).distinct().toPandas()['person_id'])

    unique_person_ids = [idx for idx in unique_person_ids if idx not in empty_person_ids]

    print("remaining person count:", len(unique_person_ids))

    feats = feats.filter(feats["person_id"].isin(unique_person_ids))

    # for table in list(tables.keys()):
        

    
    # data = feats.na.fill(0).join(labels_df, "person_id")
    data = feats.join(labels_df, "person_id")
    print("finish!!")
    return data