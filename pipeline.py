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
MERGE_LABEL = 1
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8a16f982-ef2a-47bf-9cf9-5630e535e8b7"),
    add_date_diff_cols=Input(rid="ri.foundry.main.dataset.d3dc0e61-b976-406e-917b-a7e47c925333")
)
def Produce_obs_dataset(add_date_diff_cols):
    recent_visits = add_date_diff_cols
    # def produce_dataset(train_valid_split, Long_COVID_Silver_Standard, person_information, recent_visits_w_nlp_notes_2):
    print("start")
    # unique_person_ids = obs_latent_sequence.select("person_id").distinct()
    
    unique_person_ids = list(recent_visits.select(F.col('person_id')).distinct().toPandas()['person_id'])

    print("unique person ids::", unique_person_ids[0:10])

    random.shuffle(unique_person_ids)

    print("random unique person ids::", unique_person_ids[0:10])

    train_person_ids = unique_person_ids[0:int(len(unique_person_ids)*0.9)]
    valid_person_ids = unique_person_ids[int(len(unique_person_ids)*0.9):]

    # spark.createDataFrame(data=dept, schema = deptColumns)
    # write_to_pickle([torch.zeros(10), torch.ones(10)], "sample_data")
    # # First get the splitted person ids
    # train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    # valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")
    train_recent_visits = recent_visits.filter(recent_visits["person_id"].isin(train_person_ids))
    valid_recent_visits = recent_visits.filter(recent_visits["person_id"].isin(valid_person_ids))
    train_labels = train_recent_visits.select(F.col("person_id"), F.col("outcome")).distinct()
    valid_labels = valid_recent_visits.select(F.col("person_id"), F.col("outcome")).distinct()
    train_recent_visits = train_recent_visits.drop(F.col("outcome"))
    valid_recent_visits = valid_recent_visits.drop(F.col("outcome"))
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")

    # train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # print(train_recent_visits.show())

    train_person_info = None#train_person_ids.join(person_information, on="person_id")
    valid_person_info = None#valid_person_ids.join(person_information, on="person_id")

    print("start pre-processing!!!")
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits(train_person_ids, None, train_recent_visits.toPandas(), train_labels.toPandas(), setup="both", start_col_id = 2, end_col_id=-2, label_col_name="outcome")

    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits(valid_person_ids, None, valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both", start_col_id = 2, end_col_id=-2, label_col_name="outcome")
    print("finish pre-processing!!!")

    visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)

    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    valid_subset_ids = list(range(10))

    subset_valid_visit_tensor_ls = [valid_visit_tensor_ls[idx] for idx in valid_subset_ids]    
    subset_valid_mask_ls = [valid_mask_ls[idx] for idx in valid_subset_ids]    
    subset_valid_time_step_ls = [valid_time_step_ls[idx] for idx in valid_subset_ids]    
    subset_valid_person_info_ls = [valid_person_info_ls[idx] for idx in valid_subset_ids]    
    subset_valid_label_tensor_ls = [valid_label_tensor_ls[idx] for idx in valid_subset_ids]    

    subset_valid_dataset = LongCOVIDVisitsDataset2(subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max)
    write_to_pickle([visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max], "train_data")
    write_to_pickle([valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max], "valid_data")
    
    write_to_pickle([subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max], "subset_valid_data")

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7f4133c1-cccc-4f64-b292-3055c0ade3a0"),
    add_date_diff_cols=Input(rid="ri.foundry.main.dataset.d3dc0e61-b976-406e-917b-a7e47c925333")
)
def Produce_obs_dataset_2(add_date_diff_cols):
    recent_visits = add_date_diff_cols
    # def produce_dataset(train_valid_split, Long_COVID_Silver_Standard, person_information, recent_visits_w_nlp_notes_2):
    print("start")

    write_to_pickle([None, None], "sample_data")
    # unique_person_ids = obs_latent_sequence.select("person_id").distinct()
    
    unique_person_ids = list(recent_visits.select(F.col('person_id')).distinct().toPandas()['person_id'])

    print("unique person ids::", unique_person_ids[0:10])

    random.shuffle(unique_person_ids)

    print("random unique person ids::", unique_person_ids[0:10])

    train_person_ids = unique_person_ids[0:int(len(unique_person_ids)*0.9)]
    valid_person_ids = unique_person_ids[int(len(unique_person_ids)*0.9):]

    # spark.createDataFrame(data=dept, schema = deptColumns)
    # write_to_pickle([torch.zeros(10), torch.ones(10)], "sample_data")
    # # First get the splitted person ids
    # train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    # valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")
    train_recent_visits = recent_visits.filter(recent_visits["person_id"].isin(train_person_ids))
    valid_recent_visits = recent_visits.filter(recent_visits["person_id"].isin(valid_person_ids))
    train_labels = train_recent_visits.select(F.col("person_id"), F.col("outcome")).distinct()
    valid_labels = valid_recent_visits.select(F.col("person_id"), F.col("outcome")).distinct()
    train_recent_visits = train_recent_visits.drop(F.col("outcome"))
    valid_recent_visits = valid_recent_visits.drop(F.col("outcome"))
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")

    # train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # print(train_recent_visits.show())

    train_person_info = None#train_person_ids.join(person_information, on="person_id")
    valid_person_info = None#valid_person_ids.join(person_information, on="person_id")

    print("start pre-processing!!!")
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits(train_person_ids, None, train_recent_visits.toPandas(), train_labels.toPandas(), setup="both", start_col_id = 2, end_col_id=-2, label_col_name="outcome")

    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits(valid_person_ids, None, valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both", start_col_id = 2, end_col_id=-2, label_col_name="outcome")
    print("finish pre-processing!!!")

    visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)

    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    valid_subset_ids = list(range(10))

    subset_valid_visit_tensor_ls = [valid_visit_tensor_ls[idx] for idx in valid_subset_ids]    
    subset_valid_mask_ls = [valid_mask_ls[idx] for idx in valid_subset_ids]    
    subset_valid_time_step_ls = [valid_time_step_ls[idx] for idx in valid_subset_ids]    
    if valid_person_info_ls is not None:
        subset_valid_person_info_ls = [valid_person_info_ls[idx] for idx in valid_subset_ids]    
    else:
        subset_valid_person_info_ls = None
    subset_valid_label_tensor_ls = [valid_label_tensor_ls[idx] for idx in valid_subset_ids]    

    # subset_valid_dataset = LongCOVIDVisitsDataset2(subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max)
    write_to_pickle([visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max], "train_data")
    write_to_pickle([valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max], "valid_data")
    
    write_to_pickle([subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max], "subset_valid_data")

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.26c31342-fc7f-43b8-a3e5-5b115e528bd4"),
    add_date_diff_cols=Input(rid="ri.foundry.main.dataset.d3dc0e61-b976-406e-917b-a7e47c925333"),
    top_k_concepts_data=Input(rid="ri.foundry.main.dataset.7b277d99-e39e-4a5f-9058-4e6f65fa7f58")
)
def Produce_obs_dataset_with_static_feature(add_date_diff_cols, top_k_concepts_data):
    recent_visits = add_date_diff_cols
    # def produce_dataset(train_valid_split, Long_COVID_Silver_Standard, person_information, recent_visits_w_nlp_notes_2):
    print("start")
    # unique_person_ids = obs_latent_sequence.select("person_id").distinct()
    obs_latent = top_k_concepts_data
    unique_person_ids = list(recent_visits.select(F.col('person_id')).distinct().toPandas()['person_id'])

    empty_person_ids = list(recent_visits.filter(F.col("visit_date").isNull()).select(F.col("person_id")).distinct().toPandas()['person_id'])
    print("empty person ids::", unique_person_ids[0:10])

    print("empty person id count::", len(unique_person_ids))

    unique_person_ids = [idx for idx in unique_person_ids if idx not in empty_person_ids]

    print("unique person id count::", len(unique_person_ids))

    print("unique person ids::", unique_person_ids[0:10])

    random.shuffle(unique_person_ids)

    print("random unique person ids::", unique_person_ids[0:10])

    train_person_ids = unique_person_ids[0:int(len(unique_person_ids)*0.9)]
    valid_person_ids = unique_person_ids[int(len(unique_person_ids)*0.9):]

    # spark.createDataFrame(data=dept, schema = deptColumns)
    # write_to_pickle([torch.zeros(10), torch.ones(10)], "sample_data")
    # # First get the splitted person ids
    # train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    # valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")
    train_recent_visits = recent_visits.filter(recent_visits["person_id"].isin(train_person_ids))
    valid_recent_visits = recent_visits.filter(recent_visits["person_id"].isin(valid_person_ids))
    train_labels = train_recent_visits.select(F.col("person_id"), F.col("outcome")).distinct()
    valid_labels = valid_recent_visits.select(F.col("person_id"), F.col("outcome")).distinct()
    train_recent_visits = train_recent_visits.drop(F.col("outcome"))
    valid_recent_visits = valid_recent_visits.drop(F.col("outcome"))
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")

    # train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # print(train_recent_visits.show())

    train_person_info = obs_latent.filter(obs_latent["person_id"].isin(train_person_ids))
    valid_person_info = obs_latent.filter(obs_latent["person_id"].isin(valid_person_ids))
    # train_person_info = train_person_ids.join(person_information, on="person_id")
    # valid_person_info = valid_person_ids.join(person_information, on="person_id")

    print("start pre-processing!!!")
    print(train_person_info)
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits2(train_person_ids, train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both", start_col_id = 2, end_col_id=-2, label_col_name="outcome")

    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits2(valid_person_ids, valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both", start_col_id = 2, end_col_id=-2, label_col_name="outcome")
    print("finish pre-processing!!!")

    visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)

    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    valid_subset_ids = list(range(10))

    subset_valid_visit_tensor_ls = [valid_visit_tensor_ls[idx] for idx in valid_subset_ids]    
    subset_valid_mask_ls = [valid_mask_ls[idx] for idx in valid_subset_ids]    
    subset_valid_time_step_ls = [valid_time_step_ls[idx] for idx in valid_subset_ids]    
    subset_valid_person_info_ls = [valid_person_info_ls[idx] for idx in valid_subset_ids]    
    subset_valid_label_tensor_ls = [valid_label_tensor_ls[idx] for idx in valid_subset_ids]    

    # subset_valid_dataset = LongCOVIDVisitsDataset2(subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max)
    write_to_pickle([visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max], "train_data")
    write_to_pickle([valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max], "valid_data")
    
    write_to_pickle([subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max], "subset_valid_data")

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.051f4281-a989-4f89-85d0-d641e2afe2b0"),
    add_date_diff_cols_2=Input(rid="ri.foundry.main.dataset.1f76df35-5d2f-46f1-8197-d59658d15475"),
    get_train_valid_partition=Input(rid="ri.foundry.main.dataset.1c438e0a-6066-41ff-b7a7-34352cf60ec5"),
    top_k_concepts_data=Input(rid="ri.foundry.main.dataset.7b277d99-e39e-4a5f-9058-4e6f65fa7f58")
)
def Produce_obs_dataset_with_static_feature_2(add_date_diff_cols_2, top_k_concepts_data, get_train_valid_partition):
    recent_visits = add_date_diff_cols_2
    # def produce_dataset(train_valid_split, Long_COVID_Silver_Standard, person_information, recent_visits_w_nlp_notes_2):
    print("start")
    # unique_person_ids = obs_latent_sequence.select("person_id").distinct()
    obs_latent = top_k_concepts_data
    # unique_person_ids = list(recent_visits.select(F.col('person_id')).distinct().toPandas()['person_id'])

    # empty_person_ids = list(recent_visits.filter(F.col("visit_date").isNull()).select(F.col("person_id")).distinct().toPandas()['person_id'])
    # print("empty person ids::", unique_person_ids[0:10])

    # print("empty person id count::", len(unique_person_ids))

    # unique_person_ids = [idx for idx in unique_person_ids if idx not in empty_person_ids]

    # print("unique person id count::", len(unique_person_ids))

    # print("unique person ids::", unique_person_ids[0:10])

    # random.shuffle(unique_person_ids)

    # print("random unique person ids::", unique_person_ids[0:10])

    # train_person_ids = unique_person_ids[0:int(len(unique_person_ids)*0.9)]
    # valid_person_ids = unique_person_ids[int(len(unique_person_ids)*0.9):]
    train_person_ids = read_from_pickle(get_train_valid_partition, "train_person_ids.pickle")
    valid_person_ids = read_from_pickle(get_train_valid_partition, "test_person_ids.pickle")

    # spark.createDataFrame(data=dept, schema = deptColumns)
    # write_to_pickle([torch.zeros(10), torch.ones(10)], "sample_data")
    # # First get the splitted person ids
    # train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    # valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")
    train_recent_visits = recent_visits.filter(recent_visits["person_id"].isin(train_person_ids))
    valid_recent_visits = recent_visits.filter(recent_visits["person_id"].isin(valid_person_ids))
    train_labels = train_recent_visits.select(F.col("person_id"), F.col("outcome")).distinct()
    valid_labels = valid_recent_visits.select(F.col("person_id"), F.col("outcome")).distinct()
    train_recent_visits = train_recent_visits.drop(F.col("outcome"))
    valid_recent_visits = valid_recent_visits.drop(F.col("outcome"))
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")

    # train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # print(train_recent_visits.show())

    train_person_info = obs_latent.filter(obs_latent["person_id"].isin(train_person_ids))
    valid_person_info = obs_latent.filter(obs_latent["person_id"].isin(valid_person_ids))
    # train_person_info = train_person_ids.join(person_information, on="person_id")
    # valid_person_info = valid_person_ids.join(person_information, on="person_id")

    print("start pre-processing!!!")
    print(train_person_info)
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits2(train_person_ids, train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both", start_col_id = 2, end_col_id=-2, label_col_name="outcome")

    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits2(valid_person_ids, valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both", start_col_id = 2, end_col_id=-2, label_col_name="outcome")
    print("finish pre-processing!!!")

    visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)

    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    valid_subset_ids = list(range(10))

    subset_valid_visit_tensor_ls = [valid_visit_tensor_ls[idx] for idx in valid_subset_ids]    
    subset_valid_mask_ls = [valid_mask_ls[idx] for idx in valid_subset_ids]    
    subset_valid_time_step_ls = [valid_time_step_ls[idx] for idx in valid_subset_ids]    
    subset_valid_person_info_ls = [valid_person_info_ls[idx] for idx in valid_subset_ids]    
    subset_valid_label_tensor_ls = [valid_label_tensor_ls[idx] for idx in valid_subset_ids]    

    # subset_valid_dataset = LongCOVIDVisitsDataset2(subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max)
    write_to_pickle([visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max], "train_data")
    write_to_pickle([valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max], "valid_data")
    
    write_to_pickle([subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max], "subset_valid_data")

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.17357ea0-3d3e-452d-b4d8-fef5e70d65f4"),
    add_date_diff_cols_2_200_400=Input(rid="ri.foundry.main.dataset.e6675c90-881b-4eba-957f-072505753517"),
    get_train_valid_partition=Input(rid="ri.foundry.main.dataset.1c438e0a-6066-41ff-b7a7-34352cf60ec5"),
    top_k_concepts_data=Input(rid="ri.foundry.main.dataset.7b277d99-e39e-4a5f-9058-4e6f65fa7f58")
)
def Produce_obs_dataset_with_static_feature_2_200_400(top_k_concepts_data, add_date_diff_cols_2_200_400, get_train_valid_partition):
    recent_visits = add_date_diff_cols_2_200_400
    # def produce_dataset(train_valid_split, Long_COVID_Silver_Standard, person_information, recent_visits_w_nlp_notes_2):
    print("start")
    # unique_person_ids = obs_latent_sequence.select("person_id").distinct()
    obs_latent = top_k_concepts_data
    # unique_person_ids = list(recent_visits.select(F.col('person_id')).distinct().toPandas()['person_id'])

    # empty_person_ids = list(recent_visits.filter(F.col("visit_date").isNull()).select(F.col("person_id")).distinct().toPandas()['person_id'])
    # print("empty person ids::", unique_person_ids[0:10])

    # print("empty person id count::", len(unique_person_ids))

    # unique_person_ids = [idx for idx in unique_person_ids if idx not in empty_person_ids]

    # print("unique person id count::", len(unique_person_ids))

    # print("unique person ids::", unique_person_ids[0:10])

    # random.shuffle(unique_person_ids)

    # print("random unique person ids::", unique_person_ids[0:10])

    # train_person_ids = unique_person_ids[0:int(len(unique_person_ids)*0.9)]
    # valid_person_ids = unique_person_ids[int(len(unique_person_ids)*0.9):]

    # spark.createDataFrame(data=dept, schema = deptColumns)
    # write_to_pickle([torch.zeros(10), torch.ones(10)], "sample_data")
    # # First get the splitted person ids
    # train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    # valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")
    train_person_ids = read_from_pickle(get_train_valid_partition, "train_person_ids.pickle")
    valid_person_ids = read_from_pickle(get_train_valid_partition, "test_person_ids.pickle")

    train_recent_visits = recent_visits.filter(recent_visits["person_id"].isin(train_person_ids))
    valid_recent_visits = recent_visits.filter(recent_visits["person_id"].isin(valid_person_ids))
    train_labels = train_recent_visits.select(F.col("person_id"), F.col("outcome")).distinct()
    valid_labels = valid_recent_visits.select(F.col("person_id"), F.col("outcome")).distinct()
    train_recent_visits = train_recent_visits.drop(F.col("outcome"))
    valid_recent_visits = valid_recent_visits.drop(F.col("outcome"))
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")

    # train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # print(train_recent_visits.show())

    train_person_info = obs_latent.filter(obs_latent["person_id"].isin(train_person_ids))
    valid_person_info = obs_latent.filter(obs_latent["person_id"].isin(valid_person_ids))
    # train_person_info = train_person_ids.join(person_information, on="person_id")
    # valid_person_info = valid_person_ids.join(person_information, on="person_id")

    print("start pre-processing!!!")
    print(train_person_info)
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits2(train_person_ids, train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both", start_col_id = 2, end_col_id=-2, label_col_name="outcome")

    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits2(valid_person_ids, valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both", start_col_id = 2, end_col_id=-2, label_col_name="outcome")
    print("finish pre-processing!!!")

    visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)

    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    valid_subset_ids = list(range(10))

    subset_valid_visit_tensor_ls = [valid_visit_tensor_ls[idx] for idx in valid_subset_ids]    
    subset_valid_mask_ls = [valid_mask_ls[idx] for idx in valid_subset_ids]    
    subset_valid_time_step_ls = [valid_time_step_ls[idx] for idx in valid_subset_ids]    
    subset_valid_person_info_ls = [valid_person_info_ls[idx] for idx in valid_subset_ids]    
    subset_valid_label_tensor_ls = [valid_label_tensor_ls[idx] for idx in valid_subset_ids]    

    # subset_valid_dataset = LongCOVIDVisitsDataset2(subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max)
    write_to_pickle([visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max], "train_data")
    write_to_pickle([valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max], "valid_data")
    
    write_to_pickle([subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max], "subset_valid_data")

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.27689177-ef6e-42b4-9673-d8052f7f0f92"),
    add_date_diff_cols_2_400_600=Input(rid="ri.foundry.main.dataset.c7facd44-e20f-4bef-b413-d65a7feb192d"),
    get_train_valid_partition=Input(rid="ri.foundry.main.dataset.1c438e0a-6066-41ff-b7a7-34352cf60ec5"),
    top_k_concepts_data=Input(rid="ri.foundry.main.dataset.7b277d99-e39e-4a5f-9058-4e6f65fa7f58")
)
def Produce_obs_dataset_with_static_feature_2_400_600(top_k_concepts_data, add_date_diff_cols_2_400_600, get_train_valid_partition):
    recent_visits = add_date_diff_cols_2_400_600
    # def produce_dataset(train_valid_split, Long_COVID_Silver_Standard, person_information, recent_visits_w_nlp_notes_2):
    print("start")
    # unique_person_ids = obs_latent_sequence.select("person_id").distinct()
    obs_latent = top_k_concepts_data
    # unique_person_ids = list(recent_visits.select(F.col('person_id')).distinct().toPandas()['person_id'])

    # empty_person_ids = list(recent_visits.filter(F.col("visit_date").isNull()).select(F.col("person_id")).distinct().toPandas()['person_id'])
    # print("empty person ids::", unique_person_ids[0:10])

    # print("empty person id count::", len(unique_person_ids))

    # unique_person_ids = [idx for idx in unique_person_ids if idx not in empty_person_ids]

    # print("unique person id count::", len(unique_person_ids))

    # print("unique person ids::", unique_person_ids[0:10])

    # random.shuffle(unique_person_ids)

    # print("random unique person ids::", unique_person_ids[0:10])

    # train_person_ids = unique_person_ids[0:int(len(unique_person_ids)*0.9)]
    # valid_person_ids = unique_person_ids[int(len(unique_person_ids)*0.9):]

    # spark.createDataFrame(data=dept, schema = deptColumns)
    # write_to_pickle([torch.zeros(10), torch.ones(10)], "sample_data")
    # # First get the splitted person ids
    # train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    # valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")
    train_person_ids = read_from_pickle(get_train_valid_partition, "train_person_ids.pickle")
    valid_person_ids = read_from_pickle(get_train_valid_partition, "test_person_ids.pickle")

    train_recent_visits = recent_visits.filter(recent_visits["person_id"].isin(train_person_ids))
    valid_recent_visits = recent_visits.filter(recent_visits["person_id"].isin(valid_person_ids))
    train_labels = train_recent_visits.select(F.col("person_id"), F.col("outcome")).distinct()
    valid_labels = valid_recent_visits.select(F.col("person_id"), F.col("outcome")).distinct()
    train_recent_visits = train_recent_visits.drop(F.col("outcome"))
    valid_recent_visits = valid_recent_visits.drop(F.col("outcome"))
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")

    # train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # print(train_recent_visits.show())

    train_person_info = obs_latent.filter(obs_latent["person_id"].isin(train_person_ids))
    valid_person_info = obs_latent.filter(obs_latent["person_id"].isin(valid_person_ids))
    # train_person_info = train_person_ids.join(person_information, on="person_id")
    # valid_person_info = valid_person_ids.join(person_information, on="person_id")

    print("start pre-processing!!!")
    print(train_person_info)
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits2(train_person_ids, train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both", start_col_id = 2, end_col_id=-2, label_col_name="outcome")

    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits2(valid_person_ids, valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both", start_col_id = 2, end_col_id=-2, label_col_name="outcome")
    print("finish pre-processing!!!")

    visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)

    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    valid_subset_ids = list(range(10))

    subset_valid_visit_tensor_ls = [valid_visit_tensor_ls[idx] for idx in valid_subset_ids]    
    subset_valid_mask_ls = [valid_mask_ls[idx] for idx in valid_subset_ids]    
    subset_valid_time_step_ls = [valid_time_step_ls[idx] for idx in valid_subset_ids]    
    subset_valid_person_info_ls = [valid_person_info_ls[idx] for idx in valid_subset_ids]    
    subset_valid_label_tensor_ls = [valid_label_tensor_ls[idx] for idx in valid_subset_ids]    

    # subset_valid_dataset = LongCOVIDVisitsDataset2(subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max)
    write_to_pickle([visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max], "train_data")
    write_to_pickle([valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max], "valid_data")
    
    write_to_pickle([subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max], "subset_valid_data")

@transform_pandas(
    Output(rid="ri.vector.main.execute.0687efb4-b4da-4627-aacd-a29d7c4dcb17"),
    add_date_diff_cols_2_600_800=Input(rid="ri.vector.main.execute.01ce58e8-4b9a-4c5f-91b2-5d1c5e8da5bc"),
    get_train_valid_partition=Input(rid="ri.foundry.main.dataset.1c438e0a-6066-41ff-b7a7-34352cf60ec5"),
    top_k_concepts_data=Input(rid="ri.foundry.main.dataset.7b277d99-e39e-4a5f-9058-4e6f65fa7f58")
)
def Produce_obs_dataset_with_static_feature_2_600_800(top_k_concepts_data, add_date_diff_cols_2_600_800, get_train_valid_partition):
    recent_visits = add_date_diff_cols_2_600_800
    # def produce_dataset(train_valid_split, Long_COVID_Silver_Standard, person_information, recent_visits_w_nlp_notes_2):
    print("start")
    # unique_person_ids = obs_latent_sequence.select("person_id").distinct()
    obs_latent = top_k_concepts_data
    # unique_person_ids = list(recent_visits.select(F.col('person_id')).distinct().toPandas()['person_id'])

    # empty_person_ids = list(recent_visits.filter(F.col("visit_date").isNull()).select(F.col("person_id")).distinct().toPandas()['person_id'])
    # print("empty person ids::", unique_person_ids[0:10])

    # print("empty person id count::", len(unique_person_ids))

    # unique_person_ids = [idx for idx in unique_person_ids if idx not in empty_person_ids]

    # print("unique person id count::", len(unique_person_ids))

    # print("unique person ids::", unique_person_ids[0:10])

    # random.shuffle(unique_person_ids)

    # print("random unique person ids::", unique_person_ids[0:10])

    # train_person_ids = unique_person_ids[0:int(len(unique_person_ids)*0.9)]
    # valid_person_ids = unique_person_ids[int(len(unique_person_ids)*0.9):]

    # spark.createDataFrame(data=dept, schema = deptColumns)
    # write_to_pickle([torch.zeros(10), torch.ones(10)], "sample_data")
    # # First get the splitted person ids
    # train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    # valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")
    train_person_ids = read_from_pickle(get_train_valid_partition, "train_person_ids.pickle")
    valid_person_ids = read_from_pickle(get_train_valid_partition, "test_person_ids.pickle")

    train_recent_visits = recent_visits.filter(recent_visits["person_id"].isin(train_person_ids))
    valid_recent_visits = recent_visits.filter(recent_visits["person_id"].isin(valid_person_ids))
    train_labels = train_recent_visits.select(F.col("person_id"), F.col("outcome")).distinct()
    valid_labels = valid_recent_visits.select(F.col("person_id"), F.col("outcome")).distinct()
    train_recent_visits = train_recent_visits.drop(F.col("outcome"))
    valid_recent_visits = valid_recent_visits.drop(F.col("outcome"))
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")

    # train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # print(train_recent_visits.show())

    train_person_info = obs_latent.filter(obs_latent["person_id"].isin(train_person_ids))
    valid_person_info = obs_latent.filter(obs_latent["person_id"].isin(valid_person_ids))
    # train_person_info = train_person_ids.join(person_information, on="person_id")
    # valid_person_info = valid_person_ids.join(person_information, on="person_id")

    print("start pre-processing!!!")
    print(train_person_info)
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits2(train_person_ids, train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both", start_col_id = 2, end_col_id=-2, label_col_name="outcome")

    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits2(valid_person_ids, valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both", start_col_id = 2, end_col_id=-2, label_col_name="outcome")
    print("finish pre-processing!!!")

    visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)

    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    valid_subset_ids = list(range(10))

    subset_valid_visit_tensor_ls = [valid_visit_tensor_ls[idx] for idx in valid_subset_ids]    
    subset_valid_mask_ls = [valid_mask_ls[idx] for idx in valid_subset_ids]    
    subset_valid_time_step_ls = [valid_time_step_ls[idx] for idx in valid_subset_ids]    
    subset_valid_person_info_ls = [valid_person_info_ls[idx] for idx in valid_subset_ids]    
    subset_valid_label_tensor_ls = [valid_label_tensor_ls[idx] for idx in valid_subset_ids]    

    # subset_valid_dataset = LongCOVIDVisitsDataset2(subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max)
    write_to_pickle([visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max], "train_data")
    write_to_pickle([valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max], "valid_data")
    
    write_to_pickle([subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max], "subset_valid_data")

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d3dc0e61-b976-406e-917b-a7e47c925333"),
    obs_latent_sequence_2=Input(rid="ri.foundry.main.dataset.9738c306-6d58-457d-8fbe-7d1d5a09ef01")
)
from pyspark.sql.functions import col, max as max_, min as min_
from pyspark.sql.functions import datediff
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F 
def add_date_diff_cols(obs_latent_sequence_2):
    recent_visits = obs_latent_sequence_2
    min_person_visit_data = recent_visits.groupBy("person_id").agg(min_("visit_date"))
    print("here")

    # min_person_visit_data = min_person_visit_data.reset_index()
    min_person_visit_data = min_person_visit_data.withColumnRenamed("min(visit_date)", "min_visit_date")
    # min_person_visit_data.rename(columns={"min":"min_visit_date"}, inplace="True")

    # recent_visits = recent_visits.merge(min_person_visit_data)
    recent_visits = recent_visits.join(min_person_visit_data, on = "person_id")
    
    # recent_visits["diff_days"] = (recent_visits["visit_date"] - recent_visits["min_visit_date"]).dt.days.astype('int16')
    recent_visits = recent_visits.withColumn("diff_days", datediff(recent_visits["visit_date"], recent_visits["min_visit_date"]))
    return recent_visits
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.1f76df35-5d2f-46f1-8197-d59658d15475"),
    obs_latent_sequence_0=Input(rid="ri.foundry.main.dataset.171b1464-ba7a-41eb-a191-c026ceaa1ed1")
)
from pyspark.sql.functions import col, max as max_, min as min_
from pyspark.sql.functions import datediff
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F 
def add_date_diff_cols_2(obs_latent_sequence_0):
    recent_visits = obs_latent_sequence_0
    min_person_visit_data = recent_visits.groupBy("person_id").agg(min_("visit_date"))
    print("here")

    # min_person_visit_data = min_person_visit_data.reset_index()
    min_person_visit_data = min_person_visit_data.withColumnRenamed("min(visit_date)", "min_visit_date")
    # min_person_visit_data.rename(columns={"min":"min_visit_date"}, inplace="True")

    # recent_visits = recent_visits.merge(min_person_visit_data)
    recent_visits = recent_visits.join(min_person_visit_data, on = "person_id")
    
    # recent_visits["diff_days"] = (recent_visits["visit_date"] - recent_visits["min_visit_date"]).dt.days.astype('int16')
    recent_visits = recent_visits.withColumn("diff_days", datediff(recent_visits["visit_date"], recent_visits["min_visit_date"]))
    return recent_visits
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e6675c90-881b-4eba-957f-072505753517"),
    obs_latent_sequence_0_200_to_400=Input(rid="ri.foundry.main.dataset.bebb873c-0676-432e-8196-09ec28e09053")
)
from pyspark.sql.functions import col, max as max_, min as min_
from pyspark.sql.functions import datediff
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F 
def add_date_diff_cols_2_200_400(obs_latent_sequence_0_200_to_400):
    recent_visits = obs_latent_sequence_0_200_to_400
    min_person_visit_data = recent_visits.groupBy("person_id").agg(min_("visit_date"))
    print("here")

    # min_person_visit_data = min_person_visit_data.reset_index()
    min_person_visit_data = min_person_visit_data.withColumnRenamed("min(visit_date)", "min_visit_date")
    # min_person_visit_data.rename(columns={"min":"min_visit_date"}, inplace="True")

    # recent_visits = recent_visits.merge(min_person_visit_data)
    recent_visits = recent_visits.join(min_person_visit_data, on = "person_id")
    
    # recent_visits["diff_days"] = (recent_visits["visit_date"] - recent_visits["min_visit_date"]).dt.days.astype('int16')
    recent_visits = recent_visits.withColumn("diff_days", datediff(recent_visits["visit_date"], recent_visits["min_visit_date"]))
    return recent_visits

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c7facd44-e20f-4bef-b413-d65a7feb192d"),
    obs_latent_sequence_0_400_to_600=Input(rid="ri.foundry.main.dataset.87b291fb-c532-44ec-9427-89eb450d4493")
)
from pyspark.sql.functions import col, max as max_, min as min_
from pyspark.sql.functions import datediff
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F 
def add_date_diff_cols_2_400_600(obs_latent_sequence_0_400_to_600):
    recent_visits = obs_latent_sequence_0_400_to_600
    min_person_visit_data = recent_visits.groupBy("person_id").agg(min_("visit_date"))
    print("here")

    # min_person_visit_data = min_person_visit_data.reset_index()
    min_person_visit_data = min_person_visit_data.withColumnRenamed("min(visit_date)", "min_visit_date")
    # min_person_visit_data.rename(columns={"min":"min_visit_date"}, inplace="True")

    # recent_visits = recent_visits.merge(min_person_visit_data)
    recent_visits = recent_visits.join(min_person_visit_data, on = "person_id")
    
    # recent_visits["diff_days"] = (recent_visits["visit_date"] - recent_visits["min_visit_date"]).dt.days.astype('int16')
    recent_visits = recent_visits.withColumn("diff_days", datediff(recent_visits["visit_date"], recent_visits["min_visit_date"]))
    return recent_visits

@transform_pandas(
    Output(rid="ri.vector.main.execute.01ce58e8-4b9a-4c5f-91b2-5d1c5e8da5bc"),
    obs_latent_sequence_0_600_to_800=Input(rid="ri.foundry.main.dataset.c06cda84-379a-4c67-8111-8135014d0380")
)
from pyspark.sql.functions import col, max as max_, min as min_
from pyspark.sql.functions import datediff
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F 
def add_date_diff_cols_2_600_800(obs_latent_sequence_0_600_to_800):
    recent_visits = obs_latent_sequence_0_600_to_800
    min_person_visit_data = recent_visits.groupBy("person_id").agg(min_("visit_date"))
    print("here")

    # min_person_visit_data = min_person_visit_data.reset_index()
    min_person_visit_data = min_person_visit_data.withColumnRenamed("min(visit_date)", "min_visit_date")
    # min_person_visit_data.rename(columns={"min":"min_visit_date"}, inplace="True")

    # recent_visits = recent_visits.merge(min_person_visit_data)
    recent_visits = recent_visits.join(min_person_visit_data, on = "person_id")
    
    # recent_visits["diff_days"] = (recent_visits["visit_date"] - recent_visits["min_visit_date"]).dt.days.astype('int16')
    recent_visits = recent_visits.withColumn("diff_days", datediff(recent_visits["visit_date"], recent_visits["min_visit_date"]))
    return recent_visits

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
    # df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_max_observed_or_calculated', 'postal_code', 'age')])
    
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
    #deaths_df = everyone_patient_deaths.select('person_id','patient_death')
    df = all_patients_visit_day_facts_table_de_id_testing.drop('patient_death_at_visit', 'during_macrovisit_hospitalization')
    
    df2 = all_patients_visit_day_facts_table_de_id_testing.select('person_id', 'visit_date', 'Oxygen_saturation').where(all_patients_visit_day_facts_table_de_id_testing.Oxygen_saturation>0)    
    
    df3 = all_patients_visit_day_facts_table_de_id_testing.select('person_id', 'visit_date', 'blood_sodium').where(all_patients_visit_day_facts_table_de_id_testing.blood_sodium>0) 
    
    df4 = all_patients_visit_day_facts_table_de_id_testing.select('person_id', 'visit_date', 'blood_hemoglobin').where(all_patients_visit_day_facts_table_de_id_testing.blood_hemoglobin>0) 

    df5 = all_patients_visit_day_facts_table_de_id_testing.select('person_id', 'visit_date', 'blood_Creatinine').where(all_patients_visit_day_facts_table_de_id_testing.blood_Creatinine>0)

    df6 = all_patients_visit_day_facts_table_de_id_testing.select('person_id', 'visit_date', 'blood_UreaNitrogen').where(all_patients_visit_day_facts_table_de_id_testing.blood_UreaNitrogen>0)

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
    df = everyone_cohort_de_id_testing.join(df, 'person_id','left')

    #final fill of null in non-continuous variables with 0
    # df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_max_observed_or_calculated', 'postal_code', 'age')])
    
    df = df.distinct()

    

    return df
        
#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ace57213-685a-4f18-a157-2b02b41086be"),
    everyone_conditions_of_interest=Input(rid="ri.foundry.main.dataset.514f3fe8-7565-4701-8982-174b43937006"),
    everyone_devices_of_interest=Input(rid="ri.foundry.main.dataset.15ddf371-0d59-4397-9bee-866c880620cf"),
    everyone_drugs_of_interest=Input(rid="ri.foundry.main.dataset.32bad30b-9322-4e6d-8a88-ab5133e98543"),
    everyone_measurements_of_interest=Input(rid="ri.foundry.main.dataset.99e1cf7c-8848-4a3c-8f26-5cc7499311da"),
    everyone_observations_of_interest=Input(rid="ri.foundry.main.dataset.d2eefa83-105e-404c-9e21-5475e1e1110c"),
    everyone_procedures_of_interest=Input(rid="ri.foundry.main.dataset.ff38921a-cc27-4c35-9a09-9a7ccced1ad6"),
    everyone_vaccines_of_interest=Input(rid="ri.foundry.main.dataset.202ec093-e569-4af8-897a-ab8d2c4325c0"),
    microvisits_to_macrovisits=Input(rid="ri.foundry.main.dataset.d77a701f-34df-48a1-a71c-b28112a07ffa"),
    person_top_nlp_symptom=Input(rid="ri.foundry.main.dataset.73f9d829-203f-4e2d-88d2-0d168503b0b1")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 7/8/22
#Description - All facts collected in the previous steps are combined in this cohort_all_facts_table on the basis of unique visit days for each patient. Indicators are created for the presence or absence of events, medications, conditions, measurements, device exposures, observations, procedures, and outcomes.  It also creates an indicator for whether the visit date where a fact was noted occurred during any hospitalization. This table is useful if the analyst needs to use actual dates of events as it provides more detail than the final patient-level table.  Use the max and min functions to find the first and last occurrences of any events.

def all_patients_visit_day_facts_table_de_id(everyone_conditions_of_interest, everyone_measurements_of_interest, everyone_procedures_of_interest, everyone_observations_of_interest, everyone_drugs_of_interest, everyone_devices_of_interest, microvisits_to_macrovisits, everyone_vaccines_of_interest, person_top_nlp_symptom):

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
    
    # df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_rounded')])
   
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
    # df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_rounded')])

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
    microvisits_to_macrovisits_testing_copy=Input(rid="ri.foundry.main.dataset.05de4355-6100-463e-930a-0e9d3c8a8baa")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 7/8/22
#Description - All facts collected in the previous steps are combined in this cohort_all_facts_table on the basis of unique visit days for each patient. Indicators are created for the presence or absence of events, medications, conditions, measurements, device exposures, observations, procedures, and outcomes.  It also creates an indicator for whether the visit date where a fact was noted occurred during any hospitalization. This table is useful if the analyst needs to use actual dates of events as it provides more detail than the final patient-level table.  Use the max and min functions to find the first and last occurrences of any events.

def all_patients_visit_day_facts_table_de_id_testing(everyone_conditions_of_interest_testing, everyone_measurements_of_interest_testing, everyone_procedures_of_interest_testing, everyone_observations_of_interest_testing, everyone_drugs_of_interest_testing, everyone_devices_of_interest_testing, everyone_vaccines_of_interest_testing, microvisits_to_macrovisits_testing_copy):
    macrovisits_df = microvisits_to_macrovisits_testing_copy
    vaccines_df = everyone_vaccines_of_interest_testing
    procedures_df = everyone_procedures_of_interest_testing
    devices_df = everyone_devices_of_interest_testing
    observations_df = everyone_observations_of_interest_testing
    conditions_df = everyone_conditions_of_interest_testing
    drugs_df = everyone_drugs_of_interest_testing
    measurements_df = everyone_measurements_of_interest_testing

    df = macrovisits_df.select('person_id','visit_start_date').withColumnRenamed('visit_start_date','visit_date')
    df = df.join(vaccines_df, on=list(set(df.columns)&set(vaccines_df.columns)), how='outer')
    df = df.join(procedures_df, on=list(set(df.columns)&set(procedures_df.columns)), how='outer')
    df = df.join(devices_df, on=list(set(df.columns)&set(devices_df.columns)), how='outer')
    df = df.join(observations_df, on=list(set(df.columns)&set(observations_df.columns)), how='outer')
    df = df.join(conditions_df, on=list(set(df.columns)&set(conditions_df.columns)), how='outer')
    df = df.join(drugs_df, on=list(set(df.columns)&set(drugs_df.columns)), how='outer')
    df = df.join(measurements_df, on=list(set(df.columns)&set(measurements_df.columns)), how='outer')
    # df = df.join(nlp_symptom_df, on=list(set(df.columns)&set(nlp_symptom_df.columns)), how='outer')
    
    # df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_rounded')])
   
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
    # df = df.na.fill(value=0, subset = [col for col in df.columns if col not in ('BMI_rounded')])

    for col in sorted(df.columns):
        print(col)

    return df
    
#################################################
## Global imports and functions included below ##
#################################################
    
#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ce8c17fb-63cd-41bd-b9c8-9bf54e5091da")
)
from pyspark.sql.types import *
def broad_related_concepts():
    schema = StructType([StructField("codeset_id", StringType(), True), StructField("concept_set_name", StringType(), True)])
    return spark.createDataFrame([["907542870","cough-broad"],["858501962","menstrual-cycle-broad"],["964465007","anxiety-broad"],["409823731","diabetes-broad"],["281429838","fatigue-broad"],["872295997","malaise-broad"],["934501626","fever-broad"],["963124232","dyspnea-broad"],["312841058","chest-pain-broad"],["452556220","palpitations-broad"],["233877953","brain-fog-broad"],["279146925","headache-broad"],["888708919","insomnia-broad"],["663952776","lightheadedness-broad"],["612580291","tingling-broad"],["185979799","anosmia-broad"],["528085678","depression-broad"],["345394441","diarrhea-broad"],["343711928","stomach-pain-broad"],["227682624","joint-muscle-pain-broad"],["534110802","rash-broad"]], schema=schema)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9b4ab5dd-d57f-416e-a2ee-75611c8a12bc"),
    drug_table_analysis_1=Input(rid="ri.foundry.main.dataset.2a5480ef-7699-4f0c-bf5c-2a0f8401224d"),
    study_misclassified=Input(rid="ri.foundry.main.dataset.235be874-5669-4c42-9ae3-3e6d37b645e1")
)
def combined_study( study_misclassified, drug_table_analysis_1):
    TABLE = drug_table_analysis_1
    THRESH = 0.66
    cond = [TABLE.concept_name == study_misclassified.concept_name, TABLE.pos > THRESH, study_misclassified.pos > THRESH]
    df = study_misclassified.join(TABLE, cond).select(study_misclassified.concept_name,
    study_misclassified.pos.alias("misc_pos"),
    TABLE.pos.alias("gen_pos"),
    study_misclassified.people.alias("misc_people"),
    TABLE.people.alias("gen_people"),
    )
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a32b3d71-226c-4347-aaed-2c4900e2f4fb"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2"),
    condition_occurrence_testing=Input(rid="ri.foundry.main.dataset.3e01546f-f110-4c67-a6db-9063d2939a74")
)
def condition_occurrence_testing_copy(condition_occurrence_testing, condition_occurrence):
    return condition_occurrence_testing if LOAD_TEST == 1 else condition_occurrence
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bcbf4137-1508-42b5-bb05-631492b8d3b9"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2")
)
def condition_table_analysis(condition_occurrence, Long_COVID_Silver_Standard):
    TABLE = condition_occurrence
    CONCEPT_NAME_COL = "condition_concept_name"
    l, h = 0, 750

    label = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"]))
    TABLE = TABLE.join(label, "person_id").select(F.col("person_id"), F.col(CONCEPT_NAME_COL), F.col("outcome")).filter(F.col(CONCEPT_NAME_COL) != "No matching concept")
    distinct = TABLE.groupBy(CONCEPT_NAME_COL).count().orderBy("count", ascending=False).limit(h).select(F.col(CONCEPT_NAME_COL)).toPandas()[CONCEPT_NAME_COL]
    
    pos, count, ppl, ppl_pos = [], [], [], []
    cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
    print(len(distinct))
    t = time.time()
    for cname in distinct:
        f = TABLE.agg(
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname)),
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1)),
            F.countDistinct(F.when(F.col(CONCEPT_NAME_COL) == cname, F.col("person_id")).otherwise(None)),
            F.countDistinct(F.when(((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1)), F.col("person_id")).otherwise(None)),
        ).collect()
        one_count = f[0][1]
        size = f[0][0]
        people_count =f[0][2]
        people_one = f[0][3]
        pos.append(one_count/size)
        count.append(size)
        ppl.append(people_count)
        ppl_pos.append(people_one/people_count)
    print(time.time() - t)
    r = pd.DataFrame(list(zip(distinct,pos, count, ppl, ppl_pos)), columns=["concept_name","encounter_pos", "encounter_count", "people_count", "people_pos"])
    r['domain'] = CONCEPT_NAME_COL
    r["heuristic_count"] = r.apply(lambda row: row["encounter_pos"] * row["encounter_count"] if row["encounter_pos"] > 0.7 else 0, axis=1)
    r["heuristic_person"] = r.apply(lambda row: row["people_pos"] * row["people_count"] if row["people_pos"] > 0.7 else 0, axis=1)
    r =r[["concept_name", 'domain', 'encounter_pos','people_pos','encounter_count', 'people_count','heuristic_count','heuristic_person']]
    
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
        ["40762523","Fall risk total [Morse Fall Scale]", "fall_risk_assessment_penn"],
        ["43018325", "Performance of Urinary Filtration, Continuous, Greater than 18 hours Per Day", "urinary_filtration_penn"],
        ["3661408", "Pneumonia caused by SARS-CoV-2", "pneumonia_penn"],
        ["42538827", "Uses contraception", "contraception_penn"],
        ["43018325", "Performance of Urinary Filtration, Continuous, Greater than 18 hours Per Day", "urinary_filtration_penn"],
        ["21494995", "Pain assessment [Interpretation]", "pain_assessment_penn"],
        ["3035482", "Pain duration - Reported", "pain_duration_penn"],
        ["4271661", "Characteristic of pain", "characteristic_pain_penn"],
        ["1367500", "losartan", "losartan_penn"],
        ["903963", "triamcinolone", "triamcinolone_penn"],
        ["1336926", "tadalafil", "tadalafil_penn"],
        ["1367571", "heparin", "heparin_penn"],
        ["1112921", "ipratropium", "ipratropium_penn"],
        ["798875", "clonazepam 0.5 MG Oral Tablet", "clonazepam_penn"],
        ["713823", "ropinirole", "ropinirole_penn"],
        ["19045045", "ergocalciferol", "ergocalciferol_penn"],
        ["1154343", "albuterol", "albuterol_penn"],
        ["19095164", "cholecalciferol", "cholecalciferol_penn"],
        ["1149380", "fluticasone", "fluticasone_penn"],
        ["753626", "propofol", "propofol_penn"],
        ["950637", "tacrolimus", "tacrolimus_penn"],
        ["975125", "hydrocortisone", "hydrocortisone_penn"],
        ["1308738", "vitamin B12", "B12_penn"],
        ["1136601", "benzonatate", "benzonatate_penn"],
        ["1192218","levalbuterol","levalbuterol_penn"],
        ["1545958", "atorvastatin", "atorvastatin_penn"],
        ["924566","tamsulosin","tamsulosin_penn"],
        ["2108253","Collection of blood specimen from a completely implantable venous access device", "venous_implant_penn"],
        ["74582","Primary malignant neoplasm of rectum", "neoplasm_penn"],
        ["4218813","Third trimester pregnancy","pregnant_penn"],
        ["19003999","mycophenolate mofetil", "mofetil_penn"],
        ["950637", "tacrolimus", "tacrolimus_penn"],
        ["1551860","pravastatin","pravastatin_penn"],
        ["1501700","levothyroxine","levothyroxine_penn"],
        ["1149380","fluticasone","fluticasone_penn"],
        ["2514406","Initial hospital care, per day, for the evaluation and management of a patient, which requires these 3 key components: A comprehensive history; A comprehensive examination; and Medical decision making of high complexity. Counseling and/or coordination of care with other physicians, other qualified health care professionals, or agencies are provided consistent with the nature of the problem(s) and the patient's and/or family's needs. Usually, the problem(s) requiring admission are of high severity. Typically, 70 minutes are spent at the bedside and on the patient's hospital floor or unit.", "hospitalized_penn"],
        ["2514527","Periodic comprehensive preventive medicine reevaluation and management of an individual including an age and gender appropriate history, examination, counseling/anticipatory guidance/risk factor reduction interventions, and the ordering of laboratory/diagnostic procedures, established patient; 18-39 years","periodic_checkup_penn"],
        ["19095164","cholecalciferol","cholecalciferol_penn"],
        ["923645","omeprazole","omeprazole_penn"],
        ["1136601","benzonatate","benzonatate_penn"],
        ["2787823","Assistance with Respiratory Ventilation, Less than 24 Consecutive Hours, Continuous Positive Airway Pressure","ventilator_penn"],
        ["2788038","Respiratory Ventilation, Greater than 96 Consecutive Hours", "ventilator_penn"],
        ["1781162","Assistance with Respiratory Ventilation, Greater than 96 Consecutive Hours, High Nasal Flow/Velocity", "ventilator_penn"],
        ["1781160", "Assistance with Respiratory Ventilation, Less than 24 Consecutive Hours, High Nasal Flow/Velocity", "ventilator_penn"],
        ["2788037", "Respiratory Ventilation, 24-96 Consecutive Hours", "ventilator_penn"],
        ["4230167", "Artificial respiration", "ventilator_penn"],
        ["2745444", "Insertion of Endotracheal Airway into Trachea, Via Natural or Artificial Opening", "tracheostomy_penn"],
        ["2106562", "Tracheostomy, planned (separate procedure)", "tracheostomy_penn"],
        ["2786229", "Introduction of Anti-inflammatory into Peripheral Vein, Percutaneous Approach", "antiinflammatory_penn"],
        ["2787749", "Introduction of Anti-inflammatory into Mouth and Pharynx, External Approach", "antiinflammatory_penn"],
        ["1332418","amlodipine","amlodipine_penn"],
        ["435788","Disorder of phosphorus metabolism","metabolism_disorder_penn"],
        ["2106281", "Most recent systolic blood pressure less than 130 mm Hg (DM), (HTN, CKD, CAD)", "bloodpressure_penn"],
        ["257907","Disorder of lung","lungdisorder_penn"],
        ["1567198","insulin aspart, human", "insulin_penn"],
        ["739138", "sertraline", "sertraline_penn"]
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
    df.loc[len(df.index)] = ['fall_risk_assessment_penn', 'FALL_RISK', 'procedure, observation']
    df.loc[len(df.index)] = ['urinary_filtration_penn', 'URINARY_FILTRATION', 'procedure']
    df.loc[len(df.index)] = ['pneumonia_penn', 'pneumonia', 'condition']
    df.loc[len(df.index)] = ['contraception_penn', 'contraception', 'condition']

    df.loc[len(df.index)] = ["albuterol_penn", "ALBUTEROL", "drug"]
    df.loc[len(df.index)] = ["cholecalciferol_penn", "CHOLECALCIFEROL", "drug"]
    df.loc[len(df.index)] = ["propofol_penn", "PROPOFOL", "drug"]
    df.loc[len(df.index)] = ["tacrolimus_penn", "TACROLIMUS", "drug"]
    df.loc[len(df.index)] = ["hydrocortisone_penn", "HYDROCORTISONE", "drug"]
    df.loc[len(df.index)] = ["B12_penn", "VITAMIN_B12", "drug"]
    df.loc[len(df.index)] = ["benzonatate_penn", "BENZONATATE", "drug"]
    
    df.loc[len(df.index)] = ['pain_assessment_penn', 'PAIN_ASSESSMENT', 'observation']
    df.loc[len(df.index)] = ["pain_duration_penn", "PAIN_DURATION", "observation"]
    df.loc[len(df.index)] = ["characteristic_pain_penn", "PAIN_CHARACTERISTIC", "observation"]
    df.loc[len(df.index)] = ["losartan_penn", "LOSARTAN", "drug"]
    df.loc[len(df.index)] = ["triamcinolone_penn", "TRIAMCINOLONE", "drug"]
    df.loc[len(df.index)] = ["tadalafil_penn", "TADALAFIL", "drug"]
    df.loc[len(df.index)] = ["heparin_penn", "HEPARIN", "drug"]
    df.loc[len(df.index)] = ["ipratropium_penn", "IPRATROPIUM", "drug"]
    df.loc[len(df.index)] = ["clonazepam_penn", "CLONEAZEPAM", "drug"]
    df.loc[len(df.index)] = ["ropinirole_penn", "ROPINIROLE", "drug"]
    #df.loc[len(df.index)] = ["ergocalciferol_penn", "ERGOCALCIFEROL", "drug"]
    df.loc[len(df.index)] = ["levalbuterol_penn", "LEVALBUTEROL", "drug"]
    df.loc[len(df.index)] = ["atorvastatin_penn", "ATORVASTATIN", "drug"]
    df.loc[len(df.index)] = ["tamsulosin_penn", "TAMSULOSIN", "drug"]
    df.loc[len(df.index)] = ["venous_implant_penn", "VENOUSIMPLANT", "procedure"]
    df.loc[len(df.index)] = ["pregnant_penn", "PREGNANT", "condition"]
    df.loc[len(df.index)] = ["mofetil_penn", "MOFETIL", "drug"]
    #df.loc[len(df.index)] = ["tacrolimus_penn", "TACROLIMUS", "drug"]
    #df.loc[len(df.index)] = ["pravastatin_penn", "PRAVASTATIN", "drug"]
    df.loc[len(df.index)] = ["levothyroxine_penn", "LEVOTHYROXINE", "drug"]
    df.loc[len(df.index)] = ["fluticasone_penn", "FLUTICASONE", "drug"]
    df.loc[len(df.index)] = ["hospitalized_penn", "HOSPITALIZED", "procedure"]
    df.loc[len(df.index)] = ["periodic_checkup_penn", "PERIODICCHECKUP", "procedure"]
    df.loc[len(df.index)] = ["cholecalciferol_penn", "CHOLECALCIFEROL", "drug"]
    df.loc[len(df.index)] = ["omeprazole_penn", "OMEPRAZOLE", "drug"]
    df.loc[len(df.index)] = ["benzonatate_penn", "BENZONATATE", "drug"]
    df.loc[len(df.index)] = ["ventilator_penn", "VENTILATOR", "procedure"]
    df.loc[len(df.index)] = ["tracheostomy_penn", "TRACHEOSTOMY", "procedure"]
    df.loc[len(df.index)] = ["antiinflammatory_penn", "ANTIINFLAM", "procedure"]
    df.loc[len(df.index)] = ["electrocardiogram_penn", "ELECTROCARDIOGRAM", "procedure"]
    df.loc[len(df.index)] = ["respinfection_penn", "RESPINF", "condition"]
    df.loc[len(df.index)] = ["amlodipine_penn", "AMLODIPINE", "drug"]
    df.loc[len(df.index)] = ["metabolism_disorder_penn", "METADISORDER", "condition"]
    df.loc[len(df.index)] = ["bloodpressure_penn", "BLOODPRESSURE", "observation"]
    df.loc[len(df.index)] = ["lungdisorder_penn", "LUNGDISORDER", "condition"]  
    df.loc[len(df.index)] = ["sertraline_penn", "SERTRALINE", "drug"]  
    df.loc[len(df.index)] = ["insulin_penn", "INSULIN", "drug"]  
    
    

    print(df)
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7881151d-1d96-4301-a385-5d663cc22d56"),
    LL_DO_NOT_DELETE_REQUIRED_concept_sets_all=Input(rid="ri.foundry.main.dataset.029aa987-cfef-48fc-bf45-cffd3792cd93"),
    custom_sets=Input(rid="ri.foundry.main.dataset.2a1d8398-c54a-4732-8f23-073ced750426")
)
#The purpose of this node is to optimize the user's experience connecting a customized concept set "fusion sheet" input data frame to replace LL_concept_sets_fusion_everyone.

def customized_concept_set_input( LL_DO_NOT_DELETE_REQUIRED_concept_sets_all, custom_sets):
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
    Output(rid="ri.foundry.main.dataset.407bb4de-2a25-4520-8e03-f1e07031a43f"),
    distinct_vax_person_testing=Input(rid="ri.foundry.main.dataset.783be2cb-5a74-4652-baf6-b0b7b5b6d046")
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
    Output(rid="ri.foundry.main.dataset.ca1772cd-c245-453d-ac74-d0c42e490f2e"),
    device_exposure=Input(rid="ri.foundry.main.dataset.c1fd6d67-fc80-4747-89ca-8eb04efcb874"),
    device_exposure_testing=Input(rid="ri.foundry.main.dataset.7e24a101-2206-45d9-bcaa-b9d84bd2f990")
)
def device_exposure_testing_copy(device_exposure_testing, device_exposure):
    return device_exposure_testing if LOAD_TEST == 1 else device_exposure

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ffc3d120-eaa8-4a04-8bcb-69b6dcb16ad8"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    device_exposure=Input(rid="ri.foundry.main.dataset.c1fd6d67-fc80-4747-89ca-8eb04efcb874")
)
def device_table_analysis_1(device_exposure, Long_COVID_Silver_Standard):
    TABLE = device_exposure
    CONCEPT_NAME_COL = "device_concept_name"
    l, h = 0, 750

    label = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"]))
    TABLE = TABLE.join(label, "person_id").select(F.col("person_id"), F.col(CONCEPT_NAME_COL), F.col("outcome")).filter(F.col(CONCEPT_NAME_COL) != "No matching concept")
    distinct = TABLE.groupBy(CONCEPT_NAME_COL).count().orderBy("count", ascending=False).limit(h).select(F.col(CONCEPT_NAME_COL)).toPandas()[CONCEPT_NAME_COL]
    
    pos, count, ppl, ppl_pos = [], [], [], []
    cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
    print(len(distinct))
    t = time.time()
    for cname in distinct:
        f = TABLE.agg(
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname)),
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1)),
            F.countDistinct(F.when(F.col(CONCEPT_NAME_COL) == cname, F.col("person_id")).otherwise(None)),
            F.countDistinct(F.when(((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1)), F.col("person_id")).otherwise(None)),
        ).collect()
        one_count = f[0][1]
        size = f[0][0]
        people_count =f[0][2]
        people_one = f[0][3]
        pos.append(one_count/size)
        count.append(size)
        ppl.append(people_count)
        ppl_pos.append(people_one/people_count)
    print(time.time() - t)
    r = pd.DataFrame(list(zip(distinct,pos, count, ppl, ppl_pos)), columns=["concept_name","encounter_pos", "encounter_count", "people_count", "people_pos"])
    r['domain'] = CONCEPT_NAME_COL
    r["heuristic_count"] = r.apply(lambda row: row["encounter_pos"] * row["encounter_count"] if row["encounter_pos"] > 0.7 else 0, axis=1)
    r["heuristic_person"] = r.apply(lambda row: row["people_pos"] * row["people_count"] if row["people_pos"] > 0.7 else 0, axis=1)
    r =r[["concept_name", 'domain', 'encounter_pos','people_pos','encounter_count', 'people_count','heuristic_count','heuristic_person']]
    
    return r

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.6223d2b6-e8b8-4d48-8c4c-81dd2959d131"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5"),
    drug_exposure_testing=Input(rid="ri.foundry.main.dataset.26a51cab-0279-45a6-bbc0-f44a12b52f9c")
)
def drug_exposure_testing_copy(drug_exposure_testing, drug_exposure):
    return drug_exposure_testing if LOAD_TEST == 1 else drug_exposure

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2a5480ef-7699-4f0c-bf5c-2a0f8401224d"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5")
)
def drug_table_analysis_1(drug_exposure, Long_COVID_Silver_Standard):
    TABLE = drug_exposure
    CONCEPT_NAME_COL = "drug_concept_name"
    l, h = 0, 1

    label = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"]))
    TABLE = TABLE.join(label, "person_id").select(F.col("person_id"), F.col(CONCEPT_NAME_COL), F.col("outcome")).filter(F.col(CONCEPT_NAME_COL) != "No matching concept")
    distinct = TABLE.groupBy(CONCEPT_NAME_COL).count().orderBy("count", ascending=False).limit(h).select(F.col(CONCEPT_NAME_COL)).toPandas()[CONCEPT_NAME_COL]
    
    pos, count, ppl, ppl_pos = [], [], [], []
    cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
    print(len(distinct))
    t = time.time()
    for cname in distinct:
        f = TABLE.agg(
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname)),
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1)),
            F.countDistinct(F.when(F.col(CONCEPT_NAME_COL) == cname, F.col("person_id")).otherwise(None)),
            F.countDistinct(F.when(((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1)), F.col("person_id")).otherwise(None)),
        ).collect()
        one_count = f[0][1]
        size = f[0][0]
        people_count =f[0][2]
        people_one = f[0][3]
        pos.append(one_count/size)
        count.append(size)
        ppl.append(people_count)
        ppl_pos.append(people_one/people_count)
    print(time.time() - t)
    r = pd.DataFrame(list(zip(distinct,pos, count, ppl, ppl_pos)), columns=["concept_name","encounter_pos", "encounter_count", "people_count", "people_pos"])
    r['domain'] = CONCEPT_NAME_COL
    r["heuristic_count"] = r.apply(lambda row: row["encounter_pos"] * row["encounter_count"] if row["encounter_pos"] > 0.7 else 0, axis=1)
    r["heuristic_person"] = r.apply(lambda row: row["people_pos"] * row["people_count"] if row["people_pos"] > 0.7 else 0, axis=1)
    r =r[["concept_name", 'domain', 'encounter_pos','people_pos','encounter_count', 'people_count','heuristic_count','heuristic_person']]
    
    return r

@transform_pandas(
    Output(rid="ri.vector.main.execute.dd25cecd-c113-4e2c-88c2-bca2053539e6"),
    produce_dataset=Input(rid="ri.foundry.main.dataset.ae1c108c-1813-47ba-831c-e5a37c599c49"),
    train_sequential_model_3=Input(rid="ri.foundry.main.dataset.4fa4a34a-a9e7-489f-a499-023c2d4c44ac")
)
def evaluate_mTans_shap_vals(train_sequential_model_3, produce_dataset):
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max = read_from_pickle(produce_dataset, "train_data.pickle")
    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, _,_ = read_from_pickle(produce_dataset, "subset_valid_data.pickle")

    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    # Construct dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)

    epochs=10
    # print(train_dataset.__getitem__(1)[0])
    dim = train_dataset.__getitem__(1)[0].shape[-1]
    static_input_dim = train_dataset.__getitem__(1)[3].shape[-1]
    print("data shape::", train_dataset.__getitem__(1)[0].shape)
    print("mask shape::", train_dataset.__getitem__(1)[1].shape)
    print("dim::", dim)
    # print(data_min)
    latent_dim=20
    rec_hidden=32
    learn_emb=True
    enc_num_heads=1
    num_ref_points=128
    gen_hidden=30
    dec_num_heads=1
    classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    lr = 0.0005

    

    rec = rec.to(device)
    dec = dec.to(device)
    classifier = classifier.to(device)

    rec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_rec.pickle'))
    dec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_dec.pickle'))
    classifier.load_state_dict(read_model_from_pickle(train_sequential_model_3, "mTans_classifier.pickle"))
    print("start evaluating shapley values!")
    evaluate_shapley_value(valid_loader, rec, dec, classifier, latent_dim, 1, dim, device)

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
    location_testing_copy=Input(rid="ri.foundry.main.dataset.71a84ecb-f5da-4847-937b-42a7fb9e1272"),
    manifest_safe_harbor_testing_copy=Input(rid="ri.foundry.main.dataset.f756c161-a369-4a22-9591-03ace0f5d1a5"),
    microvisits_to_macrovisits_testing_copy=Input(rid="ri.foundry.main.dataset.05de4355-6100-463e-930a-0e9d3c8a8baa"),
    person_testing_copy=Input(rid="ri.foundry.main.dataset.543e1d80-626e-4a3d-a196-d0c7b434fb41")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave. More information can be found in the README linked here (https://unite.nih.gov/workspace/report/ri.report.main.report.855e1f58-bf44-4343-9721-8b4c878154fe).
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node gathers some commonly used facts about these patients from the "person" and "location" tables, as well as some facts about the patient's institution (from the "manifest" table).  Available age, race, and locations data (including SDOH variables for L3 only) is gathered at this node.  The patient’s total number of visits as well as the number of days in their observation period is calculated from the “microvisits_to_macrovisits” table in this node.  These facts will eventually be joined with the final patient-level table in the final node.

def everyone_cohort_de_id_testing(location_testing_copy, manifest_safe_harbor_testing_copy, microvisits_to_macrovisits_testing_copy, custom_concept_set_members, person_testing_copy):
    concept_set_members = custom_concept_set_members
        
    """
    Select proportion of enclave patients to use: A value of 1.0 indicates the pipeline will use all patients in the persons table.  
    A value less than 1.0 takes a random sample of the patients with a value of 0.001 (for example) representing a 0.1% sample of the persons table will be used.
    """
    proportion_of_patients_to_use = 1.0

    concepts_df = concept_set_members
    
    person_sample = person_testing_copy \
        .select('person_id','year_of_birth','month_of_birth','day_of_birth','ethnicity_concept_name','race_concept_name','gender_concept_name','location_id','data_partner_id') \
        .distinct() \
        .sample(False, proportion_of_patients_to_use, 111)

    visits_df = microvisits_to_macrovisits_testing_copy.select("person_id", "macrovisit_start_date", "visit_start_date")

    manifest_df = manifest_safe_harbor_testing_copy \
        .select('data_partner_id','run_date','cdm_name','cdm_version','shift_date_yn','max_num_shift_days') \
        .withColumnRenamed("run_date", "data_extraction_date")

    location_df = location_testing_copy \
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
    condition_occurrence_testing_copy=Input(rid="ri.foundry.main.dataset.a32b3d71-226c-4347-aaed-2c4900e2f4fb"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    customized_concept_set_input_testing=Input(rid="ri.foundry.main.dataset.842d6169-dd15-44de-9955-c978ffb1c801"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node filters the condition_eras table for rows that have a condition_concept_id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these conditions are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_conditions_of_interest_testing(everyone_cohort_de_id_testing, condition_occurrence_testing_copy, customized_concept_set_input_testing, custom_concept_set_members):
    concept_set_members = custom_concept_set_members

    #bring in only cohort patient ids
    persons = everyone_cohort_de_id_testing.select('person_id')
    #filter observations table to only cohort patients    
    conditions_df = condition_occurrence_testing_copy \
        .select('person_id', 'condition_start_date', 'condition_concept_id') \
        .where(F.col('condition_start_date').isNotNull()) \
        .withColumnRenamed('condition_start_date','visit_date') \
        .withColumnRenamed('condition_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the conditions domain
    fusion_df = customized_concept_set_input_testing \
        .filter(customized_concept_set_input_testing.domain.contains('condition')) \
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
    device_exposure_testing_copy=Input(rid="ri.foundry.main.dataset.ca1772cd-c245-453d-ac74-d0c42e490f2e"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_devices_of_interest_testing(everyone_cohort_de_id_testing, customized_concept_set_input_testing, custom_concept_set_members, device_exposure_testing_copy):
    concept_set_members = custom_concept_set_members

    #bring in only cohort patient ids
    persons = everyone_cohort_de_id_testing.select('person_id')
    #filter device exposure table to only cohort patients
    devices_df = device_exposure_testing_copy \
        .select('person_id','device_exposure_start_date','device_concept_id') \
        .where(F.col('device_exposure_start_date').isNotNull()) \
        .withColumnRenamed('device_exposure_start_date','visit_date') \
        .withColumnRenamed('device_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the devices domain
    fusion_df = customized_concept_set_input_testing \
        .filter(customized_concept_set_input_testing.domain.contains('device')) \
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
        .withColumn('DAYS_SINCE_FCP', F.datediff(F.col('visit_date'), F.col('first_covid_positive'))) \
        .drop(F.col('first_covid_positive'))

    return df
    

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c467232f-7ce8-493a-9c58-19438b8bae42"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    customized_concept_set_input_testing=Input(rid="ri.foundry.main.dataset.842d6169-dd15-44de-9955-c978ffb1c801"),
    drug_exposure_testing_copy=Input(rid="ri.foundry.main.dataset.6223d2b6-e8b8-4d48-8c4c-81dd2959d131"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4"),
    first_covid_positive_testing=Input(rid="ri.foundry.main.dataset.5b84887d-8fd0-49bf-969e-6a78dc3060ca")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_drugs_of_interest_testing( drug_exposure_testing_copy, everyone_cohort_de_id_testing, customized_concept_set_input_testing, first_covid_positive_testing, custom_concept_set_members):
    concept_set_members = custom_concept_set_members
  
    #bring in only cohort patient ids
    persons = everyone_cohort_de_id_testing.select('person_id')
    #filter drug exposure table to only cohort patients    
    drug_df = drug_exposure_testing_copy \
        .select('person_id','drug_exposure_start_date','drug_concept_id') \
        .where(F.col('drug_exposure_start_date').isNotNull()) \
        .withColumnRenamed('drug_exposure_start_date','visit_date') \
        .withColumnRenamed('drug_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the drug domain
    fusion_df = customized_concept_set_input_testing \
        .filter(customized_concept_set_input_testing.domain.contains('drug')) \
        .select('concept_set_name','indicator_prefix')
    fusion_df.show(100)
    #filter concept set members table to only concept ids for the drugs of interest
    concepts_df = concept_set_members \
        .select('concept_set_name', 'is_most_recent_version', 'concept_id') \
        .where((F.col('is_most_recent_version')=='true') | (F.col('is_most_recent_version')==True)) \
        .join(fusion_df, 'concept_set_name', 'inner') \
        .select('concept_id','indicator_prefix', 'concept_set_name')
    concepts_df.show(100)
    #find drug exposure information based on matching concept ids for drugs of interest
    df = drug_df.join(concepts_df, 'concept_id', 'inner')
    #collapse to unique person and visit date and pivot on future variable name to create flag for rows associated with the concept sets for drugs of interest
    df = df.groupby('person_id','visit_date').pivot('indicator_prefix').agg(F.lit(1)).na.fill(0) \
        .join(first_covid_positive_testing, 'person_id', 'leftouter') \
        .withColumn('BEFORE_FCP', F.when(F.datediff(F.col('visit_date'), F.col('first_covid_positive')) < 0, 1).otherwise(0)) \
        .withColumn('DAYS_SINCE_FCP', F.datediff(F.col('visit_date'), F.col('first_covid_positive'))) \
        .drop(F.col('first_covid_positive'))

    return df

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
    
    blood_Platelets_codeset_id=[3024929]  # normal people: 130-459
    lowest_blood_Platelets = 50
    highest_blood_Platelets = 1000

    blood_Hematocrit_codeset_id=[3023314]  # normal people: 30-54
    lowest_blood_Hematocrit = 10
    highest_blood_Hematocrit = 150
    
    blood_Leukocytes_codeset_id=[3000905]  # normal people: 4-11
    lowest_blood_Leukocytes = 1
    highest_blood_Leukocytes = 30

    blood_Bilirubin_codeset_id=[3024128]  # normal people: 0.1-1.5
    lowest_blood_Bilirubin = 0.02
    highest_blood_Bilirubin = 5

    blood_Albumin_codeset_id=[3024561]  # normal people: 3.5-5.0
    lowest_blood_Albumin = 1
    highest_blood_Albumin = 30
    
    ####
    blood_Troponin_codeset_id=[3033745]  # normal people: 0-0.01
    lowest_blood_Troponin = 0
    highest_blood_Troponin = 1

    blood_Procalcitonin_codeset_id=[44817130]  # normal people: 0-0.1
    lowest_blood_Procalcitonin = 0
    highest_blood_Procalcitonin = 1

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
    
    blood_Platelets_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Platelets', F.when(df.measurement_concept_id.isin(blood_Platelets_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Platelets, highest_blood_Platelets), df.harmonized_value_as_number).otherwise(0))
    
    blood_Hematocrit_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Hematocrit', F.when(df.measurement_concept_id.isin(blood_Hematocrit_codeset_id) & df.value_as_number.between(lowest_blood_Hematocrit, highest_blood_Hematocrit), df.value_as_number).otherwise(0))
    
    blood_Leukocytes_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Leukocytes', F.when(df.measurement_concept_id.isin(blood_Leukocytes_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Leukocytes, highest_blood_Leukocytes), df.harmonized_value_as_number).otherwise(0))
    
    blood_Bilirubin_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Bilirubin', F.when(df.measurement_concept_id.isin(blood_Bilirubin_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Bilirubin, highest_blood_Bilirubin), df.harmonized_value_as_number).otherwise(0))
    
    blood_Albumin_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Albumin', F.when(df.measurement_concept_id.isin(blood_Albumin_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Albumin, highest_blood_Albumin), df.harmonized_value_as_number).otherwise(0))
    
    ####
    blood_Troponin_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Troponin', F.when(df.measurement_concept_id.isin(blood_Troponin_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Troponin, highest_blood_Troponin), df.harmonized_value_as_number).otherwise(0))
    
    blood_Procalcitonin_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Procalcitonin', F.when(df.measurement_concept_id.isin(blood_Procalcitonin_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Procalcitonin, highest_blood_Procalcitonin), df.harmonized_value_as_number).otherwise(0))

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
    
    blood_Platelets_df = blood_Platelets_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Platelets').alias('blood_Platelets')
    )

    blood_Hematocrit_df = blood_Hematocrit_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Hematocrit').alias('blood_Hematocrit')
    )

    blood_Leukocytes_df = blood_Leukocytes_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Leukocytes').alias('blood_Leukocytes')
    )

    blood_Bilirubin_df = blood_Bilirubin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Bilirubin').alias('blood_Bilirubin')
    )

    blood_Albumin_df = blood_Albumin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Albumin').alias('blood_Albumin')
    )
    ###
    blood_Troponin_df = blood_Troponin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Troponin').alias('blood_Troponin')
    )
    
    blood_Procalcitonin_df = blood_Procalcitonin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Procalcitonin').alias('blood_Procalcitonin')
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
    df = labs_df.join(BMI_df, on=['person_id', 'visit_date'], how='left').join(blood_oxygen_df, on=['person_id', 'visit_date'], how='left').join(blood_sodium_df, on=['person_id', 'visit_date'], how='left').join(blood_hemoglobin_df, on=['person_id', 'visit_date'], how='left').join(respiratory_rate_df, on=['person_id', 'visit_date'], how='left').join(blood_Creatinine_df, on=['person_id', 'visit_date'], how='left').join(blood_UreaNitrogen_df, on=['person_id', 'visit_date'], how='left').join(blood_Potassium_df, on=['person_id', 'visit_date'], how='left').join(blood_Chloride_df, on=['person_id', 'visit_date'], how='left').join(blood_Calcium_df, on=['person_id', 'visit_date'], how='left').join(MCV_df, on=['person_id', 'visit_date'], how='left').join(Erythrocytes_df, on=['person_id', 'visit_date'], how='left').join(MCHC_df, on=['person_id', 'visit_date'], how='left').join(Systolic_blood_pressure_df, on=['person_id', 'visit_date'], how='left').join(Diastolic_blood_pressure_df, on=['person_id', 'visit_date'], how='left').join(heart_rate_df, on=['person_id', 'visit_date'], how='left').join(temperature_df, on=['person_id', 'visit_date'], how='left').join(blood_Glucose_df, on=['person_id', 'visit_date'], how='left').join(blood_Platelets_df, on=['person_id', 'visit_date'], how='left').join(blood_Hematocrit_df, on=['person_id', 'visit_date'], how='left').join(blood_Leukocytes_df, on=['person_id', 'visit_date'], how='left').join(blood_Bilirubin_df, on=['person_id', 'visit_date'], how='left').join(blood_Albumin_df, on=['person_id', 'visit_date'], how='left').join(blood_Troponin_df, on=['person_id', 'visit_date'], how='left').join(blood_Procalcitonin_df, on=['person_id', 'visit_date'], how='left')

    return df

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.947ff73f-4427-404f-b65b-2e709cdcbddd"),
    custom_concept_set_members=Input(rid="ri.foundry.main.dataset.fca16979-a1a8-4e62-9661-7adc1c413729"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4"),
    measurement_testing_copy=Input(rid="ri.foundry.main.dataset.92566631-b0d5-4fab-8a14-5c3d0d6ad560")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This node filters the measurements table for rows that have a measurement_concept_id associated with one of the concept sets described in the data dictionary in the README.  Indicator names for a positive COVID PCR or AG test, negative COVID PCR or AG test, positive COVID antibody test, and negative COVID antibody test are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date. It also finds the harmonized value as a number for BMI measurements and collapses these values to unique instances on the basis of patient and visit date.  Measurement BMI cutoffs included are intended for adults. Analyses focused on pediatric measurements should use different bounds for BMI measurements. 

def everyone_measurements_of_interest_testing(measurement_testing_copy, everyone_cohort_de_id_testing, custom_concept_set_members):
    concept_set_members = custom_concept_set_members
    
    #bring in only cohort patient ids
    persons = everyone_cohort_de_id_testing.select('person_id', 'gender_concept_name')
    #filter procedure occurrence table to only cohort patients    
    df = measurement_testing_copy \
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
    
    blood_Platelets_codeset_id=[3024929]  # normal people: 130-459
    lowest_blood_Platelets = 50
    highest_blood_Platelets = 1000

    blood_Hematocrit_codeset_id=[3023314]  # normal people: 30-54
    lowest_blood_Hematocrit = 10
    highest_blood_Hematocrit = 150
    
    blood_Leukocytes_codeset_id=[3000905]  # normal people: 4-11
    lowest_blood_Leukocytes = 1
    highest_blood_Leukocytes = 30

    blood_Bilirubin_codeset_id=[3024128]  # normal people: 0.1-1.5
    lowest_blood_Bilirubin = 0.02
    highest_blood_Bilirubin = 5

    blood_Albumin_codeset_id=[3024561]  # normal people: 3.5-5.0
    lowest_blood_Albumin = 1
    highest_blood_Albumin = 30
    
    ####
    blood_Troponin_codeset_id=[3033745]  # normal people: 0-0.01
    lowest_blood_Troponin = 0
    highest_blood_Troponin = 1

    blood_Procalcitonin_codeset_id=[44817130]  # normal people: 0-0.1
    lowest_blood_Procalcitonin = 0
    highest_blood_Procalcitonin = 1

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
    
    blood_Platelets_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Platelets', F.when(df.measurement_concept_id.isin(blood_Platelets_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Platelets, highest_blood_Platelets), df.harmonized_value_as_number).otherwise(0))
    
    blood_Hematocrit_df =  df.where(F.col('value_as_number').isNotNull()) \
        .withColumn('blood_Hematocrit', F.when(df.measurement_concept_id.isin(blood_Hematocrit_codeset_id) & df.value_as_number.between(lowest_blood_Hematocrit, highest_blood_Hematocrit), df.value_as_number).otherwise(0))
    
    blood_Leukocytes_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Leukocytes', F.when(df.measurement_concept_id.isin(blood_Leukocytes_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Leukocytes, highest_blood_Leukocytes), df.harmonized_value_as_number).otherwise(0))
    
    blood_Bilirubin_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Bilirubin', F.when(df.measurement_concept_id.isin(blood_Bilirubin_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Bilirubin, highest_blood_Bilirubin), df.harmonized_value_as_number).otherwise(0))
    
    blood_Albumin_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Albumin', F.when(df.measurement_concept_id.isin(blood_Albumin_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Albumin, highest_blood_Albumin), df.harmonized_value_as_number).otherwise(0))
    
    ####
    blood_Troponin_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Troponin', F.when(df.measurement_concept_id.isin(blood_Troponin_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Troponin, highest_blood_Troponin), df.harmonized_value_as_number).otherwise(0))
    
    blood_Procalcitonin_df =  df.where(F.col('harmonized_value_as_number').isNotNull()) \
        .withColumn('blood_Procalcitonin', F.when(df.measurement_concept_id.isin(blood_Procalcitonin_codeset_id) & df.harmonized_value_as_number.between(lowest_blood_Procalcitonin, highest_blood_Procalcitonin), df.harmonized_value_as_number).otherwise(0))

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
    
    blood_Platelets_df = blood_Platelets_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Platelets').alias('blood_Platelets')
    )

    blood_Hematocrit_df = blood_Hematocrit_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Hematocrit').alias('blood_Hematocrit')
    )

    blood_Leukocytes_df = blood_Leukocytes_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Leukocytes').alias('blood_Leukocytes')
    )

    blood_Bilirubin_df = blood_Bilirubin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Bilirubin').alias('blood_Bilirubin')
    )

    blood_Albumin_df = blood_Albumin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Albumin').alias('blood_Albumin')
    )
    ###
    blood_Troponin_df = blood_Troponin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Troponin').alias('blood_Troponin')
    )
    
    blood_Procalcitonin_df = blood_Procalcitonin_df.groupby('person_id', 'visit_date').agg(
    F.max('blood_Procalcitonin').alias('blood_Procalcitonin')
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
    df = labs_df.join(BMI_df, on=['person_id', 'visit_date'], how='left').join(blood_oxygen_df, on=['person_id', 'visit_date'], how='left').join(blood_sodium_df, on=['person_id', 'visit_date'], how='left').join(blood_hemoglobin_df, on=['person_id', 'visit_date'], how='left').join(respiratory_rate_df, on=['person_id', 'visit_date'], how='left').join(blood_Creatinine_df, on=['person_id', 'visit_date'], how='left').join(blood_UreaNitrogen_df, on=['person_id', 'visit_date'], how='left').join(blood_Potassium_df, on=['person_id', 'visit_date'], how='left').join(blood_Chloride_df, on=['person_id', 'visit_date'], how='left').join(blood_Calcium_df, on=['person_id', 'visit_date'], how='left').join(MCV_df, on=['person_id', 'visit_date'], how='left').join(Erythrocytes_df, on=['person_id', 'visit_date'], how='left').join(MCHC_df, on=['person_id', 'visit_date'], how='left').join(Systolic_blood_pressure_df, on=['person_id', 'visit_date'], how='left').join(Diastolic_blood_pressure_df, on=['person_id', 'visit_date'], how='left').join(heart_rate_df, on=['person_id', 'visit_date'], how='left').join(temperature_df, on=['person_id', 'visit_date'], how='left').join(blood_Glucose_df, on=['person_id', 'visit_date'], how='left').join(blood_Platelets_df, on=['person_id', 'visit_date'], how='left').join(blood_Hematocrit_df, on=['person_id', 'visit_date'], how='left').join(blood_Leukocytes_df, on=['person_id', 'visit_date'], how='left').join(blood_Bilirubin_df, on=['person_id', 'visit_date'], how='left').join(blood_Albumin_df, on=['person_id', 'visit_date'], how='left').join(blood_Troponin_df, on=['person_id', 'visit_date'], how='left').join(blood_Procalcitonin_df, on=['person_id', 'visit_date'], how='left')

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
    observation_testing_copy=Input(rid="ri.foundry.main.dataset.9834f887-c80e-478c-b538-5bb39b9db6a7")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_observations_of_interest_testing(observation_testing_copy, everyone_cohort_de_id_testing, customized_concept_set_input_testing, custom_concept_set_members):
    concept_set_members = custom_concept_set_members
   
    #bring in only cohort patient ids
    persons = everyone_cohort_de_id_testing.select('person_id')
    #filter observations table to only cohort patients    
    observations_df = observation_testing_copy \
        .select('person_id','observation_date','observation_concept_id') \
        .where(F.col('observation_date').isNotNull()) \
        .withColumnRenamed('observation_date','visit_date') \
        .withColumnRenamed('observation_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the observations domain
    fusion_df = customized_concept_set_input_testing \
        .filter(customized_concept_set_input_testing.domain.contains('observation')) \
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
    procedure_occurrence_testing_copy=Input(rid="ri.foundry.main.dataset.2d76588c-fe75-4d07-8044-f054444ec728")
)
#Purpose - The purpose of this pipeline is to produce a visit day level and a persons level fact table for all patients in the N3C enclave.
#Creator/Owner/contact - Andrea Zhou
#Last Update - 5/6/22
#Description - This nodes filter the source OMOP tables for rows that have a standard concept id associated with one of the concept sets described in the data dictionary in the README through the use of a fusion sheet.  Indicator names for these variables are assigned, and the indicators are collapsed to unique instances on the basis of patient and visit date.

def everyone_procedures_of_interest_testing(everyone_cohort_de_id_testing, procedure_occurrence_testing_copy, customized_concept_set_input_testing, custom_concept_set_members):
    concept_set_members = custom_concept_set_members
  
    #bring in only cohort patient ids
    persons = everyone_cohort_de_id_testing.select('person_id')
    #filter procedure occurrence table to only cohort patients    
    procedures_df = procedure_occurrence_testing_copy \
        .select('person_id','procedure_date','procedure_concept_id') \
        .where(F.col('procedure_date').isNotNull()) \
        .withColumnRenamed('procedure_date','visit_date') \
        .withColumnRenamed('procedure_concept_id','concept_id') \
        .join(persons,'person_id','inner')

    #filter fusion sheet for concept sets and their future variable names that have concepts in the procedure domain
    fusion_df = customized_concept_set_input_testing \
        .filter(customized_concept_set_input_testing.domain.contains('procedure')) \
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
        .withColumn('VAX_DAYS_SINCE_FCP', F.datediff(F.col('visit_date'), F.col('first_covid_positive'))) \
        .drop(F.col('first_covid_positive'))

    return df

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.97cdf176-e012-49e9-8eff-6667e5f67e1a"),
    Vaccine_fact_de_identified_testing=Input(rid="ri.foundry.main.dataset.9392c81b-bbbf-4e66-a366-a2e7e4f9db7b"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4"),
    first_covid_positive_testing=Input(rid="ri.foundry.main.dataset.5b84887d-8fd0-49bf-969e-6a78dc3060ca")
)
def everyone_vaccines_of_interest_testing(everyone_cohort_de_id_testing, Vaccine_fact_de_identified_testing, first_covid_positive_testing):
    vaccine_fact_de_identified = Vaccine_fact_de_identified_testing
    
    persons = everyone_cohort_de_id_testing.select('person_id')
    vax_df = Vaccine_fact_de_identified_testing.select('person_id', 'vaccine_txn', '1_vax_date', '2_vax_date', '3_vax_date', '4_vax_date') \
        .join(persons, 'person_id', 'inner')
        
    vax_switch = Vaccine_fact_de_identified_testing.select('person_id', '1_vax_type', 'date_diff_1_2') \
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
        .join(first_covid_positive_testing, 'person_id', 'leftouter') \
        .withColumn('vax_before_FCP', F.when(F.datediff(F.col('visit_date'), F.col('first_covid_positive')) < 0, 1).otherwise(0)) \
        .withColumn('VAX_DAYS_SINCE_FCP', F.datediff(F.col('visit_date'), F.col('first_covid_positive'))) \
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
    Output(rid="ri.foundry.main.dataset.5b84887d-8fd0-49bf-969e-6a78dc3060ca"),
    everyone_conditions_of_interest_testing=Input(rid="ri.foundry.main.dataset.ae4f0220-6939-4f61-a97a-ff78d29df156")
)
def first_covid_positive_testing(everyone_conditions_of_interest_testing):
    w = Window.partitionBy('person_id').orderBy(F.asc('visit_date'))
    df = everyone_conditions_of_interest_testing \
        .filter(F.col('LL_COVID_diagnosis') == 1) \
        .select('person_id', F.first('visit_date').over(w).alias('first_covid_positive')) \
        .distinct()

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.fe3d99a6-3290-48a7-b171-579546a399a4"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    all_patients_summary_fact_table_de_id=Input(rid="ri.foundry.main.dataset.324a6115-7c17-4d4d-94da-a2df11a87fa6")
)
def gb_hp_tuning(Long_COVID_Silver_Standard, all_patients_summary_fact_table_de_id):
    static_cols = ['person_id','total_visits', 'age']

    cols = static_cols + [col for col in all_patients_summary_fact_table_de_id.columns if 'indicator' in col]
    
    ## get outcome column
    Long_COVID_Silver_Standard["outcome"] = Long_COVID_Silver_Standard.apply(lambda x: max([x["pasc_code_after_four_weeks"], x["pasc_code_prior_four_weeks"]]), axis=1)
    Outcome_df = all_patients_summary_fact_table_de_id[["person_id"]].merge(Long_COVID_Silver_Standard, on="person_id", how="left")
    Outcome_df = Outcome_df[["person_id", "outcome"]].sort_values('person_id')
    Outcome = list(Outcome_df["outcome"])

            
    Training_and_Holdout = all_patients_summary_fact_table_de_id[cols].fillna(0.0).sort_values('person_id')
    X_train_no_ind, X_test_no_ind, y_train, y_test = train_test_split(Training_and_Holdout, Outcome, train_size=0.9, random_state=1)
    X_train, X_test = X_train_no_ind.set_index("person_id"), X_test_no_ind.set_index("person_id")
    X, y = Training_and_Holdout.set_index("person_id"), Outcome

    model = GradientBoostingClassifier()
    # Create the random grid
    grid = {
    "loss":["log_loss", "exponential"],
    "learning_rate": [0.05, 0.075, 0.1, 0.2,.3,.4,.5],
    "min_samples_split": [0.001,0.01,0.05, 0.5,1],
    "min_samples_leaf": [0.001,0.005,0.05,0.5,1],
    "max_depth":[3,5,8,12],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "squared_error"],
    "subsample":[0.5, 0.618, 0.8, 0.9, 0.95, 1.0],
    "n_estimators":[200],
    }

    cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    randomSearch = RandomizedSearchCV(estimator=model, n_iter=60, n_jobs=-1, cv=cvFold, param_distributions=grid,verbose=2,refit=True, scoring='f1').fit(X,y)
    params=randomSearch.best_estimator_.get_params()
    print("All params: \n", params)

    new_model = GradientBoostingClassifier(**params).fit(X_train,y_train)

    test_preds = new_model.predict_proba(X_test)[:, 1]
    
    predictions = pd.DataFrame.from_dict({
        'person_id': X_test_no_ind["person_id"],
        'pred_outcome': test_preds.tolist(),
        'outcome': y_test
    }, orient='columns')

    print("Classification Report:\n{}".format(classification_report(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0))))

    print("MAE:", mean_absolute_error(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("Brier score:", brier_score_loss(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("AP:", average_precision_score(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("ROC AUC:", roc_auc_score(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))

    return predictions

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.1c438e0a-6066-41ff-b7a7-34352cf60ec5"),
    everyone_cohort_de_id=Input(rid="ri.foundry.main.dataset.120adc97-2986-4b7d-9f96-42d8b5d5bedf")
)
def get_train_valid_partition(everyone_cohort_de_id):
    Training_and_Holdout = everyone_cohort_de_id.sort_values('person_id')
    Training_and_Holdout = Training_and_Holdout.sort_index(axis=1)
    print(Training_and_Holdout)    
    # if LOAD_TEST == 0:
    X_train_no_ind, X_test_no_ind = train_test_split(Training_and_Holdout, train_size=0.9, random_state=1)
    
    # train_ids = list(X_train_no_ind.select(F.col('person_id')).distinct().toPandas()['person_id'])
    # test_ids = list(X_test_no_ind.select(F.col('person_id')).distinct().toPandas()['person_id'])
    train_ids = list(X_train_no_ind['person_id'].values.tolist())
    test_ids = list(X_test_no_ind['person_id'].values.tolist())
    write_to_pickle(train_ids, "train_person_ids")
    write_to_pickle(test_ids, "test_person_ids")

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.31ae971a-0175-49bf-b7ec-31fd900e58f5"),
    positive_symptoms=Input(rid="ri.foundry.main.dataset.7cd7ede8-524b-4fe3-8ce1-10f268fcd51b"),
    top_concept_names=Input(rid="ri.foundry.main.dataset.e5cacf25-d71d-44ee-a3da-6bad8eaf53e8")
)
def important_concepts(positive_symptoms, top_concept_names):

    names = top_concept_names[(top_concept_names["scale_above_threshold"]) > 0]
    names = names[(names["count"] > 200)][["concept_name", "count", "scale_above_threshold"]]
    concept_ids = positive_symptoms[["concept_name", "concept_id"]]
    concept_ids = concept_ids.drop_duplicates()
    df = pd.merge(names, concept_ids, "inner", "concept_name")

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.71a84ecb-f5da-4847-937b-42a7fb9e1272"),
    location=Input(rid="ri.foundry.main.dataset.4805affe-3a77-4260-8da5-4f9ff77f51ab"),
    location_testing=Input(rid="ri.foundry.main.dataset.06b728e0-0262-4a7a-b9b7-fe91c3f7da34")
)
def location_testing_copy(location_testing, location):
    return location_testing if LOAD_TEST == 1 else location

@transform_pandas(
    Output(rid="ri.vector.main.execute.dc454421-0a0a-4eb5-b6cc-b452f682a565"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    all_patients_summary_fact_table_de_id=Input(rid="ri.foundry.main.dataset.324a6115-7c17-4d4d-94da-a2df11a87fa6")
)
def lr2_hp_tuning(all_patients_summary_fact_table_de_id, Long_COVID_Silver_Standard):

    static_cols = ['person_id','total_visits', 'age']

    cols = static_cols + [col for col in all_patients_summary_fact_table_de_id.columns if 'indicator' in col]
    
    ## get outcome column
    Long_COVID_Silver_Standard["outcome"] = Long_COVID_Silver_Standard.apply(lambda x: max([x["pasc_code_after_four_weeks"], x["pasc_code_prior_four_weeks"]]), axis=1)
    Outcome_df = all_patients_summary_fact_table_de_id[["person_id"]].merge(Long_COVID_Silver_Standard, on="person_id", how="left")
    Outcome_df = Outcome_df[["person_id", "outcome"]].sort_values('person_id')
    Outcome = list(Outcome_df["outcome"])

            
    Training_and_Holdout = all_patients_summary_fact_table_de_id[cols].fillna(0.0).sort_values('person_id')
    X_train_no_ind, X_test_no_ind, y_train, y_test = train_test_split(Training_and_Holdout, Outcome, train_size=0.9, random_state=1)
    X_train, X_test = X_train_no_ind.set_index("person_id"), X_test_no_ind.set_index("person_id")
    X, y = Training_and_Holdout.set_index("person_id"), Outcome

    model = LogisticRegression()
    C = [0.001, 0.01, 0.1, 1,10,100, 1000]
    fit_intercept = [False,True]
    solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    class_weight=['balanced']
    grid = dict(C=C,fit_intercept=fit_intercept,solver=solver,class_weight=class_weight)

    cvFold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    randomSearch = RandomizedSearchCV(estimator=model, n_iter=60, n_jobs=-1, cv=cvFold, param_distributions=grid,verbose=2,refit=True, scoring='f1').fit(X,y)
    params=randomSearch.best_estimator_.get_params()
    print("All params: \n", params)

    new_model = LogisticRegression(**params).fit(X_train,y_train)

    test_preds = new_model.predict_proba(X_test)[:, 1]
    
    predictions = pd.DataFrame.from_dict({
        'person_id': X_test_no_ind["person_id"],
        'pred_outcome': test_preds.tolist(),
        'outcome': y_test
    }, orient='columns')

    print("Classification Report:\n{}".format(classification_report(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0))))

    print("MAE:", mean_absolute_error(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("Brier score:", brier_score_loss(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("AP:", average_precision_score(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("ROC AUC:", roc_auc_score(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))

    return predictions

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9832b19b-275e-4ac2-bef3-bb57c4287692"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    all_patients_summary_fact_table_de_id=Input(rid="ri.foundry.main.dataset.324a6115-7c17-4d4d-94da-a2df11a87fa6")
)
def lr_hp_tuning(Long_COVID_Silver_Standard, all_patients_summary_fact_table_de_id):
        
    static_cols = ['person_id','total_visits', 'age']

    cols = static_cols + [col for col in all_patients_summary_fact_table_de_id.columns if 'indicator' in col]
    
    ## get outcome column
    Long_COVID_Silver_Standard["outcome"] = Long_COVID_Silver_Standard.apply(lambda x: max([x["pasc_code_after_four_weeks"], x["pasc_code_prior_four_weeks"]]), axis=1)
    Outcome_df = all_patients_summary_fact_table_de_id[["person_id"]].merge(Long_COVID_Silver_Standard, on="person_id", how="left")
    Outcome_df = Outcome_df[["person_id", "outcome"]].sort_values('person_id')
    Outcome = list(Outcome_df["outcome"])

            
    Training_and_Holdout = all_patients_summary_fact_table_de_id[cols].fillna(0.0).sort_values('person_id')
    X_train_no_ind, X_test_no_ind, y_train, y_test = train_test_split(Training_and_Holdout, Outcome, train_size=0.9, random_state=1)
    X_train, X_test = X_train_no_ind.set_index("person_id"), X_test_no_ind.set_index("person_id")
    X, y = Training_and_Holdout.set_index("person_id"), Outcome

    model = LogisticRegression()
    C = [0.001, 0.01, 0.1, 1,10,100, 1000]
    fit_intercept = [False,True]
    solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    grid = dict(C=C,fit_intercept=fit_intercept,solver=solver)

    cvFold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    randomSearch = RandomizedSearchCV(estimator=model, n_iter=60, n_jobs=-1, cv=cvFold, param_distributions=grid,verbose=2,refit=True, scoring='f1').fit(X,y)
    params=randomSearch.best_estimator_.get_params()
    print("All params: \n", params)

    new_model = LogisticRegression(**params).fit(X_train,y_train)

    test_preds = new_model.predict_proba(X_test)[:, 1]
    
    predictions = pd.DataFrame.from_dict({
        'person_id': X_test_no_ind["person_id"],
        'pred_outcome': test_preds.tolist(),
        'outcome': y_test
    }, orient='columns')

    print("Classification Report:\n{}".format(classification_report(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0))))

    print("MAE:", mean_absolute_error(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("Brier score:", brier_score_loss(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("AP:", average_precision_score(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("ROC AUC:", roc_auc_score(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))

    return predictions

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f756c161-a369-4a22-9591-03ace0f5d1a5"),
    manifest_safe_harbor=Input(rid="ri.foundry.main.dataset.b4407989-1851-4e07-a13f-0539fae10f26"),
    manifest_safe_harbor_testing=Input(rid="ri.foundry.main.dataset.7a5c5585-1c69-4bf5-9757-3fd0d0a209a2")
)
def manifest_safe_harbor_testing_copy(manifest_safe_harbor_testing, manifest_safe_harbor):
    return manifest_safe_harbor_testing if LOAD_TEST == 1 else manifest_safe_harbor

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d5cfe420-8f15-45eb-af08-2b12fe71798f"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    measurement=Input(rid="ri.foundry.main.dataset.5c8b84fb-814b-4ee5-a89a-9525f4a617c7")
)
def measurement_analysis( Long_COVID_Silver_Standard, measurement):
    TABLE = measurement
    CONCEPT_NAME_COL = "measurement_concept_name"
    l, h = 0, 750

    label = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"]))
    TABLE = TABLE.join(label, "person_id").select(F.col("person_id"), F.col(CONCEPT_NAME_COL), F.col("outcome")).filter(F.col(CONCEPT_NAME_COL) != "No matching concept")
    distinct = TABLE.groupBy(CONCEPT_NAME_COL).count().orderBy("count", ascending=False).limit(h).select(F.col(CONCEPT_NAME_COL)).toPandas()[CONCEPT_NAME_COL]
    
    pos, count, ppl, ppl_pos = [], [], [], []
    cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
    print(len(distinct))
    t = time.time()
    for cname in distinct:
        f = TABLE.agg(
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname)),
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1)),
            F.countDistinct(F.when(F.col(CONCEPT_NAME_COL) == cname, F.col("person_id")).otherwise(None)),
            F.countDistinct(F.when(((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1)), F.col("person_id")).otherwise(None)),
        ).collect()
        one_count = f[0][1]
        size = f[0][0]
        people_count =f[0][2]
        people_one = f[0][3]
        pos.append(one_count/size)
        count.append(size)
        ppl.append(people_count)
        ppl_pos.append(people_one/people_count)
    print(time.time() - t)
    r = pd.DataFrame(list(zip(distinct,pos, count, ppl, ppl_pos)), columns=["concept_name","encounter_pos", "encounter_count", "people_count", "people_pos"])
    r['domain'] = CONCEPT_NAME_COL
    r["heuristic_count"] = r.apply(lambda row: row["encounter_pos"] * row["encounter_count"] if row["encounter_pos"] > 0.7 else 0, axis=1)
    r["heuristic_person"] = r.apply(lambda row: row["people_pos"] * row["people_count"] if row["people_pos"] > 0.7 else 0, axis=1)
    r =r[["concept_name", 'domain', 'encounter_pos','people_pos','encounter_count', 'people_count','heuristic_count','heuristic_person']]
    
    return r

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
    Output(rid="ri.foundry.main.dataset.92566631-b0d5-4fab-8a14-5c3d0d6ad560"),
    measurement=Input(rid="ri.foundry.main.dataset.5c8b84fb-814b-4ee5-a89a-9525f4a617c7"),
    measurement_testing=Input(rid="ri.foundry.main.dataset.b7749e49-cf01-4d0a-a154-2f00eecab21e")
)
def measurement_testing_copy(measurement_testing, measurement):
    return measurement_testing if LOAD_TEST == 1 else measurement

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.05de4355-6100-463e-930a-0e9d3c8a8baa"),
    microvisits_to_macrovisits=Input(rid="ri.foundry.main.dataset.d77a701f-34df-48a1-a71c-b28112a07ffa"),
    microvisits_to_macrovisits_testing=Input(rid="ri.foundry.main.dataset.f5008fa4-e736-4244-88e1-1da7a68efcdb")
)
def microvisits_to_macrovisits_testing_copy(microvisits_to_macrovisits_testing, microvisits_to_macrovisits):
    return microvisits_to_macrovisits_testing if LOAD_TEST == 1 else microvisits_to_macrovisits

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.02f756f5-d406-4b23-9438-a2d5a5c17cd9"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    positive_symptoms=Input(rid="ri.foundry.main.dataset.7cd7ede8-524b-4fe3-8ce1-10f268fcd51b")
)
def nlp_sym_analysis(positive_symptoms, Long_COVID_Silver_Standard):
    TABLE = positive_symptoms
    CONCEPT_NAME_COL = "concept_name"
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
    Output(rid="ri.foundry.main.dataset.d0963ac3-a28e-4423-bb15-758b9607be79"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    all_patients_summary_fact_table_de_id=Input(rid="ri.foundry.main.dataset.324a6115-7c17-4d4d-94da-a2df11a87fa6")
)
def nn_hp_tuning(Long_COVID_Silver_Standard, all_patients_summary_fact_table_de_id):
    static_cols = ['person_id','total_visits', 'age']

    cols = static_cols + [col for col in all_patients_summary_fact_table_de_id.columns if 'indicator' in col]
    
    ## get outcome column
    Long_COVID_Silver_Standard["outcome"] = Long_COVID_Silver_Standard.apply(lambda x: max([x["pasc_code_after_four_weeks"], x["pasc_code_prior_four_weeks"]]), axis=1)
    Outcome_df = all_patients_summary_fact_table_de_id[["person_id"]].merge(Long_COVID_Silver_Standard, on="person_id", how="left")
    Outcome_df = Outcome_df[["person_id", "outcome"]].sort_values('person_id')
    Outcome = list(Outcome_df["outcome"])

            
    Training_and_Holdout = all_patients_summary_fact_table_de_id[cols].fillna(0.0).sort_values('person_id')
    X_train_no_ind, X_test_no_ind, y_train, y_test = train_test_split(Training_and_Holdout, Outcome, train_size=0.9, random_state=1)
    X_train, X_test = X_train_no_ind.set_index("person_id"), X_test_no_ind.set_index("person_id")
    X, y = Training_and_Holdout.set_index("person_id"), Outcome

    model = MLPClassifier()

    grid = {
        'hidden_layer_sizes': [(15,10),(20,5),(15,5), (10,5), (16,3)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.0005, 0.001, 0.05],
        'learning_rate': ['adaptive'],
    }

    cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    randomSearch = RandomizedSearchCV(estimator=model, n_iter=16, n_jobs=-1, cv=cvFold, param_distributions=grid,verbose=2,refit=True, scoring='f1').fit(X,y)
    params=randomSearch.best_estimator_.get_params()
    print("All params: \n", params)

    new_model = MLPClassifier(**params).fit(X_train,y_train)

    test_preds = new_model.predict_proba(X_test)[:, 1]
    
    predictions = pd.DataFrame.from_dict({
        'person_id': X_test_no_ind["person_id"],
        'pred_outcome': test_preds.tolist(),
        'outcome': y_test
    }, orient='columns')

    print("Classification Report:\n{}".format(classification_report(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0))))

    print("MAE:", mean_absolute_error(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("Brier score:", brier_score_loss(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("AP:", average_precision_score(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("ROC AUC:", roc_auc_score(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))

    return predictions

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d39564f3-817f-4b8a-a8b6-81d4f8fd6bf1"),
    recent_visits_2=Input(rid="ri.foundry.main.dataset.bf18056e-2e27-4f2a-af1a-7b6cabc2a9cf")
)
def num_recent_visits(recent_visits_2):

    # Find how many recent visits are there
    df = recent_visits_2 \
        .groupby('person_id')['visit_date'] \
        .nunique() \
        .reset_index() \
        .rename(columns={"visit_date": "num_recent_visits"})

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.99cb0e77-4710-4cc5-b904-57ebc3339cf2"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2"),
    device_exposure=Input(rid="ri.foundry.main.dataset.c1fd6d67-fc80-4747-89ca-8eb04efcb874"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5"),
    measurement=Input(rid="ri.foundry.main.dataset.5c8b84fb-814b-4ee5-a89a-9525f4a617c7"),
    observation=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3")
)
def obs_latent_sequence(observation, condition_occurrence, drug_exposure, procedure_occurrence, Long_COVID_Silver_Standard, measurement, device_exposure):
    
    k = 200

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
    for TABLE, CONCEPT_ID_COL in tables.items():
        # print(TABLE.show())
        TABLE = TABLE.select(F.col("person_id"), F.col("visit_date"), F.col(CONCEPT_ID_COL))             
        distinct = TABLE.groupBy(CONCEPT_ID_COL).count().orderBy("count", ascending=False).limit(k).select(F.col(CONCEPT_ID_COL)).toPandas()[CONCEPT_ID_COL].tolist()
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
    # data = feats.na.fill(0).join(labels_df, "person_id")
    data = feats.join(labels_df, "person_id")
    print("finish!!")
    return data
    
        
    

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.171b1464-ba7a-41eb-a191-c026ceaa1ed1"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2"),
    device_exposure=Input(rid="ri.foundry.main.dataset.c1fd6d67-fc80-4747-89ca-8eb04efcb874"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5"),
    measurement=Input(rid="ri.foundry.main.dataset.5c8b84fb-814b-4ee5-a89a-9525f4a617c7"),
    observation=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3")
)
from pyspark.sql.functions import datediff
from pyspark.sql.functions import col, max as max_, min as min_

def obs_latent_sequence_0(observation, condition_occurrence, drug_exposure, procedure_occurrence, Long_COVID_Silver_Standard, measurement, device_exposure):
    
    k = 200

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
        distinct = TABLE.groupBy(CONCEPT_ID_COL).count().orderBy("count", ascending=False).limit(k).select(F.col(CONCEPT_ID_COL)).toPandas()[CONCEPT_ID_COL].tolist()
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bebb873c-0676-432e-8196-09ec28e09053"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2"),
    device_exposure=Input(rid="ri.foundry.main.dataset.c1fd6d67-fc80-4747-89ca-8eb04efcb874"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5"),
    measurement=Input(rid="ri.foundry.main.dataset.5c8b84fb-814b-4ee5-a89a-9525f4a617c7"),
    observation=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3")
)
def obs_latent_sequence_0_200_to_400(observation, condition_occurrence, drug_exposure, procedure_occurrence, Long_COVID_Silver_Standard, measurement, device_exposure):
    return obtain_latent_sequence(observation, condition_occurrence, drug_exposure, procedure_occurrence, Long_COVID_Silver_Standard, measurement, device_exposure, k = 200, start_id=1)
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.87b291fb-c532-44ec-9427-89eb450d4493"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2"),
    device_exposure=Input(rid="ri.foundry.main.dataset.c1fd6d67-fc80-4747-89ca-8eb04efcb874"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5"),
    measurement=Input(rid="ri.foundry.main.dataset.5c8b84fb-814b-4ee5-a89a-9525f4a617c7"),
    observation=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3")
)
def obs_latent_sequence_0_400_to_600(observation, condition_occurrence, drug_exposure, procedure_occurrence, Long_COVID_Silver_Standard, measurement, device_exposure):
    return obtain_latent_sequence(observation, condition_occurrence, drug_exposure, procedure_occurrence, Long_COVID_Silver_Standard, measurement, device_exposure, k = 200, start_id=2)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c06cda84-379a-4c67-8111-8135014d0380"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2"),
    device_exposure=Input(rid="ri.foundry.main.dataset.c1fd6d67-fc80-4747-89ca-8eb04efcb874"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5"),
    measurement=Input(rid="ri.foundry.main.dataset.5c8b84fb-814b-4ee5-a89a-9525f4a617c7"),
    observation=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3")
)
def obs_latent_sequence_0_600_to_800(observation, condition_occurrence, drug_exposure, procedure_occurrence, Long_COVID_Silver_Standard, measurement, device_exposure):
    return obtain_latent_sequence(observation, condition_occurrence, drug_exposure, procedure_occurrence, Long_COVID_Silver_Standard, measurement, device_exposure, k = 200, start_id=3)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9738c306-6d58-457d-8fbe-7d1d5a09ef01"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2"),
    device_exposure=Input(rid="ri.foundry.main.dataset.c1fd6d67-fc80-4747-89ca-8eb04efcb874"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5"),
    measurement=Input(rid="ri.foundry.main.dataset.5c8b84fb-814b-4ee5-a89a-9525f4a617c7"),
    observation=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3")
)
def obs_latent_sequence_2(observation, condition_occurrence, drug_exposure, procedure_occurrence, Long_COVID_Silver_Standard, measurement, device_exposure):
    
    k = 50

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
    for TABLE, CONCEPT_ID_COL in tables.items():
        # print(TABLE.show())
        TABLE = TABLE.select(F.col("person_id"), F.col("visit_date"), F.col(CONCEPT_ID_COL))             
        distinct = TABLE.groupBy(CONCEPT_ID_COL).count().orderBy("count", ascending=False).limit(k).select(F.col(CONCEPT_ID_COL)).toPandas()[CONCEPT_ID_COL].tolist()
        df = TABLE.filter(F.col(CONCEPT_ID_COL).isin(distinct))
        df= df.groupBy('person_id', 'visit_date').pivot(CONCEPT_ID_COL).agg(F.lit(1)).na.fill(0)
        df = df.select([F.col(c).alias(CONCEPT_ID_COL[:3]+c) if c != "person_id" and c != "visit_date" else c for c in df.columns ])
        print("df columns::", df.columns)
        print("feats columns::", feats.columns)
        print("common columns::", list(set(df.columns)&set(feats.columns)))
        if table_id == 0:
            feats = feats.join(df, on=list(set(df.columns)&set(feats.columns)), how = "left")
        else:
            feats = feats.join(df, on=list(set(df.columns)&set(feats.columns)), how = "outer")
        table_id += 1
    # data = feats.na.fill(0).join(labels_df, "person_id")
    data = feats.join(labels_df, "person_id")
    
    return data
    
        
    

    
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d049152c-00c4-4584-aa28-c0d4a4177b22"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    observation=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516")
)
def observation_table_analysis_1(observation, Long_COVID_Silver_Standard):
    TABLE = observation
    CONCEPT_NAME_COL = "observation_concept_name"
    l, h = 0, 750

    label = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"]))
    TABLE = TABLE.join(label, "person_id").select(F.col("person_id"), F.col(CONCEPT_NAME_COL), F.col("outcome")).filter(F.col(CONCEPT_NAME_COL) != "No matching concept")
    distinct = TABLE.groupBy(CONCEPT_NAME_COL).count().orderBy("count", ascending=False).limit(h).select(F.col(CONCEPT_NAME_COL)).toPandas()[CONCEPT_NAME_COL]
    
    pos, count, ppl, ppl_pos = [], [], [], []
    cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
    print(len(distinct))
    t = time.time()
    for cname in distinct:
        f = TABLE.agg(
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname)),
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1)),
            F.countDistinct(F.when(F.col(CONCEPT_NAME_COL) == cname, F.col("person_id")).otherwise(None)),
            F.countDistinct(F.when(((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1)), F.col("person_id")).otherwise(None)),
        ).collect()
        one_count = f[0][1]
        size = f[0][0]
        people_count =f[0][2]
        people_one = f[0][3]
        pos.append(one_count/size)
        count.append(size)
        ppl.append(people_count)
        ppl_pos.append(people_one/people_count)
    print(time.time() - t)
    r = pd.DataFrame(list(zip(distinct,pos, count, ppl, ppl_pos)), columns=["concept_name","encounter_pos", "encounter_count", "people_count", "people_pos"])
    r['domain'] = CONCEPT_NAME_COL
    r["heuristic_count"] = r.apply(lambda row: row["encounter_pos"] * row["encounter_count"] if row["encounter_pos"] > 0.7 else 0, axis=1)
    r["heuristic_person"] = r.apply(lambda row: row["people_pos"] * row["people_count"] if row["people_pos"] > 0.7 else 0, axis=1)
    r =r[["concept_name", 'domain', 'encounter_pos','people_pos','encounter_count', 'people_count','heuristic_count','heuristic_person']]
    return r

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9834f887-c80e-478c-b538-5bb39b9db6a7"),
    observation=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516"),
    observation_testing=Input(rid="ri.foundry.main.dataset.fc1ce22e-9cf6-4335-8ca7-aa8c733d506d")
)
def observation_testing_copy(observation_testing, observation):
    return observation_testing if LOAD_TEST == 1 else observation

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a0c6118a-62cf-4a55-a8f7-b71205db55a2"),
    validation_metrics=Input(rid="ri.foundry.main.dataset.def6f994-533b-46b8-95ab-3708d867119c")
)
def penn_predictions(validation_metrics):
     return validation_metrics.select(F.col("person_id"), F.col("all_ens_outcome").alias("penn_prediction"))

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
    Output(rid="ri.foundry.main.dataset.f3f8c10f-3926-480e-be48-d2bc585f2c04"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4")
)
def person_information_testing(everyone_cohort_de_id_testing):
    df = everyone_cohort_de_id_testing

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
    Output(rid="ri.foundry.main.dataset.d3124557-f100-44a9-9c43-51db33c87bd8"),
    broad_related_concepts=Input(rid="ri.foundry.main.dataset.ce8c17fb-63cd-41bd-b9c8-9bf54e5091da"),
    personal_symptom=Input(rid="ri.foundry.main.dataset.4ba26dbd-1e38-48bd-8981-823339eb97f8")
)
def person_nlp_symptom(personal_symptom, broad_related_concepts):
    
    personal_symptom_df = personal_symptom[['note_id', 'person_id', 'note_date', 'visit_occurrence_id']]
    personal_symptom_df = personal_symptom_df.drop_duplicates()
    personal_symptom_df = personal_symptom_df.set_index('note_id')
    all_symptoms_df = broad_related_concepts
    all_symptoms = list(set(all_symptoms_df["concept_set_name"]))

    print(all_symptoms)

    for symptom in all_symptoms:
        symptom_column_name = "sympt_" + symptom.replace("-", "_")
        new_symptom_df = personal_symptom[['note_id', 'term_modifier_certainty', 'concept_set_name']]
        new_symptom_df = new_symptom_df.loc[new_symptom_df.concept_set_name == symptom]
        new_symptom_df = new_symptom_df.rename(columns={'term_modifier_certainty': symptom_column_name})
        new_symptom_df = new_symptom_df[["note_id", symptom_column_name]]
        new_symptom_df = new_symptom_df.drop_duplicates()
        
        personal_symptom_df = personal_symptom_df.merge(new_symptom_df, on="note_id", how="left")
        personal_symptom_df[symptom_column_name] = personal_symptom_df[symptom_column_name].map(lambda x: 1 if x == 'Positive' else (-1 if x == 'Negated' else np.nan))
    
    return personal_symptom_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2bedf996-10f5-4562-82d1-f32eab41e2cd"),
    broad_related_concepts=Input(rid="ri.foundry.main.dataset.ce8c17fb-63cd-41bd-b9c8-9bf54e5091da"),
    personal_symptom_testing=Input(rid="ri.foundry.main.dataset.68b9ff1f-582a-4c69-b70d-a321d56c5357")
)
def person_nlp_symptom_testing(personal_symptom_testing, broad_related_concepts):
    
    personal_symptom_df = personal_symptom_testing[['note_id', 'person_id', 'note_date', 'visit_occurrence_id']]
    personal_symptom_df = personal_symptom_df.drop_duplicates()
    personal_symptom_df = personal_symptom_df.set_index('note_id')
    all_symptoms_df = broad_related_concepts
    all_symptoms = list(set(all_symptoms_df["concept_set_name"]))

    print(all_symptoms)

    for symptom in all_symptoms:
        symptom_column_name = "sympt_" + symptom.replace("-", "_")
        new_symptom_df = personal_symptom_testing[['note_id', 'term_modifier_certainty', 'concept_set_name']]
        new_symptom_df = new_symptom_df.loc[new_symptom_df.concept_set_name == symptom]
        new_symptom_df = new_symptom_df.rename(columns={'term_modifier_certainty': symptom_column_name})
        new_symptom_df = new_symptom_df[["note_id", symptom_column_name]]
        new_symptom_df = new_symptom_df.drop_duplicates()
        
        personal_symptom_df = personal_symptom_df.merge(new_symptom_df, on="note_id", how="left")
        personal_symptom_df[symptom_column_name] = personal_symptom_df[symptom_column_name].map(lambda x: 1 if x == 'Positive' else (-1 if x == 'Negated' else np.nan))
    
    return personal_symptom_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.543e1d80-626e-4a3d-a196-d0c7b434fb41"),
    person=Input(rid="ri.foundry.main.dataset.f71ffe18-6969-4a24-b81c-0e06a1ae9316"),
    person_testing=Input(rid="ri.foundry.main.dataset.06629068-25fc-4802-9b31-ead4ed515da4")
)
def person_testing_copy(person_testing, person):
    return person_testing if LOAD_TEST == 1 else person

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.73f9d829-203f-4e2d-88d2-0d168503b0b1"),
    important_concepts=Input(rid="ri.foundry.main.dataset.31ae971a-0175-49bf-b7ec-31fd900e58f5"),
    personal_symptom=Input(rid="ri.foundry.main.dataset.4ba26dbd-1e38-48bd-8981-823339eb97f8")
)
def person_top_nlp_symptom(personal_symptom, important_concepts):
    
    personal_symptom_df = personal_symptom[['note_id', 'person_id', 'note_date', 'visit_occurrence_id']]
    personal_symptom_df = personal_symptom_df.drop_duplicates()
    personal_symptom_df = personal_symptom_df.set_index('note_id')
    all_symptoms_df = important_concepts
    all_symptoms = list(set(all_symptoms_df["concept_name"]))

    print(all_symptoms)

    for symptom in all_symptoms:
        symptom_column_name = symptom.replace(' ', '_').replace('-', '_').replace(',', '').replace('(', '').replace(')', '')
        new_symptom_df = personal_symptom[['note_id', 'term_modifier_certainty', 'concept_name']]

        new_symptom_df = new_symptom_df.loc[new_symptom_df.concept_name == symptom]
        new_symptom_df = new_symptom_df.rename(columns={'term_modifier_certainty': symptom_column_name})
        new_symptom_df = new_symptom_df[["note_id", symptom_column_name]]
        new_symptom_df = new_symptom_df.drop_duplicates()
        
        personal_symptom_df = personal_symptom_df.merge(new_symptom_df, on="note_id", how="left")
        personal_symptom_df[symptom_column_name] = personal_symptom_df[symptom_column_name].map(lambda x: 1 if x == 'Positive' else (-1 if x == 'Negated' else np.nan))
    
    return personal_symptom_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bd7a9446-5e51-495e-a9bb-002150ffb664"),
    important_concepts=Input(rid="ri.foundry.main.dataset.31ae971a-0175-49bf-b7ec-31fd900e58f5"),
    personal_symptom_testing=Input(rid="ri.foundry.main.dataset.68b9ff1f-582a-4c69-b70d-a321d56c5357")
)
def person_top_nlp_symptom_testing(personal_symptom_testing, important_concepts):
    
    personal_symptom_df = personal_symptom_testing[['note_id', 'person_id', 'note_date', 'visit_occurrence_id']]
    personal_symptom_df = personal_symptom_df.drop_duplicates()
    personal_symptom_df = personal_symptom_df.set_index('note_id')
    all_symptoms_df = important_concepts
    all_symptoms = list(set(all_symptoms_df["concept_name"]))

    print(all_symptoms)

    for symptom in all_symptoms:
        symptom_column_name = symptom.replace(' ', '_').replace('-', '_').replace(',', '').replace('(', '').replace(')', '')
        new_symptom_df = personal_symptom_testing[['note_id', 'term_modifier_certainty', 'concept_name']]

        new_symptom_df = new_symptom_df.loc[new_symptom_df.concept_name == symptom]
        new_symptom_df = new_symptom_df.rename(columns={'term_modifier_certainty': symptom_column_name})
        new_symptom_df = new_symptom_df[["note_id", symptom_column_name]]
        new_symptom_df = new_symptom_df.drop_duplicates()
        
        personal_symptom_df = personal_symptom_df.merge(new_symptom_df, on="note_id", how="left")
        personal_symptom_df[symptom_column_name] = personal_symptom_df[symptom_column_name].map(lambda x: 1 if x == 'Positive' else (-1 if x == 'Negated' else np.nan))
    
    return personal_symptom_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e5cca0dd-2ac9-42dd-886d-4c1358f19cd1"),
    note=Input(rid="ri.foundry.main.dataset.e1b6b20b-72c9-4eb9-8ff9-9e30a617ff5f"),
    note_nlp=Input(rid="ri.foundry.main.dataset.71d69f3c-3929-4703-848e-dcc17752e578")
)
#Purpose - The purpose of this pipeline is to produce an organized personal nlp processed notes.
#Creator/Owner/contact - Jiani Huang
#Last Update - 11/22/22

def personal_notes(note_nlp, note):
    
    person_notes_df = note.select('person_id', 'note_id', 'note_date', 'visit_occurrence_id')
    note_concept_df = note_nlp.select('note_nlp_id', 'note_id', 'term_modifier_certainty', 'note_nlp_concept_id', 'note_nlp_concept_name', )
    df = person_notes_df.join(note_concept_df, 'note_id', 'left')

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.1904146f-f481-4cc1-ab34-e242424af285"),
    personal_notes=Input(rid="ri.foundry.main.dataset.e5cca0dd-2ac9-42dd-886d-4c1358f19cd1")
)
#Purpose - The purpose of this pipeline is to produce only positive negative symtoms that the notes are certain about
#Creator/Owner/contact - Jiani Huang
#Last Update - 11/22/22

def personal_notes_pos_neg(personal_notes):
    positive_mask = personal_notes['term_modifier_certainty'].isin(['Positive'])
    negative_mask = personal_notes['term_modifier_certainty'].isin(['Negated'])
    pos_neg_mask = personal_notes['term_modifier_certainty'].isin(['Positive', 'Negated'])

    positive_personal_notes = personal_notes.filter(positive_mask)
    negative_personal_notes = personal_notes.filter(negative_mask)
    pos_neg_personal_notes = personal_notes.filter(pos_neg_mask)

    note_with_both_pos_neg = positive_personal_notes.alias('a').join(negative_personal_notes.alias('b'), (F.col("a.note_id") == F.col("b.note_id")) & (F.col("a.note_nlp_concept_id") == F.col("b.note_nlp_concept_id")), "inner").select(F.col("a.note_id"),F.col("a.note_nlp_concept_id")) 
    
    df = pos_neg_personal_notes.alias('a').join(note_with_both_pos_neg.alias('b'), (F.col("a.note_id") == F.col("b.note_id")) & (F.col("a.note_nlp_concept_id") == F.col("b.note_nlp_concept_id")), "left_outer")\
                 .where(F.col("b.note_id").isNull() & F.col("b.note_nlp_concept_id").isNull())\
                 .select([F.col(f"a.{c}") for c in pos_neg_personal_notes.columns]).distinct()

    # mask = personal_notes['term_modifier_certainty'].isin(['Positive', 'Negated'])
    # pos_neg_personal_notes = personal_notes.filter(mask)

    # # remove the cases where both positive and negative symptom occurs for the same note
    # same_note_symptom = pos_neg_personal_notes

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b5e24d84-8796-487c-a0d9-69f4a7d56bb0"),
    personal_notes_testing=Input(rid="ri.foundry.main.dataset.26396608-ae5e-498c-ac79-872b01e61199")
)
#Purpose - The purpose of this pipeline is to produce only positive negative symtoms that the notes are certain about
#Creator/Owner/contact - Jiani Huang
#Last Update - 11/22/22

def personal_notes_pos_neg_testing(personal_notes_testing):
    positive_mask = personal_notes_testing['term_modifier_certainty'].isin(['Positive'])
    negative_mask = personal_notes_testing['term_modifier_certainty'].isin(['Negated'])
    pos_neg_mask = personal_notes_testing['term_modifier_certainty'].isin(['Positive', 'Negated'])

    positive_personal_notes = personal_notes_testing.filter(positive_mask)
    negative_personal_notes = personal_notes_testing.filter(negative_mask)
    pos_neg_personal_notes = personal_notes_testing.filter(pos_neg_mask)

    note_with_both_pos_neg = positive_personal_notes.alias('a').join(negative_personal_notes.alias('b'), (F.col("a.note_id") == F.col("b.note_id")) & (F.col("a.note_nlp_concept_id") == F.col("b.note_nlp_concept_id")), "inner").select(F.col("a.note_id"),F.col("a.note_nlp_concept_id")) 
    
    df = pos_neg_personal_notes.alias('a').join(note_with_both_pos_neg.alias('b'), (F.col("a.note_id") == F.col("b.note_id")) & (F.col("a.note_nlp_concept_id") == F.col("b.note_nlp_concept_id")), "left_outer")\
                 .where(F.col("b.note_id").isNull() & F.col("b.note_nlp_concept_id").isNull())\
                 .select([F.col(f"a.{c}") for c in pos_neg_personal_notes.columns]).distinct()

    # mask = personal_notes['term_modifier_certainty'].isin(['Positive', 'Negated'])
    # pos_neg_personal_notes = personal_notes.filter(mask)

    # # remove the cases where both positive and negative symptom occurs for the same note
    # same_note_symptom = pos_neg_personal_notes

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.26396608-ae5e-498c-ac79-872b01e61199"),
    note_nlp_testing=Input(rid="ri.foundry.main.dataset.9c668691-6880-4da9-88bf-79196c3e0f5a"),
    note_testing=Input(rid="ri.foundry.main.dataset.f841b321-04d3-4119-85c7-9bde2883f64c")
)
#Purpose - The purpose of this pipeline is to produce an organized personal nlp processed notes.
#Creator/Owner/contact - Jiani Huang
#Last Update - 11/22/22

def personal_notes_testing(note_nlp_testing, note_testing):
    
    person_notes_df = note_testing.select('person_id', 'note_id', 'note_date', 'visit_occurrence_id')
    note_concept_df = note_nlp_testing.select('note_nlp_id', 'note_id', 'term_modifier_certainty', 'note_nlp_concept_id', 'note_nlp_concept_name', )
    df = person_notes_df.join(note_concept_df, 'note_id', 'left')

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4ba26dbd-1e38-48bd-8981-823339eb97f8"),
    personal_notes_pos_neg=Input(rid="ri.foundry.main.dataset.1904146f-f481-4cc1-ab34-e242424af285"),
    related_concept=Input(rid="ri.foundry.main.dataset.73261589-80e1-4205-a060-e1d6eeb83d55")
)
def personal_symptom(personal_notes_pos_neg, related_concept):
    
    personal_notes_pos_neg_df = personal_notes_pos_neg.select('*')
    related_concept_df = related_concept.select('*')
    df = personal_notes_pos_neg_df.join(related_concept_df, personal_notes_pos_neg_df.note_nlp_concept_id == related_concept_df.concept_id, 'inner')

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.68b9ff1f-582a-4c69-b70d-a321d56c5357"),
    personal_notes_pos_neg_testing=Input(rid="ri.foundry.main.dataset.b5e24d84-8796-487c-a0d9-69f4a7d56bb0"),
    related_concept=Input(rid="ri.foundry.main.dataset.73261589-80e1-4205-a060-e1d6eeb83d55")
)
def personal_symptom_testing(personal_notes_pos_neg_testing, related_concept):
    
    personal_notes_pos_neg_df = personal_notes_pos_neg_testing.select('*')
    related_concept_df = related_concept.select('*')
    df = personal_notes_pos_neg_df.join(related_concept_df, personal_notes_pos_neg_df.note_nlp_concept_id == related_concept_df.concept_id, 'inner')

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7cd7ede8-524b-4fe3-8ce1-10f268fcd51b"),
    personal_symptom=Input(rid="ri.foundry.main.dataset.4ba26dbd-1e38-48bd-8981-823339eb97f8")
)
def positive_symptoms(personal_symptom):

    rslt_df = personal_symptom[personal_symptom['term_modifier_certainty'] == "Positive"] 
    return rslt_df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2d76588c-fe75-4d07-8044-f054444ec728"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3"),
    procedure_occurrence_testing=Input(rid="ri.foundry.main.dataset.88523aaa-75c3-4b55-a79a-ebe27e40ba4f")
)
def procedure_occurrence_testing_copy(procedure_occurrence_testing, procedure_occurrence):
    return procedure_occurrence_testing if LOAD_TEST == 1 else procedure_occurrence

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.166746be-24ec-4f37-84e0-141c8e56706b"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3")
)
def procedure_table_analysis_1(procedure_occurrence, Long_COVID_Silver_Standard):
    TABLE = procedure_occurrence
    CONCEPT_NAME_COL = "procedure_concept_name"
    l, h = 0, 750

    label = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(*["pasc_code_after_four_weeks", "pasc_code_prior_four_weeks"]))
    TABLE = TABLE.join(label, "person_id").select(F.col("person_id"), F.col(CONCEPT_NAME_COL), F.col("outcome")).filter(F.col(CONCEPT_NAME_COL) != "No matching concept")
    distinct = TABLE.groupBy(CONCEPT_NAME_COL).count().orderBy("count", ascending=False).limit(h).select(F.col(CONCEPT_NAME_COL)).toPandas()[CONCEPT_NAME_COL]
    
    pos, count, ppl, ppl_pos = [], [], [], []
    cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
    print(len(distinct))
    t = time.time()
    for cname in distinct:
        f = TABLE.agg(
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname)),
            cnt_cond((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1)),
            F.countDistinct(F.when(F.col(CONCEPT_NAME_COL) == cname, F.col("person_id")).otherwise(None)),
            F.countDistinct(F.when(((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1)), F.col("person_id")).otherwise(None)),
        ).collect()
        one_count = f[0][1]
        size = f[0][0]
        people_count =f[0][2]
        people_one = f[0][3]
        pos.append(one_count/size)
        count.append(size)
        ppl.append(people_count)
        ppl_pos.append(people_one/people_count)
    print(time.time() - t)
    r = pd.DataFrame(list(zip(distinct,pos, count, ppl, ppl_pos)), columns=["concept_name","encounter_pos", "encounter_count", "people_count", "people_pos"])
    r['domain'] = CONCEPT_NAME_COL
    r["heuristic_count"] = r.apply(lambda row: row["encounter_pos"] * row["encounter_count"] if row["encounter_pos"] > 0.7 else 0, axis=1)
    r["heuristic_person"] = r.apply(lambda row: row["people_pos"] * row["people_count"] if row["people_pos"] > 0.7 else 0, axis=1)
    r =r[["concept_name", 'domain', 'encounter_pos','people_pos','encounter_count', 'people_count','heuristic_count','heuristic_person']]
    
    return r

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ae1c108c-1813-47ba-831c-e5a37c599c49"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    person_information=Input(rid="ri.foundry.main.dataset.2f6ebf73-3a2d-43dc-ace9-da56da4b1743"),
    recent_visits_w_nlp_notes_2=Input(rid="ri.foundry.main.dataset.fc6afa83-8c7a-4b04-a92a-ff1162651b0b"),
    train_valid_split=Input(rid="ri.foundry.main.dataset.9d1a79f6-7627-4ee4-abc0-d6d6179c2f26")
)
def produce_dataset(train_valid_split, Long_COVID_Silver_Standard, person_information, recent_visits_w_nlp_notes_2):
    print("start")

    write_to_pickle([torch.zeros(10), torch.ones(10)], "sample_data")
    # First get the splitted person ids
    train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")

    train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # print(train_recent_visits.show())

    train_person_info = train_person_ids.join(person_information, on="person_id")
    valid_person_info = valid_person_ids.join(person_information, on="person_id")

    print("start pre-processing!!!")
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, person_ids = pre_processing_visits(train_person_ids.toPandas(), train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both", return_person_ids = True)

    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, valid_person_ids = pre_processing_visits(valid_person_ids.toPandas(), valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both", return_person_ids = True)
    print("finish pre-processing!!!")

    visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)
    print("non empty column ids count::", len(non_empty_column_ids))

    data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)
    write_to_pickle([non_empty_column_ids, data_min, data_max], "train_data_statistics")
    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    valid_subset_ids = list(range(10))

    subset_valid_visit_tensor_ls = [valid_visit_tensor_ls[idx] for idx in valid_subset_ids]    
    subset_valid_mask_ls = [valid_mask_ls[idx] for idx in valid_subset_ids]    
    subset_valid_time_step_ls = [valid_time_step_ls[idx] for idx in valid_subset_ids]    
    subset_valid_person_info_ls = [valid_person_info_ls[idx] for idx in valid_subset_ids]    
    subset_valid_label_tensor_ls = [valid_label_tensor_ls[idx] for idx in valid_subset_ids]    

    subset_valid_dataset = LongCOVIDVisitsDataset2(subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max)
    write_to_pickle([person_ids, visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max], "train_data")
    write_to_pickle([valid_person_ids, valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max], "valid_data")
    
    write_to_pickle([subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max], "subset_valid_data")

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ca457d29-b952-4b66-afd9-01506158c1a0"),
    person_information_testing=Input(rid="ri.foundry.main.dataset.f3f8c10f-3926-480e-be48-d2bc585f2c04"),
    produce_dataset=Input(rid="ri.foundry.main.dataset.ae1c108c-1813-47ba-831c-e5a37c599c49"),
    recent_visits_w_nlp_notes_2_testing=Input(rid="ri.foundry.main.dataset.867bc946-668a-428a-83b7-67301a762c95")
)
def produce_dataset_testing(person_information_testing, recent_visits_w_nlp_notes_2_testing, produce_dataset):
    print("start")

    test_recent_visits = recent_visits_w_nlp_notes_2_testing

    test_person_info = person_information_testing

    print("start pre-processing!!!")
    test_visit_tensor_ls, test_mask_ls, test_time_step_ls, test_person_info_ls, test_label_tensor_ls, test_person_ids = pre_processing_visits(None, test_person_info.toPandas(), test_recent_visits.toPandas(), None, setup="both", return_person_ids = True, start_col_id=4)

    # valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits(valid_person_ids.toPandas(), valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both")
    print("finish pre-processing!!!")

    non_empty_column_ids, data_min, data_max = read_from_pickle(produce_dataset, "train_data_statistics.pickle")

    # visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    print("non empty column ids::", non_empty_column_ids)
    test_visit_tensor_ls, test_mask_ls = remove_empty_columns_with_non_empty_cls(test_visit_tensor_ls, test_mask_ls, non_empty_column_ids)

    # data_min, data_max = get_data_min_max(test_visit_tensor_ls, test_mask_ls)

    test_dataset = LongCOVIDVisitsDataset2(test_visit_tensor_ls, test_mask_ls, test_time_step_ls, test_person_info_ls, None, data_min, data_max)

    # valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    # valid_subset_ids = list(range(10))

    # subset_valid_visit_tensor_ls = [valid_visit_tensor_ls[idx] for idx in valid_subset_ids]    
    # subset_valid_mask_ls = [valid_mask_ls[idx] for idx in valid_subset_ids]    
    # subset_valid_time_step_ls = [valid_time_step_ls[idx] for idx in valid_subset_ids]    
    # subset_valid_person_info_ls = [valid_person_info_ls[idx] for idx in valid_subset_ids]    
    # subset_valid_label_tensor_ls = [valid_label_tensor_ls[idx] for idx in valid_subset_ids]    

    # subset_valid_dataset = LongCOVIDVisitsDataset2(subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max)
    # write_to_pickle([visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max], "train_data")
    write_to_pickle([test_person_ids, test_visit_tensor_ls, test_mask_ls, test_time_step_ls, test_person_info_ls, test_label_tensor_ls, data_min, data_max], "test_data")
    
    # write_to_pickle([subset_valid_visit_tensor_ls, subset_valid_mask_ls, subset_valid_time_step_ls, subset_valid_person_info_ls, subset_valid_label_tensor_ls, data_min, data_max], "subset_valid_data")

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
    Output(rid="ri.foundry.main.dataset.bf18056e-2e27-4f2a-af1a-7b6cabc2a9cf"),
    all_patients_visit_day_facts_table_de_id=Input(rid="ri.foundry.main.dataset.ace57213-685a-4f18-a157-2b02b41086be")
)
from pyspark.sql.functions import col, max as max_, min as min_
from pyspark.sql.functions import datediff
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F 
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
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.da8e41a5-92ce-4bc6-be6c-ff808dec67c9"),
    all_patients_visit_day_facts_table_de_id_testing=Input(rid="ri.foundry.main.dataset.7ace5232-cf55-4095-bb84-35ae2f2350ab")
)
from pyspark.sql.functions import col, max as max_, min as min_
from pyspark.sql.functions import datediff
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F 
def recent_visits_2_testing(all_patients_visit_day_facts_table_de_id_testing):
    # First sort the visits
    # all_patients_visit_day_facts_table_de_id_testing = all_patients_visit_day_facts_table_de_id_testing.sort_values(["person_id", "visit_date"])

    all_patients_visit_day_facts_table_de_id_testing = all_patients_visit_day_facts_table_de_id_testing.orderBy(all_patients_visit_day_facts_table_de_id_testing.person_id, all_patients_visit_day_facts_table_de_id_testing.visit_date)

    # Get the number of visits
    # num_visits = all_patients_visit_day_facts_table_de_id_testing \
    #     .groupby('person_id')['visit_date'] \
    #     .nunique() \
    #     .reset_index() \
    #     .rename(columns={"visit_date": "num_visits"})

    # The maximum number of visits is around 1000
    # print(num_visits.max())

    # Get the last visit of each patient
    # print(all_patients_visit_day_facts_table_de_id_testing.groupBy("person_id").agg(max_("visit_date")).show())
    last_visit = all_patients_visit_day_facts_table_de_id_testing \
        .groupBy("person_id") \
        .agg(max_("visit_date")).withColumnRenamed("max(visit_date)", "last_visit_date") \
        # .reset_index("person_id") \
        # .rename(columns={"visit_date": "last_visit_date"})
    
    # Add a six-month before the last visit column to the dataframe
    # last_visit["six_month_before_last_visit"] = last_visit["last_visit_date"].map(lambda x: x - pd.Timedelta(days=180))
    last_visit = last_visit.withColumn("six_month_before_last_visit", F.date_sub(last_visit["last_visit_date"], 180))
    # print(last_visit.show())

    # Merge last_visit back
    # df = all_patients_visit_day_facts_table_de_id_testing.merge(last_visit, on="person_id", how="left")
    df = all_patients_visit_day_facts_table_de_id_testing.join(last_visit, on = "person_id", how = "left")

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
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f3ea4ea5-6240-4bd5-a633-b5ea0628226e"),
    person_nlp_symptom=Input(rid="ri.foundry.main.dataset.d3124557-f100-44a9-9c43-51db33c87bd8"),
    recent_visits=Input(rid="ri.foundry.main.dataset.d42a9b47-a06f-4d43-b113-7414a1bdb9b6")
)
def recent_visits_w_nlp_notes(recent_visits, person_nlp_symptom):
    person_nlp_symptom = person_nlp_symptom
    person_nlp_symptom = person_nlp_symptom.merge(recent_visits[["person_id", "six_month_before_last_visit"]].drop_duplicates(), on="person_id", how="left")
    person_nlp_symptom = person_nlp_symptom.loc[person_nlp_symptom["note_date"] >= person_nlp_symptom["six_month_before_last_visit"]]
    person_nlp_symptom = person_nlp_symptom.rename(columns={"note_date": "visit_date"}).drop(columns=["six_month_before_last_visit", "note_id", "visit_occurrence_id"])
    person_nlp_symptom["has_nlp_note"] = 1.0

    # Make sure type checks
    df = recent_visits.merge(person_nlp_symptom, on=["person_id", "visit_date"], how="left").sort_values(["person_id", "visit_date"]).fillna(0.0)

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.fc6afa83-8c7a-4b04-a92a-ff1162651b0b"),
    person_nlp_symptom=Input(rid="ri.foundry.main.dataset.d3124557-f100-44a9-9c43-51db33c87bd8"),
    recent_visits_2=Input(rid="ri.foundry.main.dataset.bf18056e-2e27-4f2a-af1a-7b6cabc2a9cf")
)
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

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.867bc946-668a-428a-83b7-67301a762c95"),
    person_nlp_symptom_testing=Input(rid="ri.foundry.main.dataset.2bedf996-10f5-4562-82d1-f32eab41e2cd"),
    recent_visits_2_testing=Input(rid="ri.foundry.main.dataset.da8e41a5-92ce-4bc6-be6c-ff808dec67c9")
)
def recent_visits_w_nlp_notes_2_testing(recent_visits_2_testing, person_nlp_symptom_testing):
    person_nlp_symptom_testing = person_nlp_symptom_testing.join(recent_visits_2_testing.select(["person_id", "six_month_before_last_visit"]).distinct(), on="person_id", how="left")
    person_nlp_symptom_testing = person_nlp_symptom_testing.where(person_nlp_symptom_testing["note_date"] >= person_nlp_symptom_testing["six_month_before_last_visit"])
    person_nlp_symptom_testing = person_nlp_symptom_testing.withColumnRenamed("note_date", "visit_date").drop(*["six_month_before_last_visit", "note_id", "visit_occurrence_id"])
    person_nlp_symptom_testing = person_nlp_symptom_testing.withColumn("has_nlp_note", lit(1.0))
    df = recent_visits_2_testing.join(person_nlp_symptom_testing, on = ["person_id", "visit_date"], how="left").orderBy(*["person_id", "visit_date"])#.fillna(0.0)

    # person_nlp_symptom_testing = person_nlp_symptom_testing.merge(recent_visits_2[["person_id", "six_month_before_last_visit"]].drop_duplicates(), on="person_id", how="left")
    # person_nlp_symptom_testing = person_nlp_symptom_testing.loc[person_nlp_symptom_testing["note_date"] >= person_nlp_symptom_testing["six_month_before_last_visit"]]
    # person_nlp_symptom_testing = person_nlp_symptom_testing.rename(columns={"note_date": "visit_date"}).drop(columns=["six_month_before_last_visit", "note_id", "visit_occurrence_id"])
    # person_nlp_symptom_testing["has_nlp_note"] = 1.0

    # # Make sure type checks
    # df = recent_visits_2.merge(person_nlp_symptom_testing, on=["person_id", "visit_date"], how="left").sort_values(["person_id", "visit_date"])#.fillna(0.0)

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.73261589-80e1-4205-a060-e1d6eeb83d55"),
    broad_related_concepts=Input(rid="ri.foundry.main.dataset.ce8c17fb-63cd-41bd-b9c8-9bf54e5091da"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6")
)
def related_concept(concept_set_members, broad_related_concepts):
    related_concepts = broad_related_concepts
    
    concept_set_members_df = concept_set_members.select("codeset_id",  "concept_id", "concept_name",)
    related_concepts_df = related_concepts.select("codeset_id", "concept_set_name",)
    df = concept_set_members_df.join(related_concepts_df, "codeset_id", 'right')
    return df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.48f1d68a-4344-4b76-ae4e-7fadb12349ea"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    all_patients_summary_fact_table_de_id=Input(rid="ri.foundry.main.dataset.324a6115-7c17-4d4d-94da-a2df11a87fa6")
)
def rf_hp_tuning(Long_COVID_Silver_Standard, all_patients_summary_fact_table_de_id):
    static_cols = ['person_id','total_visits', 'age']

    cols = static_cols + [col for col in all_patients_summary_fact_table_de_id.columns if 'indicator' in col]
    
    ## get outcome column
    Long_COVID_Silver_Standard["outcome"] = Long_COVID_Silver_Standard.apply(lambda x: max([x["pasc_code_after_four_weeks"], x["pasc_code_prior_four_weeks"]]), axis=1)
    Outcome_df = all_patients_summary_fact_table_de_id[["person_id"]].merge(Long_COVID_Silver_Standard, on="person_id", how="left")
    Outcome_df = Outcome_df[["person_id", "outcome"]].sort_values('person_id')
    Outcome = list(Outcome_df["outcome"])

            
    Training_and_Holdout = all_patients_summary_fact_table_de_id[cols].fillna(0.0).sort_values('person_id')
    X_train_no_ind, X_test_no_ind, y_train, y_test = train_test_split(Training_and_Holdout, Outcome, train_size=0.9, random_state=1)
    X_train, X_test = X_train_no_ind.set_index("person_id"), X_test_no_ind.set_index("person_id")
    X, y = Training_and_Holdout.set_index("person_id"), Outcome

    model = RandomForestClassifier()
    n_estimators = [100,200,300,400,500,600]
    bootstrap = [True, False]
    n_jobs=[-1]
    class_weight = [None,'balanced','balanced_subsample']
    # Create the random grid
    grid = {'n_estimators': n_estimators,
            'bootstrap': bootstrap,
            'n_jobs' : n_jobs,
            'class_weight' : class_weight
            }

    cvFold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    randomSearch = RandomizedSearchCV(estimator=model, n_iter=60, n_jobs=-1, cv=cvFold, param_distributions=grid,verbose=2,refit=True, scoring='f1').fit(X,y)
    params=randomSearch.best_estimator_.get_params()
    print("All params: \n", params)

    new_model = RandomForestClassifier(**params).fit(X_train,y_train)

    test_preds = new_model.predict_proba(X_test)[:, 1]
    
    predictions = pd.DataFrame.from_dict({
        'person_id': X_test_no_ind["person_id"],
        'pred_outcome': test_preds.tolist(),
        'outcome': y_test
    }, orient='columns')

    print("Classification Report:\n{}".format(classification_report(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0))))

    print("MAE:", mean_absolute_error(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("Brier score:", brier_score_loss(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("AP:", average_precision_score(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))
    print("ROC AUC:", roc_auc_score(predictions['outcome'], np.where(predictions['pred_outcome'] > 0.5, 1, 0)))

    return predictions

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.235be874-5669-4c42-9ae3-3e6d37b645e1"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2"),
    device_exposure=Input(rid="ri.foundry.main.dataset.c1fd6d67-fc80-4747-89ca-8eb04efcb874"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5"),
    measurement=Input(rid="ri.foundry.main.dataset.5c8b84fb-814b-4ee5-a89a-9525f4a617c7"),
    observation=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3"),
    train_test_model=Input(rid="ri.foundry.main.dataset.ea6c836a-9d51-4402-b1b7-0e30fb514fc8")
)
def study_misclassified(train_test_model, procedure_occurrence, condition_occurrence, drug_exposure, observation, device_exposure, measurement):
    tables = {measurement:"measurement_concept_name"}#{procedure_occurrence:"procedure_concept_name",condition_occurrence:"condition_concept_name", drug_exposure:"drug_concept_name", observation:"observation_concept_name", device_exposure:"device_concept_name",}
    top_k = 50
    
    ret_tables = []
    for TABLE, CONCEPT_NAME_COL in tables.items():
        TABLE = TABLE.join(train_test_model, "person_id").filter((F.col(CONCEPT_NAME_COL) != "No matching concept") & (F.col("ens_outcome") != F.col("outcome"))).select(F.col("person_id"), F.col(CONCEPT_NAME_COL), F.col("outcome"))
        distinct = TABLE.groupBy(CONCEPT_NAME_COL).count().orderBy("count", ascending=False).limit(top_k).select(F.col(CONCEPT_NAME_COL)).toPandas()[CONCEPT_NAME_COL]
        
        pos, count, ppl, ppl_pos = [], [], [], []
        cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
        print(len(distinct))
        t = time.time()
        for cname in distinct:
            f = TABLE.agg(
                cnt_cond((F.col(CONCEPT_NAME_COL) == cname)),
                cnt_cond((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1)),
                F.countDistinct(F.when(F.col(CONCEPT_NAME_COL) == cname, F.col("person_id")).otherwise(None)),
                F.countDistinct(F.when(((F.col(CONCEPT_NAME_COL) == cname) & (F.col("outcome") == 1)), F.col("person_id")).otherwise(None)),
            ).collect()
            one_count = f[0][1]
            size = f[0][0]
            people_count =f[0][2]
            people_one = f[0][3]
            pos.append(one_count/size)
            count.append(size)
            ppl.append(people_count)
            ppl_pos.append(people_one/people_count)
        print(time.time() - t)
        r = pd.DataFrame(list(zip(distinct,pos, count, ppl, ppl_pos)), columns=["concept_name","encounter_pos", "encounter_count", "people_count", "people_pos"])
        r['domain'] = CONCEPT_NAME_COL
        r["heuristic_count"] = r.apply(lambda row: row["encounter_pos"] * row["encounter_count"], axis=1)
        r["heuristic_person"] = r.apply(lambda row: row["people_pos"] * row["people_count"], axis=1)
        r =r[["concept_name", 'domain', 'encounter_pos','people_pos','encounter_count', 'people_count','heuristic_count','heuristic_person']]
        ret_tables.append(r)
    ret =  pd.concat(ret_tables)
    return ret
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c6673c1c-6c0c-4ff1-8b29-cc4c268e650a"),
    produce_dataset_testing=Input(rid="ri.foundry.main.dataset.ca457d29-b952-4b66-afd9-01506158c1a0"),
    train_sequential_model_3=Input(rid="ri.foundry.main.dataset.4fa4a34a-a9e7-489f-a499-023c2d4c44ac")
)
def test_mTan(train_sequential_model_3, produce_dataset_testing):
    test_person_ids, test_visit_tensor_ls, test_mask_ls, test_time_step_ls, test_person_info_ls, _, data_min, data_max = read_from_pickle(produce_dataset_testing, "test_data.pickle")

    test_dataset = LongCOVIDVisitsDataset2(test_visit_tensor_ls, test_mask_ls, test_time_step_ls, test_person_info_ls, None, data_min, data_max)

    static_input_dim = test_dataset.__getitem__(1)[3].shape[-1]

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    dim = test_dataset.__getitem__(1)[0].shape[-1]
    latent_dim=20
    rec_hidden=32
    learn_emb=True
    enc_num_heads=1
    num_ref_points=128
    gen_hidden=30
    dec_num_heads=1
    classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    lr = 0.001

    

    rec = rec.to(device)
    dec = dec.to(device)
    classifier = classifier.to(device)

    rec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_rec.pickle'))
    dec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_dec.pickle'))
    classifier.load_state_dict(read_model_from_pickle(train_sequential_model_3, "mTans_classifier.pickle"))

    
    test_pred_labels, pred_scores =  test_classifier(rec, test_loader,latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)

    test_predictions = pd.DataFrame.from_dict({
            'person_id': test_person_ids,
            'mTans': pred_scores.reshape(-1).tolist()
        }, orient='columns')

    return test_predictions

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
    Output(rid="ri.foundry.main.dataset.e5cacf25-d71d-44ee-a3da-6bad8eaf53e8"),
    nlp_sym_analysis=Input(rid="ri.foundry.main.dataset.02f756f5-d406-4b23-9438-a2d5a5c17cd9")
)
def top_concept_names(nlp_sym_analysis):
    r = nlp_sym_analysis.withColumn("scale_above_threshold", F.when((F.col("max") > 0.65), F.col("max")*F.col("count")).otherwise(F.lit(0)))
    return r

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7b277d99-e39e-4a5f-9058-4e6f65fa7f58"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.2f496793-6a4e-4bf4-b0fc-596b277fb7e2"),
    device_exposure=Input(rid="ri.foundry.main.dataset.c1fd6d67-fc80-4747-89ca-8eb04efcb874"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.469b3181-6336-4d0e-8c11-5e33a99876b5"),
    everyone_cohort_de_id=Input(rid="ri.foundry.main.dataset.120adc97-2986-4b7d-9f96-42d8b5d5bedf"),
    measurement=Input(rid="ri.foundry.main.dataset.5c8b84fb-814b-4ee5-a89a-9525f4a617c7"),
    observation=Input(rid="ri.foundry.main.dataset.f9d8b08e-3c9f-4292-b603-f1bfa4336516"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.9a13eb06-de7d-482b-8f91-fb8c144269e3")
)
def top_k_concepts_data(observation, condition_occurrence, drug_exposure, procedure_occurrence, measurement, device_exposure, everyone_cohort_de_id):
    tables = {procedure_occurrence:("procedure_concept_id",1000),condition_occurrence:("condition_concept_id",1000), drug_exposure:("drug_concept_id",1000), observation:("observation_concept_id",1000), measurement:("measurement_concept_id",1000), device_exposure:("device_concept_id",1000)}

    feats = everyone_cohort_de_id.select(F.col("person_id"))
    for TABLE, (CONCEPT_ID_COL,k) in tables.items():
        TABLE = TABLE.select(F.col("person_id"), F.col(CONCEPT_ID_COL))                 
        distinct = TABLE.groupBy(CONCEPT_ID_COL).count().orderBy("count", ascending=False).limit(k).select(F.col(CONCEPT_ID_COL)).toPandas()[CONCEPT_ID_COL].tolist()
        df = TABLE.filter(F.col(CONCEPT_ID_COL).isin(distinct))
        df= df.groupBy("person_id").pivot(CONCEPT_ID_COL).agg(F.lit(1)).na.fill(0)
        df = df.select([F.col(c).alias(CONCEPT_ID_COL[:3]+c) if c != "person_id" else c for c in df.columns ])
        feats = feats.join(df, how="left",on="person_id")
    data = feats.na.fill(0)
    
    return data
    
        
    

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.10a82ffa-f748-4e12-9c88-0f8fc74dcd7f"),
    condition_occurrence_testing_copy=Input(rid="ri.foundry.main.dataset.a32b3d71-226c-4347-aaed-2c4900e2f4fb"),
    device_exposure_testing_copy=Input(rid="ri.foundry.main.dataset.ca1772cd-c245-453d-ac74-d0c42e490f2e"),
    drug_exposure_testing_copy=Input(rid="ri.foundry.main.dataset.6223d2b6-e8b8-4d48-8c4c-81dd2959d131"),
    everyone_cohort_de_id_testing=Input(rid="ri.foundry.main.dataset.4f510f7a-bb5b-455d-bb9d-7bcbae1a37b4"),
    measurement_testing_copy=Input(rid="ri.foundry.main.dataset.92566631-b0d5-4fab-8a14-5c3d0d6ad560"),
    observation_testing_copy=Input(rid="ri.foundry.main.dataset.9834f887-c80e-478c-b538-5bb39b9db6a7"),
    procedure_occurrence_testing_copy=Input(rid="ri.foundry.main.dataset.2d76588c-fe75-4d07-8044-f054444ec728"),
    top_k_concepts_data=Input(rid="ri.foundry.main.dataset.7b277d99-e39e-4a5f-9058-4e6f65fa7f58")
)
def top_k_concepts_data_test(device_exposure_testing_copy, procedure_occurrence_testing_copy, observation_testing_copy, drug_exposure_testing_copy, measurement_testing_copy, condition_occurrence_testing_copy, everyone_cohort_de_id_testing, top_k_concepts_data):
    tables = {procedure_occurrence_testing_copy:("procedure_concept_id",1000),condition_occurrence_testing_copy:("condition_concept_id",1000), drug_exposure_testing_copy:("drug_concept_id",1000), observation_testing_copy:("observation_concept_id",1000), measurement_testing_copy:("measurement_concept_id",1000), device_exposure_testing_copy:("device_concept_id",1000)}

    feats = everyone_cohort_de_id_testing.select(F.col("person_id"))
    for TABLE, (CONCEPT_ID_COL,k) in tables.items():
        TABLE = TABLE.select(F.col("person_id"), F.col(CONCEPT_ID_COL))                 
        distinct = [i[3:] for i in top_k_concepts_data.columns if i[:3] == CONCEPT_ID_COL[:3]]
        df = TABLE.filter(F.col(CONCEPT_ID_COL).isin(distinct))
        df= df.groupBy("person_id").pivot(CONCEPT_ID_COL).agg(F.lit(1)).na.fill(0)
        df = df.select([F.col(c).alias(CONCEPT_ID_COL[:3]+c) if c != "person_id" else c for c in df.columns ])
        feats = feats.join(df, how="left",on="person_id")
    data = feats.na.fill(0)
    
    return data
    
        
    

    

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
    Output(rid="ri.vector.main.execute.806be09f-422c-48e3-a40c-f88e3ca4d0ee"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    person_information=Input(rid="ri.foundry.main.dataset.2f6ebf73-3a2d-43dc-ace9-da56da4b1743"),
    recent_visits_w_nlp_notes_2=Input(rid="ri.foundry.main.dataset.fc6afa83-8c7a-4b04-a92a-ff1162651b0b"),
    train_valid_split=Input(rid="ri.foundry.main.dataset.9d1a79f6-7627-4ee4-abc0-d6d6179c2f26")
)
def train_sequential_model_2(train_valid_split, Long_COVID_Silver_Standard, person_information, recent_visits_w_nlp_notes_2):
    print("start")
    # First get the splitted person ids
    train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")

    train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # print(train_recent_visits.show())

    train_person_info = train_person_ids.join(person_information, on="person_id")
    valid_person_info = valid_person_ids.join(person_information, on="person_id")
    # train_person_ids = train_valid_split.loc[train_valid_split["split"] == "train"]
    # valid_person_ids = train_valid_split.loc[train_valid_split["split"] == "valid"]

    # Use it to split the data into training x/y and validation x/y
    # train_recent_visits = train_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")

    # Basic person information
    # train_person_info = train_person_ids.merge(person_information, on="person_id")
    # valid_person_info = valid_person_ids.merge(person_information, on="person_id")
    

    print("start pre-processing!!!")
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits(train_person_ids.toPandas(), train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both")
    # torch.save(visit_tensor_ls, "train_visit_tensor_ls")
    # torch.save(mask_ls, "train_mask_ls")
    # torch.save(time_step_ls, "train_mask_ls")
    # torch.save(label_tensor_ls, "train_label_tensor_ls")
    # torch.save(person_info_ls, "train_person_info_ls")
    
    # visit_tensor_ls, mask_ls = remove_empty_columns(visit_tensor_ls, mask_ls)

    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits(valid_person_ids.toPandas(), valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both")
    print("finish pre-processing!!!")
    visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)

    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)
    # train_dataset = LongCOVIDVisitsDataset2(train_person_ids, train_person_info, train_recent_visits, train_labels)
    # valid_dataset = LongCOVIDVisitsDataset2(valid_person_ids, valid_person_info, valid_recent_visits, valid_labels)

    # Construct dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)

    epochs=1
    print(train_dataset.__getitem__(1)[0])
    dim = train_dataset.__getitem__(1)[0].shape[-1]
    static_input_dim = train_dataset.__getitem__(1)[3].shape[-1]
    print("data shape::", train_dataset.__getitem__(1)[0].shape)
    print("mask shape::", train_dataset.__getitem__(1)[1].shape)
    print("dim::", dim)
    print(data_min)
    latent_dim=20
    rec_hidden=32
    learn_emb=True
    enc_num_heads=1
    num_ref_points=128
    gen_hidden=30
    dec_num_heads=1
    classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    lr = 0.0005

    rec = rec.to(device)
    dec = dec.to(device)
    classifier = classifier.to(device)

    best_valid_true, best_valid_pred_labels, best_train_true, best_train_pred_labels = train_mTans(lr, True, 0.01, 100, 1, dim, latent_dim, rec, dec, classifier, epochs, train_loader, valid_loader, is_kl=True)
    print("train true::", best_train_true)
    print("train pred labels::",best_train_pred_labels)
    incorrect_labeled_train_ids = torch.from_numpy(np.nonzero(best_train_true.reshape(-1) != best_train_pred_labels.reshape(-1))[0])
    incorrect_sample_weight=3
    train_set_weight = torch.ones(len(train_dataset))
    train_set_weight[incorrect_labeled_train_ids] = incorrect_sample_weight
    train_set_weight = train_set_weight/incorrect_sample_weight
    sampler = torch.utils.data.WeightedRandomSampler(train_set_weight, len(train_set_weight))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=sampler, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    print("start reweighted training::")

    classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    rec = rec.to(device)
    dec = dec.to(device)
    classifier = classifier.to(device)
    train_mTans(lr, True, 0.01, 100, 1, dim, latent_dim, rec, dec, classifier, epochs, train_loader, valid_loader, is_kl=True)

    # #computing shapley values
    # data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    # evaluate_shapley_value(data_loader, rec, dec, classifier, latent_dim, 1, dim, device)

    # # Construct the two datasets
    # train_dataset = LongCOVIDVisitsDataset(train_person_ids, train_person_info, train_recent_visits, train_labels)
    # valid_dataset = LongCOVIDVisitsDataset(valid_person_ids, valid_person_info, valid_recent_visits, valid_labels)

    # # Construct dataloaders
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=LongCOVIDVisitsDataset.collate_fn)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True, collate_fn=LongCOVIDVisitsDataset.collate_fn)

    # # Construct model
    # model = LongCOVIDVisitsLSTMModel()

    # # # Training loop
    # trainer = Trainer(train_loader, valid_loader, model)
    # result_model = trainer.train()

    # return train_person_ids

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.4fa4a34a-a9e7-489f-a499-023c2d4c44ac"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    person_information=Input(rid="ri.foundry.main.dataset.2f6ebf73-3a2d-43dc-ace9-da56da4b1743"),
    recent_visits_w_nlp_notes_2=Input(rid="ri.foundry.main.dataset.fc6afa83-8c7a-4b04-a92a-ff1162651b0b"),
    train_valid_split=Input(rid="ri.foundry.main.dataset.9d1a79f6-7627-4ee4-abc0-d6d6179c2f26")
)
def train_sequential_model_3(train_valid_split, Long_COVID_Silver_Standard, person_information, recent_visits_w_nlp_notes_2):
    print("start")
    # dim=10
    # latent_dim=20
    # rec_hidden=32
    # learn_emb=True
    # enc_num_heads=1
    # num_ref_points=128
    # gen_hidden=30
    # dec_num_heads=1
    # static_input_dim = 10
    # classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available() else 'cpu')
    # print("device::", device)
    # rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    # dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    # lr = 0.0005

    # write_to_pickle(rec.state_dict(), "mTans_rec")
    # write_to_pickle(dec.state_dict(), "mTans_dec")
    # write_to_pickle(classifier.state_dict(), "mTans_classifier")

    # First get the splitted person ids
    train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")

    train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # print(train_recent_visits.show())

    train_person_info = train_person_ids.join(person_information, on="person_id")
    valid_person_info = valid_person_ids.join(person_information, on="person_id")
    # train_person_ids = train_valid_split.loc[train_valid_split["split"] == "train"]
    # valid_person_ids = train_valid_split.loc[train_valid_split["split"] == "valid"]

    # Use it to split the data into training x/y and validation x/y
    # train_recent_visits = train_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")

    # Basic person information
    # train_person_info = train_person_ids.merge(person_information, on="person_id")
    # valid_person_info = valid_person_ids.merge(person_information, on="person_id")
    

    print("start pre-processing!!!")
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits(train_person_ids.toPandas(), train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both")
    # torch.save(visit_tensor_ls, "train_visit_tensor_ls")
    # torch.save(mask_ls, "train_mask_ls")
    # torch.save(time_step_ls, "train_mask_ls")
    # torch.save(label_tensor_ls, "train_label_tensor_ls")
    # torch.save(person_info_ls, "train_person_info_ls")
    
    # visit_tensor_ls, mask_ls = remove_empty_columns(visit_tensor_ls, mask_ls)

    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits(valid_person_ids.toPandas(), valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both")
    print("finish pre-processing!!!")
    visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)

    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)
    # train_dataset = LongCOVIDVisitsDataset2(train_person_ids, train_person_info, train_recent_visits, train_labels)
    # valid_dataset = LongCOVIDVisitsDataset2(valid_person_ids, valid_person_info, valid_recent_visits, valid_labels)

    # Construct dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)

    epochs=10
    print(train_dataset.__getitem__(1)[0])
    dim = train_dataset.__getitem__(1)[0].shape[-1]
    static_input_dim = train_dataset.__getitem__(1)[3].shape[-1]
    print("data shape::", train_dataset.__getitem__(1)[0].shape)
    print("mask shape::", train_dataset.__getitem__(1)[1].shape)
    print("dim::", dim)
    print(data_min)
    latent_dim=20
    rec_hidden=32
    learn_emb=True
    enc_num_heads=1
    num_ref_points=128
    gen_hidden=30
    dec_num_heads=1
    classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    lr = 0.0005

    

    rec = rec.to(device)
    dec = dec.to(device)
    classifier = classifier.to(device)

    best_valid_true, best_valid_pred_labels, best_train_true, best_train_pred_labels, rec_state_dict, dec_state_dict, classifier_state_dict = train_mTans(lr, True, 0.01, 100, 1, dim, latent_dim, rec, dec, classifier, epochs, train_loader, valid_loader, is_kl=True)
    device = torch.device('cpu')
    write_to_pickle(rec_state_dict, "mTans_rec")
    write_to_pickle(dec_state_dict, "mTans_dec")
    write_to_pickle(classifier_state_dict, "mTans_classifier")
    # read_from_pickle()
    print("save models successfully")
    # incorrect_labeled_train_ids = torch.from_numpy(np.nonzero(best_train_true != best_train_pred_labels).reshape(-1))
    # incorrect_sample_weight=3
    # train_set_weight = torch.ones(len(train_dataset))
    # train_set_weight[incorrect_labeled_train_ids] = incorrect_sample_weight
    # train_set_weight = train_set_weight/incorrect_sample_weight
    # sampler = WeightedRandomSampler(train_set_weight, len(train_set_weight))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, sampler=sampler, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    # print("start reweighted training::")

    # classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    # rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    # dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    # rec = rec.to(device)
    # dec = dec.to(device)
    # classifier = classifier.to(device)
    # train_mTans(lr, True, 0.01, 100, 1, dim, latent_dim, rec, dec, classifier, epochs, train_loader, valid_loader, is_kl=True)

    # device = torch.device('cpu')

    # write_to_pickle(rec.to(device), "mTans_rec")
    # write_to_pickle(dec.to(device), "mTans_dec")
    # write_to_pickle(classifier.to(device), "mTans_classifier")
    # # read_from_pickle()
    # print("save models successfully")

    
    # # Construct the two datasets
    # train_dataset = LongCOVIDVisitsDataset(train_person_ids, train_person_info, train_recent_visits, train_labels)
    # valid_dataset = LongCOVIDVisitsDataset(valid_person_ids, valid_person_info, valid_recent_visits, valid_labels)

    # # Construct dataloaders
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=LongCOVIDVisitsDataset.collate_fn)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True, collate_fn=LongCOVIDVisitsDataset.collate_fn)

    # # Construct model
    # model = LongCOVIDVisitsLSTMModel()

    # # # Training loop
    # trainer = Trainer(train_loader, valid_loader, model)
    # result_model = trainer.train()

    # return train_person_ids

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c1baf738-71b8-4ffd-8acd-47f6d38af10f"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    person_information=Input(rid="ri.foundry.main.dataset.2f6ebf73-3a2d-43dc-ace9-da56da4b1743"),
    recent_visits_w_nlp_notes_2=Input(rid="ri.foundry.main.dataset.fc6afa83-8c7a-4b04-a92a-ff1162651b0b"),
    train_sequential_model_3=Input(rid="ri.foundry.main.dataset.4fa4a34a-a9e7-489f-a499-023c2d4c44ac"),
    train_valid_split=Input(rid="ri.foundry.main.dataset.9d1a79f6-7627-4ee4-abc0-d6d6179c2f26")
)
import pandas as pd

def train_sequential_model_3_rebalance(train_sequential_model_3, train_valid_split, Long_COVID_Silver_Standard, person_information, recent_visits_w_nlp_notes_2):

    print("start")
    # dim=10
    # latent_dim=20
    # rec_hidden=32
    # learn_emb=True
    # enc_num_heads=1
    # num_ref_points=128
    # gen_hidden=30
    # dec_num_heads=1
    # static_input_dim = 10
    # classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available() else 'cpu')
    # print("device::", device)
    # rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    # dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    # lr = 0.0005

    # write_to_pickle(rec.state_dict(), "mTans_rec")
    # write_to_pickle(dec.state_dict(), "mTans_dec")
    # write_to_pickle(classifier.state_dict(), "mTans_classifier")

    # First get the splitted person ids
    train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")

    train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # print(train_recent_visits.show())

    train_person_info = train_person_ids.join(person_information, on="person_id")
    valid_person_info = valid_person_ids.join(person_information, on="person_id")
    # train_person_ids = train_valid_split.loc[train_valid_split["split"] == "train"]
    # valid_person_ids = train_valid_split.loc[train_valid_split["split"] == "valid"]

    # Use it to split the data into training x/y and validation x/y
    # train_recent_visits = train_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")

    # Basic person information
    # train_person_info = train_person_ids.merge(person_information, on="person_id")
    # valid_person_info = valid_person_ids.merge(person_information, on="person_id")
    

    print("start pre-processing!!!")
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits(train_person_ids.toPandas(), train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both")
    # torch.save(visit_tensor_ls, "train_visit_tensor_ls")
    # torch.save(mask_ls, "train_mask_ls")
    # torch.save(time_step_ls, "train_mask_ls")
    # torch.save(label_tensor_ls, "train_label_tensor_ls")
    # torch.save(person_info_ls, "train_person_info_ls")
    
    # visit_tensor_ls, mask_ls = remove_empty_columns(visit_tensor_ls, mask_ls)

    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits(valid_person_ids.toPandas(), valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both")
    print("finish pre-processing!!!")
    visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)

    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    # write_to_pickle(train_dataset, "train_dataset")
    # write_to_pickle(valid_dataset, "valid_dataset")
    # train_dataset = LongCOVIDVisitsDataset2(train_person_ids, train_person_info, train_recent_visits, train_labels)
    # valid_dataset = LongCOVIDVisitsDataset2(valid_person_ids, valid_person_info, valid_recent_visits, valid_labels)

    # Construct dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)

    epochs=10
    print(train_dataset.__getitem__(1)[0])
    dim = train_dataset.__getitem__(1)[0].shape[-1]
    static_input_dim = train_dataset.__getitem__(1)[3].shape[-1]
    print("data shape::", train_dataset.__getitem__(1)[0].shape)
    print("mask shape::", train_dataset.__getitem__(1)[1].shape)
    print("dim::", dim)
    print(data_min)
    latent_dim=20
    rec_hidden=32
    learn_emb=True
    enc_num_heads=1
    num_ref_points=128
    gen_hidden=30
    dec_num_heads=1
    classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    lr = 0.0005

    

    rec = rec.to(device)
    dec = dec.to(device)
    classifier = classifier.to(device)

    rec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_rec.pickle'))
    dec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_dec.pickle'))
    classifier.load_state_dict(read_model_from_pickle(train_sequential_model_3, "mTans_classifier.pickle"))
    print("load model successfully")
    print("start evaluating model::")

    # val_loss, val_acc, val_auc, val_recall, val_precision, _,_ = evaluate_classifier(rec, train_loader, latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)

    # print("validation performance at epoch::", 0)
    # print("validation loss::", val_loss)
    # print("validation accuracy::", val_acc)
    # print("validation auc score::", val_auc)
    # print("validation recall::", val_recall)
    # print("validation precision score::", val_precision)

    # _, _, _, _, _, best_train_true, best_train_pred_labels = evaluate_classifier(rec, train_loader, latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)

    

    # incorrect_labeled_train_ids = torch.from_numpy(np.nonzero(best_train_true.reshape(-1) != best_train_pred_labels.reshape(-1))[0])
    # print("incorrect_labeled_train_ids::", incorrect_labeled_train_ids)
    # print("incorrect_labeled_train_ids count::", len(incorrect_labeled_train_ids))
    incorrect_sample_weight=2

    pos_ids = torch.tensor([idx for idx in range(len(train_dataset.label_tensor_ls)) if train_dataset.label_tensor_ls[idx] == 1])
    neg_ids = torch.tensor([idx for idx in range(len(train_dataset.label_tensor_ls)) if train_dataset.label_tensor_ls[idx] == 0])
    print("positive training count::", len(pos_ids))
    print("negative training count::", len(neg_ids))

    train_set_weight = torch.ones(len(train_dataset))
    train_set_weight[pos_ids] = incorrect_sample_weight
    train_set_weight = train_set_weight/incorrect_sample_weight
    sampler = torch.utils.data.sampler.WeightedRandomSampler(train_set_weight, len(train_set_weight))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, sampler=sampler, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    print("start reweighted training::")

    # classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available() else 'cpu')
    # print("device::", device)
    # rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    # dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)

    # rec = rec.to(device)
    # dec = dec.to(device)
    # classifier = classifier.to(device)
    best_valid_true, best_valid_pred_labels, best_train_true, best_train_pred_labels, rec_state_dict, dec_state_dict, classifier_state_dict = train_mTans(lr, True, 0.01, 100, 1, dim, latent_dim, rec, dec, classifier, epochs, train_loader, valid_loader, is_kl=True)
    # device = torch.device('cpu')
    # write_to_pickle(rec_state_dict, "mTans_rec_reweight")
    # write_to_pickle(dec_state_dict, "mTans_dec_reweight")
    # write_to_pickle(classifier_state_dict, "mTans_classifier_reweight")
    # read_from_pickle()
    print("save models successfully")
    # Make predictions on test data
    # df = testing_data[['median_income', 'housing_median_age', 'total_rooms']]
    # predictions = my_model.predict(df)
    # df['predicted_house_values'] = pd.DataFrame(predictions)    

@transform_pandas(
    Output(rid="ri.vector.main.execute.1842252f-8cfa-4dac-ae3e-62e629d09151"),
    Produce_obs_dataset_2=Input(rid="ri.foundry.main.dataset.7f4133c1-cccc-4f64-b292-3055c0ade3a0")
)
import pandas as pd

def train_sequential_model_new(Produce_obs_dataset_2):

    print("start")
    # dim=10
    # latent_dim=20
    # rec_hidden=32
    # learn_emb=True
    # enc_num_heads=1
    # num_ref_points=128
    # gen_hidden=30
    # dec_num_heads=1
    # static_input_dim = 10
    # classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available() else 'cpu')
    # print("device::", device)
    # rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    # dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    # lr = 0.0005

    # write_to_pickle(rec.state_dict(), "mTans_rec")
    # write_to_pickle(dec.state_dict(), "mTans_dec")
    # write_to_pickle(classifier.state_dict(), "mTans_classifier")

    # First get the splitted person ids
    # train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    # valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")

    # train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # # print(train_recent_visits.show())

    # train_person_info = train_person_ids.join(person_information, on="person_id")
    # valid_person_info = valid_person_ids.join(person_information, on="person_id")
    # # train_person_ids = train_valid_split.loc[train_valid_split["split"] == "train"]
    # # valid_person_ids = train_valid_split.loc[train_valid_split["split"] == "valid"]

    # # Use it to split the data into training x/y and validation x/y
    # # train_recent_visits = train_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # # valid_recent_visits = valid_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # # train_labels = train_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")
    # # valid_labels = valid_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")

    # # Basic person information
    # # train_person_info = train_person_ids.merge(person_information, on="person_id")
    # # valid_person_info = valid_person_ids.merge(person_information, on="person_id")
    

    # print("start pre-processing!!!")
    # visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits(train_person_ids.toPandas(), train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both")
    # # torch.save(visit_tensor_ls, "train_visit_tensor_ls")
    # # torch.save(mask_ls, "train_mask_ls")
    # # torch.save(time_step_ls, "train_mask_ls")
    # # torch.save(label_tensor_ls, "train_label_tensor_ls")
    # # torch.save(person_info_ls, "train_person_info_ls")
    
    # # visit_tensor_ls, mask_ls = remove_empty_columns(visit_tensor_ls, mask_ls)

    # valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits(valid_person_ids.toPandas(), valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both")
    # print("finish pre-processing!!!")
    # visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    # valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    # data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max = read_from_pickle(Produce_obs_dataset_2, "train_data.pickle")
    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, _,_ = read_from_pickle(Produce_obs_dataset_2, "valid_data.pickle")
    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    # write_to_pickle(train_dataset, "train_dataset")
    # write_to_pickle(valid_dataset, "valid_dataset")
    # train_dataset = LongCOVIDVisitsDataset2(train_person_ids, train_person_info, train_recent_visits, train_labels)
    # valid_dataset = LongCOVIDVisitsDataset2(valid_person_ids, valid_person_info, valid_recent_visits, valid_labels)

    # Construct dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)

    epochs=10
    print(train_dataset.__getitem__(1)[0])
    dim = train_dataset.__getitem__(1)[0].shape[-1]
    # static_input_dim = train_dataset.__getitem__(1)[3].shape[-1]
    print("data shape::", train_dataset.__getitem__(1)[0].shape)
    print("mask shape::", train_dataset.__getitem__(1)[1].shape)
    print("dim::", dim)
    print(data_min)
    latent_dim=20
    rec_hidden=32
    learn_emb=True
    enc_num_heads=1
    num_ref_points=128
    gen_hidden=30
    dec_num_heads=1
    classifier = create_classifier(latent_dim, 20, has_static=False)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    lr = 0.0005

    

    rec = rec.to(device)
    dec = dec.to(device)
    classifier = classifier.to(device)

    # rec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_rec.pickle'))
    # dec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_dec.pickle'))
    # classifier.load_state_dict(read_model_from_pickle(train_sequential_model_3, "mTans_classifier.pickle"))
    # print("load model successfully")
    # print("start evaluating model::")

    # val_loss, val_acc, val_auc, val_recall, val_precision, _,_ = evaluate_classifier(rec, train_loader, latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)

    # print("validation performance at epoch::", 0)
    # print("validation loss::", val_loss)
    # print("validation accuracy::", val_acc)
    # print("validation auc score::", val_auc)
    # print("validation recall::", val_recall)
    # print("validation precision score::", val_precision)

    # _, _, _, _, _, best_train_true, best_train_pred_labels = evaluate_classifier(rec, train_loader, latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)

    

    # incorrect_labeled_train_ids = torch.from_numpy(np.nonzero(best_train_true.reshape(-1) != best_train_pred_labels.reshape(-1))[0])
    # print("incorrect_labeled_train_ids::", incorrect_labeled_train_ids)
    # print("incorrect_labeled_train_ids count::", len(incorrect_labeled_train_ids))
    # incorrect_sample_weight=2

    # pos_ids = torch.tensor([idx for idx in range(len(train_dataset.label_tensor_ls)) if train_dataset.label_tensor_ls[idx] == 1])
    # neg_ids = torch.tensor([idx for idx in range(len(train_dataset.label_tensor_ls)) if train_dataset.label_tensor_ls[idx] == 0])
    # print("positive training count::", len(pos_ids))
    # print("negative training count::", len(neg_ids))

    # train_set_weight = torch.ones(len(train_dataset))
    # train_set_weight[pos_ids] = incorrect_sample_weight
    # train_set_weight = train_set_weight/incorrect_sample_weight
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(train_set_weight, len(train_set_weight))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, sampler=sampler, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    # print("start training::")

    # classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available() else 'cpu')
    # print("device::", device)
    # rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    # dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)

    # rec = rec.to(device)
    # dec = dec.to(device)
    # classifier = classifier.to(device)
    best_valid_true, best_valid_pred_labels, best_train_true, best_train_pred_labels, rec_state_dict, dec_state_dict, classifier_state_dict = train_mTans(lr, True, 0.01, 100, 1, dim, latent_dim, rec, dec, classifier, epochs, train_loader, valid_loader, is_kl=True)
    device = torch.device('cpu')
    write_to_pickle(rec_state_dict, "mTans_rec")
    write_to_pickle(dec_state_dict, "mTans_dec")
    write_to_pickle(classifier_state_dict, "mTans_classifier")
    # read_from_pickle()
    print("save models successfully")
    # Make predictions on test data
    # df = testing_data[['median_income', 'housing_median_age', 'total_rooms']]
    # predictions = my_model.predict(df)
    # df['predicted_house_values'] = pd.DataFrame(predictions)    

@transform_pandas(
    Output(rid="ri.vector.main.execute.44340edf-afab-4e85-b224-9d5cab2fec8d"),
    Produce_obs_dataset_with_static_feature=Input(rid="ri.foundry.main.dataset.26c31342-fc7f-43b8-a3e5-5b115e528bd4")
)
import pandas as pd

def train_sequential_model_new_with_static_feature(Produce_obs_dataset_with_static_feature):

    print("start")
    # dim=10
    # latent_dim=20
    # rec_hidden=32
    # learn_emb=True
    # enc_num_heads=1
    # num_ref_points=128
    # gen_hidden=30
    # dec_num_heads=1
    # static_input_dim = 10
    # classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available() else 'cpu')
    # print("device::", device)
    # rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    # dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    # lr = 0.0005

    # write_to_pickle(rec.state_dict(), "mTans_rec")
    # write_to_pickle(dec.state_dict(), "mTans_dec")
    # write_to_pickle(classifier.state_dict(), "mTans_classifier")


    # First get the splitted person ids
    # train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    # valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")

    # train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # # print(train_recent_visits.show())

    # train_person_info = train_person_ids.join(person_information, on="person_id")
    # valid_person_info = valid_person_ids.join(person_information, on="person_id")
    # # train_person_ids = train_valid_split.loc[train_valid_split["split"] == "train"]
    # # valid_person_ids = train_valid_split.loc[train_valid_split["split"] == "valid"]

    # # Use it to split the data into training x/y and validation x/y
    # # train_recent_visits = train_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # # valid_recent_visits = valid_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # # train_labels = train_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")
    # # valid_labels = valid_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")

    # # Basic person information
    # # train_person_info = train_person_ids.merge(person_information, on="person_id")
    # # valid_person_info = valid_person_ids.merge(person_information, on="person_id")
    

    # print("start pre-processing!!!")
    # visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits(train_person_ids.toPandas(), train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both")
    # # torch.save(visit_tensor_ls, "train_visit_tensor_ls")
    # # torch.save(mask_ls, "train_mask_ls")
    # # torch.save(time_step_ls, "train_mask_ls")
    # # torch.save(label_tensor_ls, "train_label_tensor_ls")
    # # torch.save(person_info_ls, "train_person_info_ls")
    
    # # visit_tensor_ls, mask_ls = remove_empty_columns(visit_tensor_ls, mask_ls)

    # valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits(valid_person_ids.toPandas(), valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both")
    # print("finish pre-processing!!!")
    # visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    # valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    # data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max = read_from_pickle(Produce_obs_dataset_with_static_feature, "train_data.pickle")
    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, _,_ = read_from_pickle(Produce_obs_dataset_with_static_feature, "valid_data.pickle")
    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    # write_to_pickle(train_dataset, "train_dataset")
    # write_to_pickle(valid_dataset, "valid_dataset")
    # train_dataset = LongCOVIDVisitsDataset2(train_person_ids, train_person_info, train_recent_visits, train_labels)
    # valid_dataset = LongCOVIDVisitsDataset2(valid_person_ids, valid_person_info, valid_recent_visits, valid_labels)

    # Construct dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)

    epochs=10
    print(train_dataset.__getitem__(1)[0])
    dim = train_dataset.__getitem__(1)[0].shape[-1]
    static_input_dim = train_dataset.__getitem__(1)[3].shape[-1]
    print("data shape::", train_dataset.__getitem__(1)[0].shape)
    print("mask shape::", train_dataset.__getitem__(1)[1].shape)
    print("dim::", dim)
    print(data_min)
    latent_dim=20
    rec_hidden=32
    learn_emb=True
    enc_num_heads=1
    num_ref_points=128
    gen_hidden=30
    dec_num_heads=1
    classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    lr = 0.0005

    

    rec = rec.to(device)
    dec = dec.to(device)
    classifier = classifier.to(device)


    # rec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_rec.pickle'))
    # dec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_dec.pickle'))
    # classifier.load_state_dict(read_model_from_pickle(train_sequential_model_3, "mTans_classifier.pickle"))
    # print("load model successfully")
    # print("start evaluating model::")

    # val_loss, val_acc, val_auc, val_recall, val_precision, _,_ = evaluate_classifier(rec, train_loader, latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)

    # print("validation performance at epoch::", 0)
    # print("validation loss::", val_loss)
    # print("validation accuracy::", val_acc)
    # print("validation auc score::", val_auc)
    # print("validation recall::", val_recall)
    # print("validation precision score::", val_precision)



    # _, _, _, _, _, best_train_true, best_train_pred_labels = evaluate_classifier(rec, train_loader, latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)

    

    # incorrect_labeled_train_ids = torch.from_numpy(np.nonzero(best_train_true.reshape(-1) != best_train_pred_labels.reshape(-1))[0])
    # print("incorrect_labeled_train_ids::", incorrect_labeled_train_ids)
    # print("incorrect_labeled_train_ids count::", len(incorrect_labeled_train_ids))
    # incorrect_sample_weight=2

    # pos_ids = torch.tensor([idx for idx in range(len(train_dataset.label_tensor_ls)) if train_dataset.label_tensor_ls[idx] == 1])
    # neg_ids = torch.tensor([idx for idx in range(len(train_dataset.label_tensor_ls)) if train_dataset.label_tensor_ls[idx] == 0])
    # print("positive training count::", len(pos_ids))
    # print("negative training count::", len(neg_ids))


    # train_set_weight = torch.ones(len(train_dataset))
    # train_set_weight[pos_ids] = incorrect_sample_weight
    # train_set_weight = train_set_weight/incorrect_sample_weight
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(train_set_weight, len(train_set_weight))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, sampler=sampler, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    # print("start training::")


    # classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available() else 'cpu')
    # print("device::", device)
    # rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    # dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)

    # rec = rec.to(device)
    # dec = dec.to(device)
    # classifier = classifier.to(device)
    best_valid_true, best_valid_pred_labels, best_train_true, best_train_pred_labels, rec_state_dict, dec_state_dict, classifier_state_dict = train_mTans(lr, True, 0.01, 100, 1, dim, latent_dim, rec, dec, classifier, epochs, train_loader, valid_loader, is_kl=True)
    device = torch.device('cpu')
    write_to_pickle(rec_state_dict, "mTans_rec")
    write_to_pickle(dec_state_dict, "mTans_dec")
    write_to_pickle(classifier_state_dict, "mTans_classifier")
    # read_from_pickle()
    print("save models successfully")
    # Make predictions on test data
    # df = testing_data[['median_income', 'housing_median_age', 'total_rooms']]
    # predictions = my_model.predict(df)
    # df['predicted_house_values'] = pd.DataFrame(predictions)    

@transform_pandas(
    Output(rid="ri.vector.main.execute.62832a7b-121e-45a3-a04f-657c6c8fa493"),
    Produce_obs_dataset_with_static_feature_2=Input(rid="ri.foundry.main.dataset.051f4281-a989-4f89-85d0-d641e2afe2b0")
)
import pandas as pd

def train_sequential_model_new_with_static_feature_2(Produce_obs_dataset_with_static_feature_2):

    print("start")
    # dim=10
    # latent_dim=20
    # rec_hidden=32
    # learn_emb=True
    # enc_num_heads=1
    # num_ref_points=128
    # gen_hidden=30
    # dec_num_heads=1
    # static_input_dim = 10
    # classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available() else 'cpu')
    # print("device::", device)
    # rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    # dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    # lr = 0.0005

    # write_to_pickle(rec.state_dict(), "mTans_rec")
    # write_to_pickle(dec.state_dict(), "mTans_dec")
    # write_to_pickle(classifier.state_dict(), "mTans_classifier")

    # First get the splitted person ids
    # train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    # valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")

    # train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # # print(train_recent_visits.show())

    # train_person_info = train_person_ids.join(person_information, on="person_id")
    # valid_person_info = valid_person_ids.join(person_information, on="person_id")
    # # train_person_ids = train_valid_split.loc[train_valid_split["split"] == "train"]
    # # valid_person_ids = train_valid_split.loc[train_valid_split["split"] == "valid"]

    # # Use it to split the data into training x/y and validation x/y
    # # train_recent_visits = train_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # # valid_recent_visits = valid_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # # train_labels = train_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")
    # # valid_labels = valid_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")

    # # Basic person information
    # # train_person_info = train_person_ids.merge(person_information, on="person_id")
    # # valid_person_info = valid_person_ids.merge(person_information, on="person_id")
    

    # print("start pre-processing!!!")
    # visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits(train_person_ids.toPandas(), train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both")
    # # torch.save(visit_tensor_ls, "train_visit_tensor_ls")
    # # torch.save(mask_ls, "train_mask_ls")
    # # torch.save(time_step_ls, "train_mask_ls")
    # # torch.save(label_tensor_ls, "train_label_tensor_ls")
    # # torch.save(person_info_ls, "train_person_info_ls")
    
    # # visit_tensor_ls, mask_ls = remove_empty_columns(visit_tensor_ls, mask_ls)

    # valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits(valid_person_ids.toPandas(), valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both")
    # print("finish pre-processing!!!")
    # visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    # valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    # data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max = read_from_pickle(Produce_obs_dataset_with_static_feature_2, "train_data.pickle")
    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, _,_ = read_from_pickle(Produce_obs_dataset_with_static_feature_2, "valid_data.pickle")
    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    # write_to_pickle(train_dataset, "train_dataset")
    # write_to_pickle(valid_dataset, "valid_dataset")
    # train_dataset = LongCOVIDVisitsDataset2(train_person_ids, train_person_info, train_recent_visits, train_labels)
    # valid_dataset = LongCOVIDVisitsDataset2(valid_person_ids, valid_person_info, valid_recent_visits, valid_labels)

    # Construct dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)

    epochs=10
    print(train_dataset.__getitem__(1)[0])
    dim = train_dataset.__getitem__(1)[0].shape[-1]
    static_input_dim = train_dataset.__getitem__(1)[3].shape[-1]
    print("data shape::", train_dataset.__getitem__(1)[0].shape)
    print("mask shape::", train_dataset.__getitem__(1)[1].shape)
    print("dim::", dim)
    print(data_min)
    latent_dim=20
    rec_hidden=32
    learn_emb=True
    enc_num_heads=1
    num_ref_points=128
    gen_hidden=30
    dec_num_heads=1
    classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    lr = 0.0005

    

    rec = rec.to(device)
    dec = dec.to(device)
    classifier = classifier.to(device)

    # rec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_rec.pickle'))
    # dec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_dec.pickle'))
    # classifier.load_state_dict(read_model_from_pickle(train_sequential_model_3, "mTans_classifier.pickle"))
    # print("load model successfully")
    # print("start evaluating model::")

    # val_loss, val_acc, val_auc, val_recall, val_precision, _,_ = evaluate_classifier(rec, train_loader, latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)

    # print("validation performance at epoch::", 0)
    # print("validation loss::", val_loss)
    # print("validation accuracy::", val_acc)
    # print("validation auc score::", val_auc)
    # print("validation recall::", val_recall)
    # print("validation precision score::", val_precision)

    # _, _, _, _, _, best_train_true, best_train_pred_labels = evaluate_classifier(rec, train_loader, latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)

    

    # incorrect_labeled_train_ids = torch.from_numpy(np.nonzero(best_train_true.reshape(-1) != best_train_pred_labels.reshape(-1))[0])
    # print("incorrect_labeled_train_ids::", incorrect_labeled_train_ids)
    # print("incorrect_labeled_train_ids count::", len(incorrect_labeled_train_ids))
    # incorrect_sample_weight=2

    # pos_ids = torch.tensor([idx for idx in range(len(train_dataset.label_tensor_ls)) if train_dataset.label_tensor_ls[idx] == 1])
    # neg_ids = torch.tensor([idx for idx in range(len(train_dataset.label_tensor_ls)) if train_dataset.label_tensor_ls[idx] == 0])
    # print("positive training count::", len(pos_ids))
    # print("negative training count::", len(neg_ids))

    # train_set_weight = torch.ones(len(train_dataset))
    # train_set_weight[pos_ids] = incorrect_sample_weight
    # train_set_weight = train_set_weight/incorrect_sample_weight
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(train_set_weight, len(train_set_weight))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, sampler=sampler, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    # print("start training::")

    # classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available() else 'cpu')
    # print("device::", device)
    # rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    # dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)

    # rec = rec.to(device)
    # dec = dec.to(device)
    # classifier = classifier.to(device)
    best_valid_true, best_valid_pred_labels, best_train_true, best_train_pred_labels, rec_state_dict, dec_state_dict, classifier_state_dict = train_mTans(lr, True, 0.01, 100, 1, dim, latent_dim, rec, dec, classifier, epochs, train_loader, valid_loader, is_kl=True)
    device = torch.device('cpu')
    write_to_pickle(rec_state_dict, "mTans_rec")
    write_to_pickle(dec_state_dict, "mTans_dec")
    write_to_pickle(classifier_state_dict, "mTans_classifier")
    # read_from_pickle()
    print("save models successfully")
    # Make predictions on test data
    # df = testing_data[['median_income', 'housing_median_age', 'total_rooms']]
    # predictions = my_model.predict(df)
    # df['predicted_house_values'] = pd.DataFrame(predictions)    

@transform_pandas(
    Output(rid="ri.vector.main.execute.ee71a7b0-2ec3-40e0-b8e2-2106f6feac1f"),
    produce_dataset=Input(rid="ri.foundry.main.dataset.ae1c108c-1813-47ba-831c-e5a37c599c49"),
    train_sequential_model_3=Input(rid="ri.foundry.main.dataset.4fa4a34a-a9e7-489f-a499-023c2d4c44ac")
)
import pandas as pd

def train_sequential_model_rebalance(train_sequential_model_3, produce_dataset):

    print("start")
    # dim=10
    # latent_dim=20
    # rec_hidden=32
    # learn_emb=True
    # enc_num_heads=1
    # num_ref_points=128
    # gen_hidden=30
    # dec_num_heads=1
    # static_input_dim = 10
    # classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available() else 'cpu')
    # print("device::", device)
    # rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    # dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    # lr = 0.0005

    # write_to_pickle(rec.state_dict(), "mTans_rec")
    # write_to_pickle(dec.state_dict(), "mTans_dec")
    # write_to_pickle(classifier.state_dict(), "mTans_classifier")

    # First get the splitted person ids
    # train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
    # valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")

    # train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
    # train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
    # # print(train_recent_visits.show())

    # train_person_info = train_person_ids.join(person_information, on="person_id")
    # valid_person_info = valid_person_ids.join(person_information, on="person_id")
    # # train_person_ids = train_valid_split.loc[train_valid_split["split"] == "train"]
    # # valid_person_ids = train_valid_split.loc[train_valid_split["split"] == "valid"]

    # # Use it to split the data into training x/y and validation x/y
    # # train_recent_visits = train_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # # valid_recent_visits = valid_person_ids.merge(recent_visits_w_nlp_notes_2, on="person_id")
    # # train_labels = train_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")
    # # valid_labels = valid_person_ids.merge(Long_COVID_Silver_Standard, on="person_id")

    # # Basic person information
    # # train_person_info = train_person_ids.merge(person_information, on="person_id")
    # # valid_person_info = valid_person_ids.merge(person_information, on="person_id")
    

    # print("start pre-processing!!!")
    # visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits(train_person_ids.toPandas(), train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both")
    # # torch.save(visit_tensor_ls, "train_visit_tensor_ls")
    # # torch.save(mask_ls, "train_mask_ls")
    # # torch.save(time_step_ls, "train_mask_ls")
    # # torch.save(label_tensor_ls, "train_label_tensor_ls")
    # # torch.save(person_info_ls, "train_person_info_ls")
    
    # # visit_tensor_ls, mask_ls = remove_empty_columns(visit_tensor_ls, mask_ls)

    # valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits(valid_person_ids.toPandas(), valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both")
    # print("finish pre-processing!!!")
    # visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
    # valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

    # data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)
    visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max = read_from_pickle(produce_dataset, "train_data.pickle")
    valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, _,_ = read_from_pickle(produce_dataset, "valid_data.pickle")
    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)

    # write_to_pickle(train_dataset, "train_dataset")
    # write_to_pickle(valid_dataset, "valid_dataset")
    # train_dataset = LongCOVIDVisitsDataset2(train_person_ids, train_person_info, train_recent_visits, train_labels)
    # valid_dataset = LongCOVIDVisitsDataset2(valid_person_ids, valid_person_info, valid_recent_visits, valid_labels)

    # Construct dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)

    epochs=10
    print(train_dataset.__getitem__(1)[0])
    dim = train_dataset.__getitem__(1)[0].shape[-1]
    static_input_dim = train_dataset.__getitem__(1)[3].shape[-1]
    print("data shape::", train_dataset.__getitem__(1)[0].shape)
    print("mask shape::", train_dataset.__getitem__(1)[1].shape)
    print("dim::", dim)
    print(data_min)
    latent_dim=20
    rec_hidden=32
    learn_emb=True
    enc_num_heads=1
    num_ref_points=128
    gen_hidden=30
    dec_num_heads=1
    classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    lr = 0.0005

    

    rec = rec.to(device)
    dec = dec.to(device)
    classifier = classifier.to(device)

    rec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_rec.pickle'))
    dec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_dec.pickle'))
    classifier.load_state_dict(read_model_from_pickle(train_sequential_model_3, "mTans_classifier.pickle"))
    print("load model successfully")
    print("start evaluating model::")

    # val_loss, val_acc, val_auc, val_recall, val_precision, _,_ = evaluate_classifier(rec, train_loader, latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)

    # print("validation performance at epoch::", 0)
    # print("validation loss::", val_loss)
    # print("validation accuracy::", val_acc)
    # print("validation auc score::", val_auc)
    # print("validation recall::", val_recall)
    # print("validation precision score::", val_precision)

    # _, _, _, _, _, best_train_true, best_train_pred_labels = evaluate_classifier(rec, train_loader, latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)

    

    # incorrect_labeled_train_ids = torch.from_numpy(np.nonzero(best_train_true.reshape(-1) != best_train_pred_labels.reshape(-1))[0])
    # print("incorrect_labeled_train_ids::", incorrect_labeled_train_ids)
    # print("incorrect_labeled_train_ids count::", len(incorrect_labeled_train_ids))
    incorrect_sample_weight=2

    pos_ids = torch.tensor([idx for idx in range(len(train_dataset.label_tensor_ls)) if train_dataset.label_tensor_ls[idx] == 1])
    neg_ids = torch.tensor([idx for idx in range(len(train_dataset.label_tensor_ls)) if train_dataset.label_tensor_ls[idx] == 0])
    print("positive training count::", len(pos_ids))
    print("negative training count::", len(neg_ids))

    train_set_weight = torch.ones(len(train_dataset))
    train_set_weight[pos_ids] = incorrect_sample_weight
    train_set_weight = train_set_weight/incorrect_sample_weight
    sampler = torch.utils.data.sampler.WeightedRandomSampler(train_set_weight, len(train_set_weight))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, sampler=sampler, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    print("start reweighted training::")

    # classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available() else 'cpu')
    # print("device::", device)
    # rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    # dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)

    # rec = rec.to(device)
    # dec = dec.to(device)
    # classifier = classifier.to(device)
    best_valid_true, best_valid_pred_labels, best_train_true, best_train_pred_labels, rec_state_dict, dec_state_dict, classifier_state_dict = train_mTans(lr, True, 0.01, 100, 1, dim, latent_dim, rec, dec, classifier, epochs, train_loader, valid_loader, is_kl=True)
    device = torch.device('cpu')
    write_to_pickle(rec_state_dict, "mTans_rec_reweight")
    write_to_pickle(dec_state_dict, "mTans_dec_reweight")
    write_to_pickle(classifier_state_dict, "mTans_classifier_reweight")
    # read_from_pickle()
    print("save models successfully")
    # Make predictions on test data
    # df = testing_data[['median_income', 'housing_median_age', 'total_rooms']]
    # predictions = my_model.predict(df)
    # df['predicted_house_values'] = pd.DataFrame(predictions)    

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

    for col in all_patients_summary_fact_table_de_id.columns:
        if col not in all_patients_summary_fact_table_de_id_testing.columns:
            print(col, " not in summary testing set.")
            all_patients_summary_fact_table_de_id_testing[col] = 0
    for col in all_patients_summary_fact_table_de_id_testing.columns:
        if col not in all_patients_summary_fact_table_de_id.columns:
            print(col, " not in summary training set.")
            all_patients_summary_fact_table_de_id_testing.drop(col, axis=1)

    Training_and_Holdout = all_patients_summary_fact_table_de_id[cols].fillna(0.0).sort_values('person_id')
    Training_and_Holdout = Training_and_Holdout.sort_index(axis=1)
    Testing = all_patients_summary_fact_table_de_id_testing[cols].fillna(0.0).sort_values('person_id')
    Testing = Testing.sort_index(axis=1)
    
    if LOAD_TEST == 0:
        X_train_no_ind, X_test_no_ind, y_train, y_test = train_test_split(Training_and_Holdout, Outcome, train_size=0.9, random_state=1)
    else:
        X_train_no_ind, y_train = Training_and_Holdout, Outcome
        X_test_no_ind, y_test = Testing, None
    X_train, X_test = X_train_no_ind.set_index("person_id"), X_test_no_ind.set_index("person_id")

    lrc_params = {'C': 1, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
    lrc2_params = {'C': 1, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 500, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
    rfc_params = {'bootstrap': False, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 400, 'n_jobs': -1, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
    gbc_params = {'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.075, 'loss': 'exponential', 'max_depth': 12, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 0.001, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 500, 'n_iter_no_change': None, 'random_state': None, 'subsample': 0.618, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}

    lrc = LogisticRegression(**lrc_params).fit(X_train, y_train)
    lrc2 = LogisticRegression(**lrc2_params).fit(X_train, y_train)
    rfc = RandomForestClassifier(**rfc_params).fit(X_train, y_train)
    gbc = GradientBoostingClassifier(**gbc_params).fit(X_train, y_train)

    lrc_sort_features = np.argsort(lrc.coef_.flatten())[-20:]
    lrc_sort_features_least = np.argsort(lrc.coef_.flatten())[:20]
    lrc2_sort_features = np.argsort(lrc2.coef_.flatten())[-20:]
    lrc2_sort_features_least = np.argsort(lrc2.coef_.flatten())[:20]
    rfc_sort_features = np.argsort(rfc.feature_importances_.flatten())[-20:]
    rfc_sort_features_least = np.argsort(rfc.feature_importances_.flatten())[:20]
    gbc_sort_features = np.argsort(gbc.feature_importances_.flatten())[-20:]
    gbc_sort_features_least = np.argsort(gbc.feature_importances_.flatten())[:20]

    fig, axs = plt.subplots(2,2, constrained_layout=True, figsize=(8,8))
    fig.suptitle('Twenty Most Important Features Per Model (Handpicked)')
    axs[0][0].bar(x = np.arange(20), height = rfc.feature_importances_.flatten()[rfc_sort_features], tick_label=[cols[1:][i] for i in rfc_sort_features])
    axs[0][0].set_xticklabels(labels=[cols[1:][i] for i in rfc_sort_features], fontdict={'rotation':"vertical",'size':'xx-small'})
    axs[0][0].set_ylabel("Purity-based Importance", size='small')
    axs[0][0].set_title("Random Forest Classifier")
    
    axs[0][1].bar(x = np.arange(20), height = lrc.coef_.flatten()[lrc_sort_features], tick_label=[cols[1:][i] for i in lrc_sort_features])
    axs[0][1].set_xticklabels(labels=[cols[1:][i] for i in lrc_sort_features], fontdict={'rotation':"vertical", 'size':'xx-small'})
    axs[0][1].set_ylabel("Coefficient-based Importance", size='small')
    axs[0][1].set_title("Logistic Regression Classifier")

    axs[1][0].bar(x = np.arange(20), height = gbc.feature_importances_.flatten()[gbc_sort_features], tick_label=[cols[1:][i] for i in gbc_sort_features])
    axs[1][0].set_xticklabels(labels=[cols[1:][i] for i in gbc_sort_features], fontdict={'rotation':"vertical",'size':'xx-small'})
    axs[1][0].set_ylabel("Purity-based Importance", size='small')
    axs[1][0].set_title("Gradient Boosted Classifier")

    axs[1][1].bar(x = np.arange(20), height = lrc2.coef_.flatten()[lrc2_sort_features], tick_label=[cols[1:][i] for i in lrc2_sort_features])
    axs[1][1].set_xticklabels(labels=[cols[1:][i] for i in lrc2_sort_features], fontdict={'rotation':"vertical", 'size':'xx-small'})
    axs[1][1].set_ylabel("Coefficient-based Importance", size='small')
    axs[1][1].set_title(" Balanced Logistic Regression Classifier")
    plt.show()

    print("lrc important features:", [cols[1:][int(i)] for i in lrc_sort_features])
    print("rfc important features:", [cols[1:][int(i)] for i in rfc_sort_features])
    print("lrc least important features:", [cols[1:][int(i)] for i in lrc_sort_features_least ])
    print("rfc least important features:", [cols[1:][int(i)] for i in rfc_sort_features_least ])
    print("combined least important features:", [cols[1:][int(i)] for i in rfc_sort_features_least if i in lrc_sort_features_least])
    print("column variance: \n", all_patients_summary_fact_table_de_id.var().to_string())
    nn_scaler = StandardScaler().fit(X_train)
    nnc = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 10), random_state=1).fit(nn_scaler.transform(X_train), y_train)

    #preds = clf.predict_proba(Testing)[:,1]
    print(X_test.shape)
    lr_test_preds = lrc.predict_proba(X_test)[:, 1]
    lr_train_preds = lrc.predict_proba(X_train)[:, 1]
    lr2_test_preds = lrc2.predict_proba(X_test)[:, 1]
    rf_test_preds = rfc.predict_proba(X_test)[:, 1]
    rf_train_preds = rfc.predict_proba(X_train)[:, 1]
    gb_test_preds = gbc.predict_proba(X_test)[:, 1]
    nnc_test_preds = nnc.predict_proba(nn_scaler.transform(X_test))[:, 1]

    print(X_train.shape)
    print(X_test.shape)
    predictions = pd.DataFrame.from_dict({
        'person_id': list(X_test_no_ind["person_id"]),
        'lr_outcome': lr_test_preds.tolist(),
        'lr2_outcome': lr2_test_preds.tolist(),
        'rf_outcome': rf_test_preds.tolist(),
        'gb_outcome': gb_test_preds.tolist(),
        'nn_outcome': nnc_test_preds.tolist(),
    }, orient='columns')
    
    if MERGE_LABEL  == 1:
        predictions = predictions.merge(Outcome_df, on="person_id", how="left")
    outcomes = ['lr2_outcome', 'rf_outcome', 'gb_outcome']
    predictions['ens_outcome'] = predictions.apply(lambda row: 1 if sum([row[c] for c in outcomes])/len(outcomes) >=0.5 else 0, axis=1)

    return predictions

@transform_pandas(
    Output(rid="ri.vector.main.execute.4bcf67c2-4fa9-43cc-b6dd-eb9ae2b8c1d2"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    all_patients_summary_fact_table_de_id=Input(rid="ri.foundry.main.dataset.324a6115-7c17-4d4d-94da-a2df11a87fa6"),
    all_patients_summary_fact_table_de_id_testing=Input(rid="ri.foundry.main.dataset.4b4d2bc0-b43f-4d63-abc6-ed115f0cd117")
)
def train_test_model_shap(all_patients_summary_fact_table_de_id, all_patients_summary_fact_table_de_id_testing, Long_COVID_Silver_Standard):
    
    static_cols = ['person_id','total_visits', 'age']

    cols = static_cols + [col for col in all_patients_summary_fact_table_de_id.columns if 'indicator' in col]
    
    ## get outcome column
    Long_COVID_Silver_Standard["outcome"] = Long_COVID_Silver_Standard.apply(lambda x: max([x["pasc_code_after_four_weeks"], x["pasc_code_prior_four_weeks"]]), axis=1)
    Outcome_df = all_patients_summary_fact_table_de_id[["person_id"]].merge(Long_COVID_Silver_Standard, on="person_id", how="left")
    Outcome_df = Outcome_df[["person_id", "outcome"]].sort_values('person_id')

    Outcome = list(Outcome_df["outcome"])

    for col in all_patients_summary_fact_table_de_id.columns:
        if col not in all_patients_summary_fact_table_de_id_testing.columns:
            print(col, " not in summary testing set.")
            all_patients_summary_fact_table_de_id_testing[col] = 0
    for col in all_patients_summary_fact_table_de_id_testing.columns:
        if col not in all_patients_summary_fact_table_de_id.columns:
            print(col, " not in summary training set.")
            all_patients_summary_fact_table_de_id_testing.drop(col, axis=1)

    Training_and_Holdout = all_patients_summary_fact_table_de_id[cols].fillna(0.0).sort_values('person_id')
    Training_and_Holdout = Training_and_Holdout.sort_index(axis=1)
    Testing = all_patients_summary_fact_table_de_id_testing[cols].fillna(0.0).sort_values('person_id')
    Testing = Testing.sort_index(axis=1)
    
    if LOAD_TEST == 0:
        X_train_no_ind, X_test_no_ind, y_train, y_test = train_test_split(Training_and_Holdout, Outcome, train_size=0.9, random_state=1)
    else:
        X_train_no_ind, y_train = Training_and_Holdout, Outcome
        X_test_no_ind, y_test = Testing, None
    X_train, X_test = X_train_no_ind.set_index("person_id"), X_test_no_ind.set_index("person_id")

    lrc_params = {'C': 1, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
    lrc2_params = {'C': 1, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 500, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
    rfc_params = {'bootstrap': False, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 400, 'n_jobs': -1, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
    gbc_params = {'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.075, 'loss': 'exponential', 'max_depth': 12, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 0.001, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 500, 'n_iter_no_change': None, 'random_state': None, 'subsample': 0.618, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}

    lrc = LogisticRegression(**lrc_params).fit(X_train, y_train)
    lrc2 = LogisticRegression(**lrc2_params).fit(X_train, y_train)
    rfc = RandomForestClassifier(**rfc_params).fit(X_train, y_train)
    gbc = GradientBoostingClassifier(**gbc_params).fit(X_train, y_train)

    ens_fn = lambda x: (lrc.predict_proba(x) + lrc2.predict_proba(x) + rfc.predict_proba(x) + gbc.predict_proba(x))[:, 1] / 4

    X_train_sample = shap.kmeans(X_train, 20)
    X_test_sample = X_test.head(100)

    preds = ens_fn(X_test.head(100)) > 0.5
    for i, pred in enumerate(preds):
        if pred and Outcome[i] == 1:
            print(i)

    print("Starting shap calculations")
    explainer = shap.KernelExplainer(ens_fn, X_train_sample)
    shap_values = explainer.shap_values(X_test_sample)

    # Issues with the Enclave don't allow us to save the data, so we perform the visualizations in the same node.
    plt.figure(figsize=(40,40))
    plt.subplot(1,2,1)
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[38], X_test_sample.iloc[38], feature_names=X_test_sample.columns, show=False)

    plt.subplot(1,2,2)
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], X_test_sample.iloc[0], feature_names=X_test_sample.columns, show=False)

    # plt.subplot(3,1,3)
    # shap.summary_plot(shap_values, X_test_sample.values, feature_names=X_test_sample.columns, show=False)

    # plt.subplot(4,1,4)
    # shap.plots._bar.bar_legacy(shap_values, feature_names=X_test_sample.columns, show=False)

    plt.tight_layout()
    plt.show()
    
    # write_to_pickle(explainer.expected_value, "ens_explainer_ev")
    # write_to_pickle(shap_values, "ens_shap_values")
    print("Finished ens shap")

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d7140ada-9148-4d0a-956c-adab7b0af033"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    all_patients_summary_fact_table_de_id=Input(rid="ri.foundry.main.dataset.324a6115-7c17-4d4d-94da-a2df11a87fa6"),
    all_patients_summary_fact_table_de_id_testing=Input(rid="ri.foundry.main.dataset.4b4d2bc0-b43f-4d63-abc6-ed115f0cd117"),
    person_information=Input(rid="ri.foundry.main.dataset.2f6ebf73-3a2d-43dc-ace9-da56da4b1743"),
    recent_visits_w_nlp_notes_2=Input(rid="ri.foundry.main.dataset.fc6afa83-8c7a-4b04-a92a-ff1162651b0b"),
    train_valid_split=Input(rid="ri.foundry.main.dataset.9d1a79f6-7627-4ee4-abc0-d6d6179c2f26")
)
def train_test_model_with_sequence_model(all_patients_summary_fact_table_de_id, all_patients_summary_fact_table_de_id_testing, Long_COVID_Silver_Standard, person_information, train_valid_split, recent_visits_w_nlp_notes_2):
    
    def prepare_sequence_data():
        train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
        valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")

        train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
        valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
        train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
        valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
        # print(train_recent_visits.show())

        train_person_info = train_person_ids.join(person_information, on="person_id")
        valid_person_info = valid_person_ids.join(person_information, on="person_id")
        return train_person_ids, valid_person_ids, train_recent_visits, valid_recent_visits, train_person_info, valid_person_info, train_labels, valid_labels

    def prepare_static_data(train_person_ids, valid_person_ids, Long_COVID_Silver_Standard):
        
        train_person_ids = train_person_ids.toPandas()
        valid_person_ids = valid_person_ids.toPandas()
        Long_COVID_Silver_Standard = Long_COVID_Silver_Standard.toPandas()

        static_cols = ['person_id','total_visits', 'age']

        cols = static_cols + [col for col in all_patients_summary_fact_table_de_id.columns if 'indicator' in col]
        
        ## get outcome column
        Long_COVID_Silver_Standard["outcome"] = Long_COVID_Silver_Standard.apply(lambda x: max([x["pasc_code_after_four_weeks"], x["pasc_code_prior_four_weeks"]]), axis=1)
        Outcome_df = all_patients_summary_fact_table_de_id[["person_id"]].merge(Long_COVID_Silver_Standard, on="person_id", how="left")
        # Outcome_df = all_patients_summary_fact_table_de_id.select(["person_id"]).join(Long_COVID_Silver_Standard, on="person_id", how="left")
        Outcome_df = Outcome_df[["person_id", "outcome"]].sort_values('person_id')

        Outcome = list(Outcome_df["outcome"])

        Training_and_Holdout = all_patients_summary_fact_table_de_id[cols].fillna(0.0).sort_values('person_id')
        #Testing = all_patients_summary_fact_table_de_id_testing[cols].fillna(0.0)
        X_train_no_ind = Training_and_Holdout.merge(train_person_ids, on="person_id")
        X_test_no_ind = Training_and_Holdout.merge(valid_person_ids, on="person_id")
        y_train = Outcome_df.merge(train_person_ids, on="person_id")
        y_test = Outcome_df.merge(valid_person_ids, on="person_id")
        X_train_no_ind = X_train_no_ind.sort_values("person_id")
        X_test_no_ind = X_test_no_ind.sort_values("person_id")
        y_train = y_train.sort_values("person_id")
        y_test = y_test.sort_values("person_id")
        print("first 10 train person ids::", list(X_train_no_ind["person_id"])[0:10])
        print("first 10 train person ids::", list(y_train["person_id"])[0:10])
        print("first 10 test person ids::", list(X_test_no_ind["person_id"])[0:10])
        print("first 10 test person ids::", list(y_test["person_id"])[0:10])
        y_train = y_train.set_index("person_id")
        y_test = y_test.set_index("person_id")
        X_train, X_test = X_train_no_ind.set_index("person_id"), X_test_no_ind.set_index("person_id")
        # X_train_no_ind, X_test_no_ind, y_train, y_test = train_test_split(Training_and_Holdout, Outcome, train_size=0.9, random_state=1)
        # X_train, X_test = X_train_no_ind.set_index("person_id"), X_test_no_ind.set_index("person_id")
        
        X_train = X_train[list(X_train.columns)[0:-1]]
        person_ids_test = list(X_test.index)
        X_test = X_test[list(X_test.columns)[0:-1]]
        print("X_train::", X_train)
        print("Y_train::", y_train)
        return np.array(X_train.values.tolist()), np.array(X_test.values.tolist()), np.array(list(y_train["outcome"])), np.array(list(y_test["outcome"])), person_ids_test

    def train_mTan_model_main(train_person_ids, valid_person_ids, train_recent_visits, valid_recent_visits, train_person_info, valid_person_info, train_labels, valid_labels):
        print("start pre-processing!!!")
        visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits(train_person_ids.toPandas(), train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both")
        # torch.save(visit_tensor_ls, "train_visit_tensor_ls")
        # torch.save(mask_ls, "train_mask_ls")
        # torch.save(time_step_ls, "train_mask_ls")
        # torch.save(label_tensor_ls, "train_label_tensor_ls")
        # torch.save(person_info_ls, "train_person_info_ls")
        
        # visit_tensor_ls, mask_ls = remove_empty_columns(visit_tensor_ls, mask_ls)

        valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits(valid_person_ids.toPandas(), valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both")
        print("finish pre-processing!!!")
        visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
        valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

        data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)

        train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

        valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)
        # train_dataset = LongCOVIDVisitsDataset2(train_person_ids, train_person_info, train_recent_visits, train_labels)
        # valid_dataset = LongCOVIDVisitsDataset2(valid_person_ids, valid_person_info, valid_recent_visits, valid_labels)

        # Construct dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)

        epochs=30
        print(train_dataset.__getitem__(1)[0])
        dim = train_dataset.__getitem__(1)[0].shape[-1]
        print("data shape::", train_dataset.__getitem__(1)[0].shape)
        print("mask shape::", train_dataset.__getitem__(1)[1].shape)
        print("dim::", dim)
        print(data_min)
        latent_dim=20
        rec_hidden=64
        learn_emb=True
        enc_num_heads=1
        num_ref_points=128
        gen_hidden=30
        dec_num_heads=1
        static_input_dim = train_dataset.__getitem__(1)[3].shape[-1]
        classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print("device::", device)
        rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=128, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
        dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=128, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
        lr = 0.0002

        rec = rec.to(device)
        dec = dec.to(device)
        classifier = classifier.to(device)

        return train_mTans(lr, True, 0.01, 100, 1, dim, latent_dim, rec, dec, classifier, epochs, train_loader, valid_loader, is_kl=True)

    train_person_ids, valid_person_ids, train_recent_visits, valid_recent_visits, train_person_info, valid_person_info, train_labels, valid_labels = prepare_sequence_data()
    X_train, X_test, y_train, y_test, person_ids_test = prepare_static_data(train_person_ids, valid_person_ids, Long_COVID_Silver_Standard)

    Y_true, Y_mTans, _, _ = train_mTan_model_main(train_person_ids, valid_person_ids, train_recent_visits, valid_recent_visits, train_person_info, valid_person_info, train_labels, valid_labels)

    def train_simple_models(X_train,y_train, X_test, y_test):
        print(X_train)
        lrc = LogisticRegression(penalty='l2', solver='liblinear', random_state=0, max_iter=500).fit(X_train, y_train)
        lrc2 = LogisticRegression(penalty='l2', class_weight='balanced', solver='liblinear', random_state=0, max_iter=500).fit(X_train, y_train)
        rfc = RandomForestClassifier().fit(X_train, y_train)
        gbc = GradientBoostingClassifier().fit(X_train, y_train)

        # lrc_sort_features = np.argsort(lrc.coef_.flatten())[-20:]
        # lrc_sort_features_least = np.argsort(lrc.coef_.flatten())[:20]
        # rfc_sort_features = np.argsort(rfc.feature_importances_.flatten())[-20:]
        # rfc_sort_features_least = np.argsort(rfc.feature_importances_.flatten())[:20]
        # plt.bar(np.arange(20), rfc.feature_importances_.flatten()[rfc_sort_features])
        # plt.xticks(np.arange(20), [cols[1:][i] for i in rfc_sort_features], rotation='vertical')
        # plt.tight_layout()
        # plt.show()

        # print("lrc important features:", [cols[1:][int(i)] for i in lrc_sort_features])
        # print("rfc important features:", [cols[1:][int(i)] for i in rfc_sort_features])
        # print("lrc least important features:", [cols[1:][int(i)] for i in lrc_sort_features_least ])
        # print("rfc least important features:", [cols[1:][int(i)] for i in rfc_sort_features_least ])
        # print("combined least important features:", [cols[1:][int(i)] for i in rfc_sort_features_least if i in lrc_sort_features_least])
        # print("column variance: \n", all_patients_summary_fact_table_de_id.var().to_string())
        nn_scaler = StandardScaler().fit(X_train)
        nnc = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 10), random_state=1).fit(nn_scaler.transform(X_train), y_train)

        #preds = clf.predict_proba(Testing)[:,1]

        lr_test_preds = lrc.predict_proba(X_test)[:, 1]
        lr_train_preds = lrc.predict_proba(X_train)[:, 1]
        lr2_test_preds = lrc2.predict_proba(X_test)[:, 1]
        rf_test_preds = rfc.predict_proba(X_test)[:, 1]
        rf_train_preds = rfc.predict_proba(X_train)[:, 1]
        gb_test_preds = gbc.predict_proba(X_test)[:, 1]
        nnc_test_preds = nnc.predict_proba(nn_scaler.transform(X_test))[:, 1]

        print("LR Training Classification Report:\n{}".format(classification_report(y_train, np.where(lr_train_preds > 0.5, 1, 0))))

        #test_df = 
        test_predictions = pd.DataFrame.from_dict({
            'person_id': person_ids_test,
            'lr_outcome': lr_test_preds.tolist(),
            'lr2_outcome': lr2_test_preds.tolist(),
            'rf_outcome': rf_test_preds.tolist(),
            'gb_outcome': gb_test_preds.tolist(),
            'nn_outcome': nnc_test_preds.tolist(),
        }, orient='columns')
        
        # test_predictions = test_predictions.merge(Outcome_df, on="person_id", how="left")
        # outcomes = ['lr_outcome', 'lr2_outcome', 'rf_outcome', 'gb_outcome', 'nn_outcome']
        # test_predictions['ens_outcome'] = test_predictions.apply(lambda row: 1 if sum([row[c] for c in outcomes])/len(outcomes) >=0.5 else 0, axis=1)

    # predictions = pd.DataFrame.from_dict({
    #     'person_id': list(all_patients_summary_fact_table_de_id_testing["person_id"]),
    #     'outcome_likelihood': preds.tolist()
    # }, orient='columns')

        return lr_test_preds.tolist(), lr2_test_preds.tolist(), rf_test_preds.tolist(),gb_test_preds.tolist(), nnc_test_preds.tolist()

    lr_test_preds, lr2_test_preds, rf_test_preds,gb_test_preds, nnc_test_preds = train_simple_models(X_train,y_train, X_test, y_test)

    Y_all = np.stack((gb_test_preds, nnc_test_preds, Y_mTans), axis=0)

    # Y_all = np.stack((lr_test_preds, lr2_test_preds, rf_test_preds, gb_test_preds, nnc_test_preds), axis=0)
    Y_all = (Y_all > 0.5).astype(int)
    Y_final_pred = scipy.stats.mode(Y_all, axis=0).mode[0]
    print("pred labels::", Y_final_pred.reshape(-1))
    print("gt labels::", y_test.reshape(-1))
    print("validation classification Report:\n{}".format(classification_report(y_test.reshape(-1).astype(int), Y_final_pred.reshape(-1))))

@transform_pandas(
    Output(rid="ri.vector.main.execute.ef9bccf1-8bcb-4ade-a935-23bfb3d41c87"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    all_patients_summary_fact_table_de_id=Input(rid="ri.foundry.main.dataset.324a6115-7c17-4d4d-94da-a2df11a87fa6"),
    all_patients_summary_fact_table_de_id_testing=Input(rid="ri.foundry.main.dataset.4b4d2bc0-b43f-4d63-abc6-ed115f0cd117"),
    person_information=Input(rid="ri.foundry.main.dataset.2f6ebf73-3a2d-43dc-ace9-da56da4b1743"),
    recent_visits_w_nlp_notes_2=Input(rid="ri.foundry.main.dataset.fc6afa83-8c7a-4b04-a92a-ff1162651b0b"),
    train_valid_split=Input(rid="ri.foundry.main.dataset.9d1a79f6-7627-4ee4-abc0-d6d6179c2f26")
)
def train_test_model_with_sequence_model_2(all_patients_summary_fact_table_de_id, all_patients_summary_fact_table_de_id_testing, Long_COVID_Silver_Standard, person_information, train_valid_split, recent_visits_w_nlp_notes_2):
    
    def prepare_sequence_data():
        train_person_ids = train_valid_split.where(train_valid_split["split"] == "train")
        valid_person_ids = train_valid_split.where(train_valid_split["split"] == "valid")

        train_recent_visits = train_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
        valid_recent_visits = valid_person_ids.join(recent_visits_w_nlp_notes_2, on="person_id")
        train_labels = train_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
        valid_labels = valid_person_ids.join(Long_COVID_Silver_Standard, on="person_id")
        # print(train_recent_visits.show())

        train_person_info = train_person_ids.join(person_information, on="person_id")
        valid_person_info = valid_person_ids.join(person_information, on="person_id")
        return train_person_ids, valid_person_ids, train_recent_visits, valid_recent_visits, train_person_info, valid_person_info, train_labels, valid_labels

    def prepare_static_data(train_person_ids, valid_person_ids, Long_COVID_Silver_Standard):
        
        train_person_ids = train_person_ids.toPandas()
        valid_person_ids = valid_person_ids.toPandas()
        Long_COVID_Silver_Standard = Long_COVID_Silver_Standard.toPandas()

        static_cols = ['person_id','total_visits', 'age']

        cols = static_cols + [col for col in all_patients_summary_fact_table_de_id.columns if 'indicator' in col]
        
        ## get outcome column
        Long_COVID_Silver_Standard["outcome"] = Long_COVID_Silver_Standard.apply(lambda x: max([x["pasc_code_after_four_weeks"], x["pasc_code_prior_four_weeks"]]), axis=1)
        Outcome_df = all_patients_summary_fact_table_de_id[["person_id"]].merge(Long_COVID_Silver_Standard, on="person_id", how="left")
        # Outcome_df = all_patients_summary_fact_table_de_id.select(["person_id"]).join(Long_COVID_Silver_Standard, on="person_id", how="left")
        Outcome_df = Outcome_df[["person_id", "outcome"]].sort_values('person_id')

        Outcome = list(Outcome_df["outcome"])

        Training_and_Holdout = all_patients_summary_fact_table_de_id[cols].fillna(0.0).sort_values('person_id')
        #Testing = all_patients_summary_fact_table_de_id_testing[cols].fillna(0.0)
        X_train_no_ind = Training_and_Holdout.merge(train_person_ids, on="person_id")
        X_test_no_ind = Training_and_Holdout.merge(valid_person_ids, on="person_id")
        y_train = Outcome_df.merge(train_person_ids, on="person_id")
        y_test = Outcome_df.merge(valid_person_ids, on="person_id")
        X_train_no_ind = X_train_no_ind.sort_values("person_id")
        X_test_no_ind = X_test_no_ind.sort_values("person_id")
        y_train = y_train.sort_values("person_id")
        y_test = y_test.sort_values("person_id")
        print("first 10 train person ids::", list(X_train_no_ind["person_id"])[0:10])
        print("first 10 train person ids::", list(y_train["person_id"])[0:10])
        print("first 10 test person ids::", list(X_test_no_ind["person_id"])[0:10])
        print("first 10 test person ids::", list(y_test["person_id"])[0:10])
        y_train = y_train.set_index("person_id")
        y_test = y_test.set_index("person_id")
        X_train, X_test = X_train_no_ind.set_index("person_id"), X_test_no_ind.set_index("person_id")
        # X_train_no_ind, X_test_no_ind, y_train, y_test = train_test_split(Training_and_Holdout, Outcome, train_size=0.9, random_state=1)
        # X_train, X_test = X_train_no_ind.set_index("person_id"), X_test_no_ind.set_index("person_id")
        
        X_train = X_train[list(X_train.columns)[0:-1]]
        person_ids_test = list(X_test.index)
        X_test = X_test[list(X_test.columns)[0:-1]]
        print("X_train::", X_train)
        print("Y_train::", y_train)
        return np.array(X_train.values.tolist()), np.array(X_test.values.tolist()), np.array(list(y_train["outcome"])), np.array(list(y_test["outcome"])), person_ids_test

    def train_mTan_model_main(train_person_ids, valid_person_ids, train_recent_visits, valid_recent_visits, train_person_info, valid_person_info, train_labels, valid_labels):
        print("start pre-processing!!!")
        visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls = pre_processing_visits(train_person_ids.toPandas(), train_person_info.toPandas(), train_recent_visits.toPandas(), train_labels.toPandas(), setup="both")
        # torch.save(visit_tensor_ls, "train_visit_tensor_ls")
        # torch.save(mask_ls, "train_mask_ls")
        # torch.save(time_step_ls, "train_mask_ls")
        # torch.save(label_tensor_ls, "train_label_tensor_ls")
        # torch.save(person_info_ls, "train_person_info_ls")
        
        # visit_tensor_ls, mask_ls = remove_empty_columns(visit_tensor_ls, mask_ls)

        valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls = pre_processing_visits(valid_person_ids.toPandas(), valid_person_info.toPandas(), valid_recent_visits.toPandas(), valid_labels.toPandas(), setup="both")
        print("finish pre-processing!!!")
        visit_tensor_ls, mask_ls, non_empty_column_ids = remove_empty_columns(visit_tensor_ls, mask_ls)
        valid_visit_tensor_ls, valid_mask_ls = remove_empty_columns_with_non_empty_cls(valid_visit_tensor_ls, valid_mask_ls, non_empty_column_ids)

        data_min, data_max = get_data_min_max(visit_tensor_ls, mask_ls)

        train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_tensor_ls, data_min, data_max)

        valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_tensor_ls, data_min, data_max)
        # train_dataset = LongCOVIDVisitsDataset2(train_person_ids, train_person_info, train_recent_visits, train_labels)
        # valid_dataset = LongCOVIDVisitsDataset2(valid_person_ids, valid_person_info, valid_recent_visits, valid_labels)

        # Construct dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)

        epochs=30
        print(train_dataset.__getitem__(1)[0])
        dim = train_dataset.__getitem__(1)[0].shape[-1]
        print("data shape::", train_dataset.__getitem__(1)[0].shape)
        print("mask shape::", train_dataset.__getitem__(1)[1].shape)
        print("dim::", dim)
        print(data_min)
        latent_dim=20
        rec_hidden=64
        learn_emb=True
        enc_num_heads=1
        num_ref_points=128
        gen_hidden=30
        dec_num_heads=1
        static_input_dim = train_dataset.__getitem__(1)[3].shape[-1]
        classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print("device::", device)
        rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=128, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
        dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=128, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
        lr = 0.0002

        write_to_pickle(rec, "mTans_rec")
        write_to_pickle(dec, "mTans_dec")
        write_to_pickle(classifier, "mTans_classifier")
        # read_from_pickle()
        print("save models successfully")

        rec = rec.to(device)
        dec = dec.to(device)
        classifier = classifier.to(device)

        return train_mTans(lr, True, 0.01, 100, 1, dim, latent_dim, rec, dec, classifier, epochs, train_loader, valid_loader, is_kl=True)

    train_person_ids, valid_person_ids, train_recent_visits, valid_recent_visits, train_person_info, valid_person_info, train_labels, valid_labels = prepare_sequence_data()
    X_train, X_test, y_train, y_test, person_ids_test = prepare_static_data(train_person_ids, valid_person_ids, Long_COVID_Silver_Standard)

    Y_true, Y_mTans, _, _ = train_mTan_model_main(train_person_ids, valid_person_ids, train_recent_visits, valid_recent_visits, train_person_info, valid_person_info, train_labels, valid_labels)

    def train_simple_models(X_train,y_train, X_test, y_test):
        print(X_train)
        lrc = LogisticRegression(penalty='l2', solver='liblinear', random_state=0, max_iter=500).fit(X_train, y_train)
        lrc2 = LogisticRegression(penalty='l2', class_weight='balanced', solver='liblinear', random_state=0, max_iter=500).fit(X_train, y_train)
        rfc = RandomForestClassifier().fit(X_train, y_train)
        gbc = GradientBoostingClassifier().fit(X_train, y_train)

        # lrc_sort_features = np.argsort(lrc.coef_.flatten())[-20:]
        # lrc_sort_features_least = np.argsort(lrc.coef_.flatten())[:20]
        # rfc_sort_features = np.argsort(rfc.feature_importances_.flatten())[-20:]
        # rfc_sort_features_least = np.argsort(rfc.feature_importances_.flatten())[:20]
        # plt.bar(np.arange(20), rfc.feature_importances_.flatten()[rfc_sort_features])
        # plt.xticks(np.arange(20), [cols[1:][i] for i in rfc_sort_features], rotation='vertical')
        # plt.tight_layout()
        # plt.show()

        # print("lrc important features:", [cols[1:][int(i)] for i in lrc_sort_features])
        # print("rfc important features:", [cols[1:][int(i)] for i in rfc_sort_features])
        # print("lrc least important features:", [cols[1:][int(i)] for i in lrc_sort_features_least ])
        # print("rfc least important features:", [cols[1:][int(i)] for i in rfc_sort_features_least ])
        # print("combined least important features:", [cols[1:][int(i)] for i in rfc_sort_features_least if i in lrc_sort_features_least])
        # print("column variance: \n", all_patients_summary_fact_table_de_id.var().to_string())
        nn_scaler = StandardScaler().fit(X_train)
        nnc = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 10), random_state=1).fit(nn_scaler.transform(X_train), y_train)

        #preds = clf.predict_proba(Testing)[:,1]

        lr_test_preds = lrc.predict_proba(X_test)[:, 1]
        lr_train_preds = lrc.predict_proba(X_train)[:, 1]
        lr2_test_preds = lrc2.predict_proba(X_test)[:, 1]
        rf_test_preds = rfc.predict_proba(X_test)[:, 1]
        rf_train_preds = rfc.predict_proba(X_train)[:, 1]
        gb_test_preds = gbc.predict_proba(X_test)[:, 1]
        nnc_test_preds = nnc.predict_proba(nn_scaler.transform(X_test))[:, 1]

        print("LR Training Classification Report:\n{}".format(classification_report(y_train, np.where(lr_train_preds > 0.5, 1, 0))))

        #test_df = 
        test_predictions = pd.DataFrame.from_dict({
            'person_id': person_ids_test,
            'lr_outcome': lr_test_preds.tolist(),
            'lr2_outcome': lr2_test_preds.tolist(),
            'rf_outcome': rf_test_preds.tolist(),
            'gb_outcome': gb_test_preds.tolist(),
            'nn_outcome': nnc_test_preds.tolist(),
        }, orient='columns')
        
        # test_predictions = test_predictions.merge(Outcome_df, on="person_id", how="left")
        # outcomes = ['lr_outcome', 'lr2_outcome', 'rf_outcome', 'gb_outcome', 'nn_outcome']
        # test_predictions['ens_outcome'] = test_predictions.apply(lambda row: 1 if sum([row[c] for c in outcomes])/len(outcomes) >=0.5 else 0, axis=1)

    # predictions = pd.DataFrame.from_dict({
    #     'person_id': list(all_patients_summary_fact_table_de_id_testing["person_id"]),
    #     'outcome_likelihood': preds.tolist()
    # }, orient='columns')

        return lr_test_preds.tolist(), lr2_test_preds.tolist(), rf_test_preds.tolist(),gb_test_preds.tolist(), nnc_test_preds.tolist()

    lr_test_preds, lr2_test_preds, rf_test_preds,gb_test_preds, nnc_test_preds = train_simple_models(X_train,y_train, X_test, y_test)

    Y_all = np.stack((gb_test_preds, nnc_test_preds, Y_mTans), axis=0)

    # Y_all = np.stack((lr_test_preds, lr2_test_preds, rf_test_preds, gb_test_preds, nnc_test_preds), axis=0)
    Y_all = (Y_all > 0.5).astype(int)
    Y_final_pred = scipy.stats.mode(Y_all, axis=0).mode[0]
    print("pred labels::", Y_final_pred.reshape(-1))
    print("gt labels::", y_test.reshape(-1))
    print("validation classification Report:\n{}".format(classification_report(y_test.reshape(-1).astype(int), Y_final_pred.reshape(-1))))

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2b8fbb2f-c6a4-4402-bcbc-b0925e8e1003"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    top_k_concepts_data=Input(rid="ri.foundry.main.dataset.7b277d99-e39e-4a5f-9058-4e6f65fa7f58"),
    top_k_concepts_data_test=Input(rid="ri.foundry.main.dataset.10a82ffa-f748-4e12-9c88-0f8fc74dcd7f")
)
def train_test_top_k_model(top_k_concepts_data, top_k_concepts_data_test, Long_COVID_Silver_Standard):
    ## get outcome column
    cols = top_k_concepts_data.columns

    Long_COVID_Silver_Standard["outcome"] = Long_COVID_Silver_Standard.apply(lambda x: max([x["pasc_code_after_four_weeks"], x["pasc_code_prior_four_weeks"]]), axis=1)
    Outcome_df = top_k_concepts_data[["person_id"]].merge(Long_COVID_Silver_Standard, on="person_id", how="left")
    Outcome_df = Outcome_df[["person_id", "outcome"]].sort_values('person_id')

    Outcome = list(Outcome_df["outcome"])

    for col in top_k_concepts_data.columns:
        if col not in top_k_concepts_data_test.columns:
            print(col, " not in summary testing set.")
            top_k_concepts_data_test[col] = 0
    for col in top_k_concepts_data_test.columns:
        if col not in top_k_concepts_data.columns:
            print(col, " not in summary training set.")
            top_k_concepts_data_test.drop(col, axis=1)
    Training_and_Holdout = top_k_concepts_data.fillna(0.0).sort_values('person_id')
    Training_and_Holdout = Training_and_Holdout.sort_index(axis=1)
    Testing = top_k_concepts_data_test.fillna(0.0).sort_values('person_id')
    Testing = Testing.sort_index(axis=1)
    if LOAD_TEST == 0:
        X_train_no_ind, _, y_train, y_test = train_test_split(Training_and_Holdout, Outcome, train_size=0.9, random_state=1)
        _, X_test_no_ind, _, _ = train_test_split(Training_and_Holdout, Outcome, train_size=0.9, random_state=1)
    else:
        X_train_no_ind, y_train = Training_and_Holdout, Outcome
        X_test_no_ind, y_test = Testing, None
    X_train, X_test = X_train_no_ind.set_index("person_id"), X_test_no_ind.set_index("person_id")

    lrc = LogisticRegression(penalty='l2', solver='liblinear', random_state=1, max_iter=500).fit(X_train, y_train)
    lrc2 = LogisticRegression(penalty='l2', solver='liblinear', random_state=1, max_iter=500, class_weight='balanced').fit(X_train, y_train)
    rfc = RandomForestClassifier(random_state=1).fit(X_train, y_train)
    gbc = GradientBoostingClassifier(random_state=1).fit(X_train, y_train)
    nn_scaler = StandardScaler().fit(X_train)
    nnc = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 10), random_state=1).fit(nn_scaler.transform(X_train), y_train)

    lrc_sort_features = np.argsort(lrc.coef_.flatten())[-21:-1]
    lrc_sort_features_least = np.argsort(lrc.coef_.flatten())[:20]
    lrc2_sort_features = np.argsort(lrc2.coef_.flatten())[-21:-1]
    lrc2_sort_features_least = np.argsort(lrc2.coef_.flatten())[:20]
    rfc_sort_features = np.argsort(rfc.feature_importances_.flatten())[-21:-1]
    rfc_sort_features_least = np.argsort(rfc.feature_importances_.flatten())[:20]
    gbc_sort_features = np.argsort(gbc.feature_importances_.flatten())[-21:-1]
    gbc_sort_features_least = np.argsort(gbc.feature_importances_.flatten())[:20]

    fig, axs = plt.subplots(2,2, constrained_layout=True, figsize=(8,8))
    fig.suptitle('Twenty Most Important Features Per Model (Automated)')
    axs[0][0].bar(x = np.arange(20), height = rfc.feature_importances_.flatten()[rfc_sort_features], tick_label=[cols[1:][i] for i in rfc_sort_features])
    axs[0][0].set_xticklabels(labels=[cols[1:][i] for i in rfc_sort_features], fontdict={'rotation':"vertical",'size':'xx-small'})
    axs[0][0].set_ylabel("Purity-based Importance", size='small')
    axs[0][0].set_title("Random Forest Classifier")
    
    axs[0][1].bar(x = np.arange(20), height = lrc.coef_.flatten()[lrc_sort_features], tick_label=[cols[1:][i] for i in lrc_sort_features])
    axs[0][1].set_xticklabels(labels=[cols[1:][i] for i in lrc_sort_features], fontdict={'rotation':"vertical", 'size':'xx-small'})
    axs[0][1].set_ylabel("Coefficient-based Importance", size='small')
    axs[0][1].set_title("Logistic Regression Classifier")

    axs[1][0].bar(x = np.arange(20), height = gbc.feature_importances_.flatten()[gbc_sort_features], tick_label=[cols[1:][i] for i in gbc_sort_features])
    axs[1][0].set_xticklabels(labels=[cols[1:][i] for i in gbc_sort_features], fontdict={'rotation':"vertical",'size':'xx-small'})
    axs[1][0].set_ylabel("Purity-based Importance", size='small')
    axs[1][0].set_title("Gradient Boosted Classifier")

    axs[1][1].bar(x = np.arange(20), height = lrc2.coef_.flatten()[lrc2_sort_features], tick_label=[cols[1:][i] for i in lrc2_sort_features])
    axs[1][1].set_xticklabels(labels=[cols[1:][i] for i in lrc2_sort_features], fontdict={'rotation':"vertical", 'size':'xx-small'})
    axs[1][1].set_ylabel("Coefficient-based Importance", size='small')
    axs[1][1].set_title(" Balanced Logistic Regression Classifier")
    plt.show()

    print("lrc important features:", [cols[1:][int(i)] for i in lrc_sort_features])
    print("rfc important features:", [cols[1:][int(i)] for i in rfc_sort_features])
    print("lrc least important features:", [cols[1:][int(i)] for i in lrc_sort_features_least ])
    print("rfc least important features:", [cols[1:][int(i)] for i in rfc_sort_features_least ])
    print("combined least important features:", [cols[1:][int(i)] for i in rfc_sort_features_least if i in lrc_sort_features_least])

    print(X_train.shape)
    print(X_test.shape)
    lr_test_preds = lrc.predict_proba(X_test)[:, 1]
    lr_train_preds = lrc.predict_proba(X_train)[:, 1]
    lr2_test_preds = lrc2.predict_proba(X_test)[:, 1]
    rf_test_preds = rfc.predict_proba(X_test)[:, 1]
    rf_train_preds = rfc.predict_proba(X_train)[:, 1]
    gb_test_preds = gbc.predict_proba(X_test)[:, 1]
    nnc_test_preds = nnc.predict_proba(nn_scaler.transform(X_test))[:, 1]

    predictions = pd.DataFrame.from_dict({
        'person_id': list(X_test_no_ind["person_id"]),
        'tk_lr_outcome': lr_test_preds.tolist(),
        'tk_lr2_outcome': lr2_test_preds.tolist(),
        'tk_rf_outcome': rf_test_preds.tolist(),
        'tk_gb_outcome': gb_test_preds.tolist(),
        'tk_nn_outcome': nnc_test_preds.tolist(),
    }, orient='columns')
    
    if MERGE_LABEL  == 1:
        predictions = predictions.merge(Outcome_df, on="person_id", how="left")
    outcomes = ['tk_lr2_outcome', 'tk_rf_outcome', 'tk_gb_outcome']
    predictions['tk_ens_outcome'] = predictions.apply(lambda row: 1 if sum([row[c] for c in outcomes])/len(outcomes) >=0.5 else 0, axis=1)

    return predictions

@transform_pandas(
    Output(rid="ri.vector.main.execute.67b2b111-72ba-4b82-a7ce-e4d72f6b05af"),
    Long_COVID_Silver_Standard=Input(rid="ri.foundry.main.dataset.3ea1038c-e278-4b0e-8300-db37d3505671"),
    top_k_concepts_data=Input(rid="ri.foundry.main.dataset.7b277d99-e39e-4a5f-9058-4e6f65fa7f58"),
    top_k_concepts_data_test=Input(rid="ri.foundry.main.dataset.10a82ffa-f748-4e12-9c88-0f8fc74dcd7f")
)
def train_test_top_k_model_shap(top_k_concepts_data, top_k_concepts_data_test, Long_COVID_Silver_Standard):
    ## get outcome column
    print("starting")
    cols = top_k_concepts_data.columns

    # Long_COVID_Silver_Standard = Long_COVID_Silver_Standard.withColumn("outcome", F.greatest(F.col("pasc_code_after_four_weeks"), F.col("pasc_code_prior_four_weeks")))
    # outcome_df = top_k_concepts_data.select("person_id").join(Long_COVID_Silver_Standard, "person_id", "left")
    # outcome_df = outcome_df.select("person_id", "outcome").orderBy('person_id')
    # Outcome = list(outcome_df.select("outcome").toPandas()['outcome'])

    # shared_train_test_features = [col for col in top_k_concepts_data.columns if col in top_k_concepts_data_test.columns]
    # top_k_concepts_data_test = top_k_concepts_data_test.select(shared_train_test_features)

    Long_COVID_Silver_Standard["outcome"] = Long_COVID_Silver_Standard.apply(lambda x: max([x["pasc_code_after_four_weeks"], x["pasc_code_prior_four_weeks"]]), axis=1)
    Outcome_df = top_k_concepts_data[["person_id"]].merge(Long_COVID_Silver_Standard, on="person_id", how="left")
    Outcome_df = Outcome_df[["person_id", "outcome"]].sort_values('person_id')
    Outcome = list(Outcome_df["outcome"])

    for col in top_k_concepts_data.columns:
        if col not in top_k_concepts_data_test.columns:
            print(col, " not in summary testing set.")
            top_k_concepts_data_test[col] = 0
    for col in top_k_concepts_data_test.columns:
        if col not in top_k_concepts_data.columns:
            print(col, " not in summary training set.")
            top_k_concepts_data_test.drop(col, axis=1)
    
    Training_and_Holdout = top_k_concepts_data.fillna(0.0).sort_values('person_id')
    Training_and_Holdout = Training_and_Holdout.sort_index(axis=1)
    Testing = top_k_concepts_data_test.fillna(0.0).sort_values('person_id')
    Testing = Testing.sort_index(axis=1)
    if LOAD_TEST == 0:
        X_train_no_ind, X_test_no_ind, y_train, y_test = train_test_split(Training_and_Holdout, Outcome, train_size=0.9, random_state=1)
    else:
        X_train_no_ind, y_train = Training_and_Holdout, Outcome
        X_test_no_ind, y_test = Testing, None
    X_train, X_test = X_train_no_ind.set_index("person_id"), X_test_no_ind.set_index("person_id")

    print("Training LRC...")
    lrc = LogisticRegression(penalty='l2', solver='liblinear', random_state=1, max_iter=500).fit(X_train, y_train)
    print("Training LRC2...")
    lrc2 = LogisticRegression(penalty='l2', solver='liblinear', random_state=1, max_iter=500, class_weight='balanced').fit(X_train, y_train)
    print("Training RFC...")
    rfc = RandomForestClassifier(random_state=1).fit(X_train, y_train)
    print("Training GBC...")
    gbc = GradientBoostingClassifier(random_state=1).fit(X_train, y_train)
    # print("Training NN...")
    # nn_scaler = StandardScaler().fit(X_train)
    # nnc = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 10), random_state=1).fit(nn_scaler.transform(X_train), y_train)

    ens_fn = lambda x: (lrc.predict_proba(x) + lrc2.predict_proba(x) + rfc.predict_proba(x) + gbc.predict_proba(x))[:, 1] / 4

    X_test_sample = X_test.head(10)

    print("Starting shap calculations")
    explainer = shap.KernelExplainer(ens_fn, shap.kmeans(X_train, 50))
    shap_values = explainer.shap_values(X_test_sample)

    # Issues with the Enclave don't allow us to save the data, so we perform the visualizations in the same node.
    # shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], X_test_sample.iloc[0], feature_names=cols[1:], show=False)
    shap.summary_plot(shap_values, X_test_sample.values, feature_names=X_test_sample.columns, show=False)
    plt.tight_layout()
    plt.show()
    
    # write_to_pickle(explainer.expected_value, "ens_explainer_ev")
    # write_to_pickle(shap_values, "ens_shap_values")
    print("Finished ens shap")

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
    Output(rid="ri.foundry.main.dataset.78717ca8-ae81-4a08-8df8-d3ec16e75f18"),
    produce_dataset=Input(rid="ri.foundry.main.dataset.ae1c108c-1813-47ba-831c-e5a37c599c49"),
    train_sequential_model_3=Input(rid="ri.foundry.main.dataset.4fa4a34a-a9e7-489f-a499-023c2d4c44ac")
)
def valid_mTan(train_sequential_model_3, produce_dataset):
    valid_person_ids, valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_ls, data_min, data_max = read_from_pickle(produce_dataset, "valid_data.pickle")
    person_ids, visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_ls, _,_ = read_from_pickle(produce_dataset, "valid_data.pickle")

    valid_dataset = LongCOVIDVisitsDataset2(valid_visit_tensor_ls, valid_mask_ls, valid_time_step_ls, valid_person_info_ls, valid_label_ls, data_min, data_max)
    train_dataset = LongCOVIDVisitsDataset2(visit_tensor_ls, mask_ls, time_step_ls, person_info_ls, label_ls, data_min, data_max)
    static_input_dim = valid_dataset.__getitem__(1)[3].shape[-1]

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=LongCOVIDVisitsDataset2.collate_fn)
    dim = valid_dataset.__getitem__(1)[0].shape[-1]
    latent_dim=20
    rec_hidden=32
    learn_emb=True
    enc_num_heads=1
    num_ref_points=128
    gen_hidden=30
    dec_num_heads=1
    classifier = create_classifier(latent_dim, 20, has_static=True, static_input_dim=static_input_dim)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    rec = enc_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden, embed_time=32, learn_emb=learn_emb, num_heads=enc_num_heads, device=device)
    dec = dec_mtan_rnn(dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden, embed_time=32, learn_emb=learn_emb, num_heads=dec_num_heads, device=device)
    lr = 0.001

    

    rec = rec.to(device)
    dec = dec.to(device)
    classifier = classifier.to(device)

    rec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_rec.pickle'))
    dec.load_state_dict(read_model_from_pickle(train_sequential_model_3, 'mTans_dec.pickle'))
    classifier.load_state_dict(read_model_from_pickle(train_sequential_model_3, "mTans_classifier.pickle"))

    
    # valid_pred_labels =  evaluate_classifier(rec, valid_loader,latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)
    val_loss, val_acc, val_auc, val_recall, val_precision,true, pred_labels, pred_scores = evaluate_classifier_final(rec, valid_loader,latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)
    loss, acc, auc, recall, precision,true, labels, scores = evaluate_classifier_final(rec, valid_loader,latent_dim=latent_dim, classify_pertp=False, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)

    print("validation loss::", val_loss)
    print("validation accuracy::", val_acc)
    print("validation auc score::", val_auc)
    print("validation recall::", val_recall)
    print("validation precision score::", val_precision)

    valid_predictions = pd.DataFrame.from_dict({
            'person_id': list(valid_person_ids) + list(person_ids),
            'mTans_outcome': pred_scores.tolist() + scores.tolist()
        }, orient='columns')

    return valid_predictions

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.def6f994-533b-46b8-95ab-3708d867119c"),
    test_mTan=Input(rid="ri.foundry.main.dataset.c6673c1c-6c0c-4ff1-8b29-cc4c268e650a"),
    train_test_model=Input(rid="ri.foundry.main.dataset.ea6c836a-9d51-4402-b1b7-0e30fb514fc8"),
    train_test_top_k_model=Input(rid="ri.foundry.main.dataset.2b8fbb2f-c6a4-4402-bcbc-b0925e8e1003"),
    valid_mTan=Input(rid="ri.foundry.main.dataset.78717ca8-ae81-4a08-8df8-d3ec16e75f18")
)
def validation_metrics( train_test_model, train_test_top_k_model, valid_mTan, test_mTan):
    train_test_top_k_model = train_test_top_k_model.drop("outcome", axis=1)
    if not LOAD_TEST:
        df = train_test_model.merge(train_test_top_k_model, on="person_id", how="left").merge(valid_mTan, on="person_id", how="left")
    else:
        df = train_test_model.merge(train_test_top_k_model, on="person_id", how="left").merge(test_mTan, on="person_id", how="left")

    outcomes = [i for i in df.columns if i.endswith("_outcome") and not "ens" in i and not "nn" in i]
    print(outcomes)
    df['all_ens_outcome'] = df.apply(lambda row: sum([row[c] for c in outcomes])/len(outcomes), axis=1)
    df['all_ens_outcome'] = df.apply(lambda row: (row["mTans_outcome"] + row["all_ens_outcome"])/2, axis=1)
    for i in [i for i in df.columns if i != "person_id"]:
        print("{} Classification Report:\n{}".format(i, classification_report(df["outcome"], np.where(df[i] > 0.5, 1, 0))))
        print(i, " MAE:", mean_absolute_error(df['outcome'], np.where(df[i] > 0.5, 1, 0)))
        print(i, " Brier score:", brier_score_loss(df['outcome'], df[i]))
        print(i, " AP:", average_precision_score(df['outcome'], df[i]))
        print(i, " ROC AUC:", roc_auc_score(df['outcome'], df[i]))
        print("-"*10)

    return df

