import pickle
import torch
import numpy as np
import pyspark.pandas as ps
from pyspark.sql import SparkSession
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
    

def spark_to_pandas(df):
    return df.to_pandas()

def pandas_to_spark(df):
    spark = SparkSession.builder \
    .master("local[1]").config("spark.driver.memory", "500g").getOrCreate()
    return spark.createDataFrame(df)