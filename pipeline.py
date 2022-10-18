from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

import pandas as pd
from sklearn.linear_model import LogisticRegression

@transform_pandas(
    Output(rid="ri.vector.main.execute.fab1d6ae-e7fb-434e-bf54-ddc10591ac6d"),
    LL_DO_NOT_DELETE_REQUIRED_concept_sets_all=Input(rid="ri.foundry.main.dataset.029aa987-cfef-48fc-bf45-cffd3792cd93"),
    LL_concept_sets_fusion_everyone=Input(rid="ri.foundry.main.dataset.b36c87be-4e43-4f55-a1b2-fc48b0576a77")
)
#The purpose of this node is to optimize the user's experience connecting a customized concept set "fusion sheet" input data frame to replace LL_concept_sets_fusion_everyone.

def customized_concept_set_input(LL_concept_sets_fusion_everyone, LL_DO_NOT_DELETE_REQUIRED_concept_sets_all):
    PM_LL_DO_NOT_DELETE_REQUIRED_concept_sets_all = LL_DO_NOT_DELETE_REQUIRED_concept_sets_all

    required = LL_DO_NOT_DELETE_REQUIRED_concept_sets_all
    customizable = LL_concept_sets_fusion_everyone
    
    df = required.join(customizable, on = required.columns, how = 'outer')
    
    return df

#################################################
## Global imports and functions included below ##
#################################################

@transform_pandas(
    Output(rid="ri.vector.main.execute.4156447b-8bad-4af5-9150-010e304fa65a"),
    LL_DO_NOT_DELETE_REQUIRED_concept_sets_all=Input(rid="ri.foundry.main.dataset.029aa987-cfef-48fc-bf45-cffd3792cd93"),
    LL_concept_sets_fusion_everyone=Input(rid="ri.foundry.main.dataset.b36c87be-4e43-4f55-a1b2-fc48b0576a77")
)
#The purpose of this node is to optimize the user's experience connecting a customized concept set "fusion sheet" input data frame to replace LL_concept_sets_fusion_everyone.

def customized_concept_set_input_testing(LL_concept_sets_fusion_everyone, LL_DO_NOT_DELETE_REQUIRED_concept_sets_all):
    PM_LL_DO_NOT_DELETE_REQUIRED_concept_sets_all = LL_DO_NOT_DELETE_REQUIRED_concept_sets_all

    required = LL_DO_NOT_DELETE_REQUIRED_concept_sets_all
    customizable = LL_concept_sets_fusion_everyone
    
    df = required.join(customizable, on = required.columns, how = 'outer')
    
    return df

#################################################
## Global imports and functions included below ##
#################################################

