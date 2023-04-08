import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import pyspark.sql.functions as F

from joblib import dump, load

import os

def train_top_k_models(top_k_concepts_data, Long_COVID_Silver_Standard, show_stats = False):
    ## get outcome column
    cols = top_k_concepts_data.columns

    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    dump(cols, os.path.join(root_dir, "model_checkpoints/topk_metadata"))

    top_k_concepts_data = top_k_concepts_data.toPandas()
    Long_COVID_Silver_Standard = Long_COVID_Silver_Standard.toPandas()
    Long_COVID_Silver_Standard.fillna(0)

    Long_COVID_Silver_Standard["outcome"] = Long_COVID_Silver_Standard.apply(lambda x: max([x["pasc_code_after_four_weeks"], x["pasc_code_prior_four_weeks"]]), axis=1)
    
    Outcome_df = top_k_concepts_data[["person_id"]].merge(Long_COVID_Silver_Standard, on="person_id", how="left")
    Outcome_df = Outcome_df[["person_id", "outcome"]].sort_values('person_id')
    Outcome_df["outcome"]=Outcome_df["outcome"].replace(np.nan, 0)

    Outcome = list(Outcome_df["outcome"])

    # for col in top_k_concepts_data.columns:
    #     if col not in top_k_concepts_data_test.columns:
    #         print(col, " not in summary testing set.")
    #         top_k_concepts_data_test[col] = 0
    # for col in top_k_concepts_data_test.columns:
    #     if col not in top_k_concepts_data.columns:
    #         print(col, " not in summary training set.")
    #         top_k_concepts_data_test.drop(col, axis=1)
    Training_and_Holdout = top_k_concepts_data.fillna(0.0).sort_values('person_id')
    Training_and_Holdout = Training_and_Holdout.sort_index(axis=1)
    # Testing = top_k_concepts_data_test.fillna(0.0).sort_values('person_id')
    # Testing = Testing.sort_index(axis=1)

    X_train_no_ind, y_train = Training_and_Holdout, Outcome
    X_train = X_train_no_ind[sorted(X_train_no_ind)].set_index("person_id")

    lrc = LogisticRegression(penalty='l2', solver='liblinear', random_state=1, max_iter=500).fit(X_train, y_train)
    lrc2 = LogisticRegression(penalty='l2', solver='liblinear', random_state=1, max_iter=500, class_weight='balanced').fit(X_train, y_train)
    rfc = RandomForestClassifier(random_state=1).fit(X_train, y_train)
    gbc = GradientBoostingClassifier(random_state=1).fit(X_train, y_train)
    nn_scaler = StandardScaler().fit(X_train)
    nnc = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 10), random_state=1).fit(nn_scaler.transform(X_train), y_train)

    for model, name in [(lrc, "lrc_topk"), (lrc2, "lrc_bal_topk"), (rfc, "rfc_topk"), (gbc, "gbc_topk"), (nnc, "nnc_topk")]:
        dump(model, os.path.join(root_dir, "model_checkpoints/" + name))

    
    if show_stats:

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


def train_static_models(all_patients_summary_fact_table_de_id, Long_COVID_Silver_Standard, show_stats = False):



    all_patients_summary_fact_table_de_id = all_patients_summary_fact_table_de_id.toPandas()
    Long_COVID_Silver_Standard = Long_COVID_Silver_Standard.toPandas()

    static_cols = ['person_id','total_visits', 'age']
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    cols = static_cols + [col for col in all_patients_summary_fact_table_de_id.columns if 'indicator' in col]
    dump(cols, os.path.join(root_dir, "model_checkpoints/static_metadata"))
    
    ## get outcome column
    Long_COVID_Silver_Standard["outcome"] = Long_COVID_Silver_Standard.apply(lambda x: max([x["pasc_code_after_four_weeks"], x["pasc_code_prior_four_weeks"]]), axis=1)
    Outcome_df = all_patients_summary_fact_table_de_id[["person_id"]].merge(Long_COVID_Silver_Standard, on="person_id", how="left")
    Outcome_df = Outcome_df[["person_id", "outcome"]].sort_values('person_id')
    Outcome_df["outcome"]=Outcome_df["outcome"].replace(np.nan, 0)


    Outcome = list(Outcome_df["outcome"])

    Training_and_Holdout = all_patients_summary_fact_table_de_id[cols].fillna(0.0).sort_values('person_id')
    Training_and_Holdout = Training_and_Holdout.sort_index(axis=1)

    X_train_no_ind, y_train = Training_and_Holdout, Outcome
    X_train = X_train_no_ind[sorted(X_train_no_ind)].set_index("person_id")

    lrc_params = {'C': 1, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
    lrc2_params = {'C': 1, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 500, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
    rfc_params = {'bootstrap': False, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 400, 'n_jobs': -1, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
    gbc_params = {'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.075, 'loss': 'exponential', 'max_depth': 12, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 0.001, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 500, 'n_iter_no_change': None, 'random_state': None, 'subsample': 0.618, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}

    lrc = LogisticRegression(**lrc_params).fit(X_train, y_train)
    lrc2 = LogisticRegression(**lrc2_params).fit(X_train, y_train)
    rfc = RandomForestClassifier(**rfc_params).fit(X_train, y_train)
    gbc = GradientBoostingClassifier(**gbc_params).fit(X_train, y_train)

    for model, name in [(lrc, "lrc_static"), (lrc2, "lrc_bal_static"), (rfc, "rfc_static"), (gbc, "gbc_static")]:
        dump(model, os.path.join(root_dir, "model_checkpoints/") + name)

    if show_stats:

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

def sklearn_models_predict(top_k_data, static_data):
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    top_k_cols = load(os.path.join(root_dir, "model_checkpoints/topk_metadata"))
    static_cols = load(os.path.join(root_dir, "model_checkpoints/static_metadata"))

    for col in top_k_cols:
        if col not in list(top_k_data.columns):
            top_k_data = top_k_data.withColumn(col, F.lit(0))
    for col in list(top_k_data.columns):
        if col not in top_k_cols:
            top_k_data = top_k_data.drop(col)

    for col in static_cols:
        if col not in list(static_data.columns):
            static_data = static_data.withColumn(col, F.lit(0))
    for col in list(static_data.columns):
        if col not in static_cols:
            static_data = static_data.drop(col)

    top_k_data = top_k_data.toPandas().fillna(0.0).sort_values('person_id')
    static_data =  static_data.toPandas()[static_cols].fillna(0.0).sort_values('person_id')

    person_id = list(top_k_data['person_id'])

    top_k_data = top_k_data[sorted(top_k_data)].set_index("person_id")
    static_data = static_data[sorted(static_data)].set_index("person_id")
    
    models = ["gbc_static", "gbc_topk", "lrc_bal_static", "lrc_bal_topk", "lrc_static", "lrc_topk", "rfc_static", "rfc_topk"]
    model_preds = {"person_id": person_id}
    for model_name in models:
        model = load(os.path.join(root_dir, "model_checkpoints/" + model_name))
        data = top_k_data if model_name[-1] == 'k' else static_data
        model_pred = model.predict_proba(data)[:, 1]
        model_preds[model_name] = model_pred

    predictions = pd.DataFrame.from_dict(model_preds)
    predictions["outcome_proba"] = predictions[[c for c in predictions.columns if c != "person_id"]].mean(axis=1)
    predictions["outcome"] = predictions.apply(lambda r: 1 if r["outcome_proba"] > 0.5 else 0, axis=1)
    predictions.to_csv(os.path.join(root_dir, "predictions.csv"), encoding='utf-8', index=False)

    

