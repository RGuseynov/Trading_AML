import sys

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import os
import pickle
import json
import glob
import datetime

sys.path.append("src")
import helpers.ml_models_managementHelper as mm
import helpers.load_dataHelper as dh
import logics.features_engineering as fe
import logics.backtest as bk


# informations of a run
ml_experiment = {}
ml_experiment["general_informations"] = {}

# model name, iteration and date for management purpose
ml_experiment["general_informations"]["model_name_folder"] = "testing"
ml_experiment["general_informations"]["iteration_set"] = True


mm1 = mm.ManagementHelper(ml_experiment)
iteration = 0
parameters1 = [1,7,8,15]
parameters2 = [1]
parameters3 = [1]

results = {}

# sortir les parametres fixes de la boucle (comme la data si pas de variation)
for param1 in parameters1:
    for param2 in parameters2:
        for param3 in parameters3:

            iteration += 1
            ml_experiment["general_informations"]["iteration"] = iteration

            # data source parameters
            ml_experiment["data_folder"] = "bitcoin_prepared_data"
            ml_experiment["timeframe_data"] = "30T"
            ml_experiment["years_for_training"] = [2018, 2019]
            ml_experiment["years_for_test"] = [2020]

            # model parameters
            ml_experiment["model_hyperparameters"] = {"objective":"binary:logistic",'colsample_bytree': 0.3,
                                                      'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10}
            ml_experiment["num_boost_round"] = param1

            # backtest parameters
            ml_experiment["backtest_parameters_1"] = {"thresold": 0, "cash": 10000, "commission": 0}
            ml_experiment["backtest_parameters_2"] = {"thresold": 0, "cash": 10000, "commission": 0.001}


            # load data
            df_train = dh.load_years(ml_experiment["data_folder"], ml_experiment["timeframe_data"], ml_experiment["years_for_training"])
            df_test = dh.load_years(ml_experiment["data_folder"], ml_experiment["timeframe_data"], ml_experiment["years_for_test"])


            # process data
            df_train = fe.add_TA(df_train)
            df_test = fe.add_TA(df_test)


            # set objective
            X_train = df_train.iloc[:-5,:].copy()
            y_train = df_train.iloc[5:,df_train.columns.get_loc("Close")].copy()
            y_train = y_train.values > X_train["Close"]

            X_test = df_test.iloc[:-5,:].copy()
            y_test = df_test.iloc[5:,df_test.columns.get_loc("Close")].copy()
            y_test = y_test.values > X_test["Close"]


            # log features and target information
            ml_experiment["features_informations"] = {}
            ml_experiment["features_informations"]["features"] = list(X_train.columns)
            ml_experiment["features_informations"]["target"] = "predict if Up or Down 5 timeframes later"


            # prepare data for XGBoost training and testing
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=ml_experiment["features"])
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=ml_experiment["features"])


            # model training
            print("model training started")
            ml_experiment["begin_time"] = datetime.datetime.now().replace(microsecond=0)
            print(ml_experiment["begin_time"])
            xg_reg_model = xgb.train(params=ml_experiment["model_hyperparameters"], 
                               dtrain=dtrain, 
                               num_boost_round=ml_experiment["num_boost_round"])
            ml_experiment["traning_time"] = datetime.datetime.now().replace(microsecond=0) - ml_experiment["begin_time"]
            print(ml_experiment["traning_time"])

            # make prediction and save in the test df
            y_pred = xg_reg_model.predict(dtest)
            X_test["Prediction"] = y_pred

            # evaluate predictions
            y_pred_round = [round(value) for value in y_pred]
            accuracy = accuracy_score(y_test, y_pred_round)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            f1 = f1_score(y_test, y_pred_round, average='binary')
            print("F1_score: {}", f1)
            #cm = confusion_matrix(y_test, y_pred_round) 

            # log evaluations metrics
            ml_experiment["model_metrics"] = {}
            ml_experiment["model_metrics"]["Accuracy"] = str(accuracy)
            ml_experiment["model_metrics"]["F1_score"] = str(f1)


            # running backtest with backtest module
            bk_output_1 = bk.do_back_test(X_test, bk.BinaryPrediction, ml_experiment["backtest_parameters_1"]["thresold"], 
                                          ml_experiment["backtest_parameters_1"]["cash"], ml_experiment["backtest_parameters_1"]["commission"])
            ml_experiment["backtest_results_1"] = bk_output_1.to_dict()
            bk_output_2 = bk.do_back_test(X_test, bk.BinaryPrediction, ml_experiment["backtest_parameters_2"]["thresold"], 
                                          ml_experiment["backtest_parameters_2"]["cash"], ml_experiment["backtest_parameters_2"]["commission"])
            ml_experiment["backtest_results_2"] = bk_output_2.to_dict()
            {"thresold": 0, "cash": 10000, "commission": 0}


            mm1.ml_experiment.update(ml_experiment)
            mm1.save_experiment(xg_reg_model)


            results[ml_experiment["general_informations"]["iteration"]] = ml_experiment["backtest_results_1"]["Equity Final [$]"]


mm1.save_iteration_results(results)
