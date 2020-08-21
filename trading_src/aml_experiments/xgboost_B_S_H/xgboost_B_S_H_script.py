import sys
import os
import pickle
import glob
import json

import pandas as pd
import numpy as np
import xgboost as xgb

from azureml.core import Run
from azureml.core import Model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

sys.path.append("trading_src")
import helpers.save_plotsHelper as ph
import helpers.ml_models_managementHelper as mm
import helpers.load_dataHelper as dh
import logics.features_engineering as fe
import logics.backtest as bk
import logics.data_labeling as dl


# training info, data source parameters
ml_experiment = {}
ml_experiment["data_folder"] = "bitcoin_prepared_data"
ml_experiment["timeframe_data"] = "1H"
ml_experiment["years_for_training"] = [2016, 2019]
ml_experiment["years_for_test"] = [2020]

# model parameters
ml_experiment["model_hyperparameters"] = {"objective":"multi:softprob", "num_class": 3, 'colsample_bytree': 0.3,
                                          'learning_rate': 0.1, 'max_depth': 15, 'alpha': 10}
ml_experiment["Num_boost_round"] = 15

# backtests parameters
ml_experiment["backtest_parameters_1"] = {"thresold": 0, "cash": 10000, "commission": 0}
ml_experiment["backtest_parameters_2"] = {"thresold": 0, "cash": 10000, "commission": 0.001}


# Get the experiment run context
run = Run.get_context()

# load the diabetes data (passed as an input dataset)
print("Loading Data...")
df = run.input_datasets['bitcoin'].to_pandas_dataframe()

df = df.set_index(df["Timestamp"])
df = df.drop(["Timestamp"], axis=1)

df.index = pd.to_datetime(df.index)


df = fe.add_TA(df)
X_train = df[(df.index.year >= ml_experiment["years_for_training"][0]) & (df.index.year <= ml_experiment["years_for_training"][1])]
X_test = df[df.index.year == ml_experiment["years_for_test"][0]]

window_size = 11
y_train = dl.create_labels(X_train, window_size)
y_test = dl.create_labels(X_test, window_size)

#logs
ml_experiment["features_informations"] = {}
ml_experiment["features_informations"]["features"] = list(X_train.columns)
ml_experiment["features_informations"]["target"] = "predict if local min(BUY:1), max(SELL:0) or not(HOLD:2)"
ml_experiment["features_informations"]["target_window_size"] = window_size

# prepare data for XGBoost training and testing
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X_train.columns))

# model training
print("model training started")
xg_model = xgb.train(params=ml_experiment["model_hyperparameters"], 
                   dtrain=dtrain, 
                   num_boost_round=ml_experiment["Num_boost_round"])

y_pred = xg_model.predict(dtest)
y_best_preds = np.asarray([np.argmax(line) for line in y_pred])
X_test["Prediction"] = y_best_preds

target_names = ['0 SELL', '1 BUY', '2 HOLD']
classification_r = classification_report(y_test, y_best_preds, target_names=target_names, output_dict=True)
ml_experiment["model_metrics"] = {}
ml_experiment["model_metrics"]["classification_report"] = classification_r

confusion_m = confusion_matrix(y_test, y_best_preds) 
cm_aml = {"schema_type": "confusion_matrix",
            "schema_version": "v1",
            "data": {"class_labels": target_names,
                     "matrix": confusion_m.tolist()}}
run.log_confusion_matrix("confusion matrix", cm_aml)

plot_helper = ph.XGBoostPlotHelper(xg_model)
run.log_image("features importances", plot=plot_helper.get_features_importances())

# running backtest
bk_output_1 = bk.do_back_test(X_test, bk.B_S_H, ml_experiment["backtest_parameters_1"]["thresold"], 
                                ml_experiment["backtest_parameters_1"]["cash"], ml_experiment["backtest_parameters_1"]["commission"], "test_sans_frais")
ml_experiment["backtest_results_1"] = bk_output_1.to_dict()
bk_output_2 = bk.do_back_test(X_test, bk.B_S_H, ml_experiment["backtest_parameters_2"]["thresold"], 
                                ml_experiment["backtest_parameters_2"]["cash"], ml_experiment["backtest_parameters_2"]["commission"], "test")
ml_experiment["backtest_results_2"] = bk_output_2.to_dict()

run.log("return no fee", ml_experiment["backtest_results_1"]["Return [%]"])
run.log("return with fee", ml_experiment["backtest_results_2"]["Return [%]"])

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
with open("outputs/xgboost_model_test.pkl", "wb") as file:
    pickle.dump(xg_model, file)
with open("outputs/training_info.json", "w") as file:
    json.dump(ml_experiment, file, indent=4, default=str)

run.complete()



## Separate features and labels
#X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

## Split data into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

## Train a decision tree model
#print('Training a decision tree model')
#model = DecisionTreeClassifier().fit(X_train, y_train)

## calculate accuracy
#y_hat = model.predict(X_test)
#acc = np.average(y_hat == y_test)
#print('Accuracy:', acc)
#run.log('Accuracy', np.float(acc))

## calculate AUC
#y_scores = model.predict_proba(X_test)
#auc = roc_auc_score(y_test,y_scores[:,1])
#print('AUC: ' + str(auc))
#run.log('AUC', np.float(auc))

## plot ROC curve
#fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
#fig = plt.figure(figsize=(6, 4))
## Plot the diagonal 50% line
#plt.plot([0, 1], [0, 1], 'k--')
## Plot the FPR and TPR achieved by our model
#plt.plot(fpr, tpr)
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve')
#run.log_image(name = "ROC", plot = fig)
#plt.show()

#os.makedirs('outputs', exist_ok=True)
## note file saved in the outputs folder is automatically uploaded into experiment record
#with open("outputs/xgboost_model.pkl", "wb") as file:
#    pickle.dump(model. file)

#run.complete()
