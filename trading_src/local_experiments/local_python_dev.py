import pickle
import glob
import sys

import pandas as pd
import numpy as np
import xgboost as xgb

from azureml.core import Run
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

sys.path.append("trading_src")
import helpers.ml_models_managementHelper as mm
import helpers.load_dataHelper as dh
import logics.features_engineering as fe
import logics.backtest as bk
import logics.data_labeling as dl


path = "data/bitcoin_prepared_data/1H"
all_files = glob.glob(path + "/*.csv")
df = pd.concat((pd.read_csv(f) for f in all_files))

#for year in years:
#    file = glob.glob(path + "/*" + str(year) + ".csv")[0]
#    df = pd.read_csv(file, index_col="Timestamp", parse_dates=['Timestamp'])
#    df_list.append(df)
#df = pd.concat(df_list, axis=0, ignore_index=False)

df = df.set_index(df["Timestamp"])
df = df.drop(["Timestamp"], axis=1)

df.index = pd.to_datetime(df.index)

print(df)
df = df.drop(["Open", "High", "Low", "Close", "Volume_(BTC)"], axis=1)
print(df)

#df = fe.add_TA(df)
#X_train = df[(df.index.year >= 2016) & (df.index.year <= 2019)]
#X_test = df[df.index.year == 2020]

#window_size = 11
#y_train = dl.create_labels(X_train, window_size)
#y_test = dl.create_labels(X_test, window_size)


## prepare data for XGBoost training and testing
#dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
#dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X_train.columns))

## model training
#print("model training started")
#xg_reg = xgb.train(params={"objective":"multi:softprob", 'num_class': 3, 'colsample_bytree': 0.3,
#                           'learning_rate': 0.1, 'max_depth': 15, 'alpha': 10}, 
#                   dtrain=dtrain, 
#                   num_boost_round=10)

#y_pred = xg_reg.predict(dtest)
#y_best_preds = np.asarray([np.argmax(line) for line in y_pred])

#target_names = ['0 SELL', '1 BUY', '2 HOLD']
#classification_r = classification_report(y_test, y_best_preds, target_names=target_names, output_dict=False)

#print(classification_r)

#confusion_m = confusion_matrix(y_test, y_best_preds) 
#print(confusion_m)