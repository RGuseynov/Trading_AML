import pickle
import glob
import sys
import os

import pandas as pd
import numpy as np
import xgboost as xgb

from azureml.core import Run
from azureml.core import Model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

sys.path.append("trading_src")
import helpers.ml_models_managementHelper as mm
import helpers.load_dataHelper as dh
import logics.features_engineering as fe
import logics.backtest as bk
import logics.data_labeling as dl

# Get the experiment run context
run = Run.get_context()

# load the diabetes data (passed as an input dataset)
print("Loading Data...")
df = run.input_datasets['bitcoin'].to_pandas_dataframe()

df = df.set_index(df["Timestamp"])
df = df.drop(["Timestamp"], axis=1)

df.index = pd.to_datetime(df.index)


df = fe.add_TA(df)
X_train = df[(df.index.year >= 2016) & (df.index.year <= 2019)]
X_test = df[df.index.year == 2020]

window_size = 11
y_train = dl.create_labels(X_train, window_size)
y_test = dl.create_labels(X_test, window_size)


# prepare data for XGBoost training and testing
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X_train.columns))

# model training
print("model training started")
xg_reg = xgb.train(params={"objective":"multi:softprob", 'num_class': 3, 'colsample_bytree': 0.3,
                           'learning_rate': 0.1, 'max_depth': 15, 'alpha': 10}, 
                   dtrain=dtrain, 
                   num_boost_round=10)

y_pred = xg_reg.predict(dtest)
y_best_preds = np.asarray([np.argmax(line) for line in y_pred])

target_names = ['0 SELL', '1 BUY', '2 HOLD']
classification_r = classification_report(y_test, y_best_preds, target_names=target_names, output_dict=True)

print(classification_r)

confusion_m = confusion_matrix(y_test, y_best_preds) 
print(confusion_m)

run.log('Classification_result', classification_r)
run.log('Confusion_matrix', confusion_m)

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
with open("outputs/xgboost_model_test.pkl", "wb") as file:
    pickle.dump(xg_reg, file)

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
