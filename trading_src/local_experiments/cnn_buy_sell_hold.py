import sys
import datetime

from collections import Counter

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

# personal modules
sys.path.append("src")
import utils.utils as utils
import helpers.ml_models_managementHelper as mm
import helpers.load_dataHelper as dh
import helpers.save_plotsHelper as ph
import logics.features_engineering as fe
import logics.backtest as bk
import logics.data_labeling as dl
import logics.machine_learning as ml


pd.set_option('display.max_rows', 50)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_columns',10)
pd.set_option('display.width', 150)


# informations of a run, model name and iteration for management purpose
ml_experiment = {}
ml_experiment["general_informations"] = {}
ml_experiment["general_informations"]["model_name_folder"] = "CNN_B_S_H"
ml_experiment["general_informations"]["iteration_set"] = False

mm1 = mm.ManagementHelper(ml_experiment)


# data source parameters
ml_experiment["data_folder"] = "bitcoin_prepared_data"
ml_experiment["timeframe_data"] = "1H"
ml_experiment["years_for_training"] = [2017, 2018, 2019]
ml_experiment["years_for_test"] = [2020]


# hyperparameters
layers = {"Conv2D_1": {"filters": 32, "kernel_size":3, "strides":1, "padding":"same", "activation":"relu", 
                        "use_bias":True, "kernel_initializer":'glorot_uniform', "kernel_regularizer": regularizers.l2(0.0), 
                        "input_shape":(11, 11, 1), "name":"Conv2D_1"},
          "Dropout_Conv_1": {"rate": 0.2, "name": "Dropout_Conv_1"},
          "Conv2D_2": {"filters": 64, "kernel_size":3, "strides":1, "padding":"same", "activation":"relu", 
                        "use_bias":True, "kernel_initializer":'glorot_uniform', "kernel_regularizer": regularizers.l2(0.0), 
                        "name":"Conv2D_2"},
          "MaxPool2D": {"pool_size": (2, 2), "padding": "same", "name": "MaxPool2D"},
          "Dropout_Conv_2": {"rate": 0.2, "name": "Dropout_Conv_2"},
          "Flatten": {"name": "Flatten"},
          "Dense_1": {"units": 128, "activation": "relu", "name": "Dense_1"},
          "Dropout_Dense_1": {"rate": 0.2, "name": "Dropout_Dense_2"},
          "Dense_2": {"units": 3, "activation": "softmax", "name": "Dense_2"}
          }
optimizer = {"Adam": {"learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999, "amsgrad": False}}

ml_experiment["hyperparameters"] = {"layers": layers, 'optimizer': optimizer, "loss": "categorical_crossentropy", 
                                    'validation_split': 0.3, 'batch_size': 80, 'epochs': 3}


# backtests parameters
ml_experiment["backtest_parameters_1"] = {"thresold": 0, "cash": 10000, "commission": 0}
ml_experiment["backtest_parameters_2"] = {"thresold": 0, "cash": 10000, "commission": 0.001}


# load data
df_train = dh.load_years(ml_experiment["data_folder"], ml_experiment["timeframe_data"], ml_experiment["years_for_training"])
df_test = dh.load_years(ml_experiment["data_folder"], ml_experiment["timeframe_data"], ml_experiment["years_for_test"])


# process features columns
df_train = fe.add_TA(df_train)
df_test = fe.add_TA(df_test)

# drop first rows that don't have technical analysis data
df_train = df_train.dropna()
df_test = df_test.dropna()


# set target column with windows_size for local min and max
window_size = 11
X_train = df_train
y_train = dl.create_labels(X_train, window_size)

X_test = df_test
y_test = dl.create_labels(X_test, window_size)


# features normalization
mm_scaler = MinMaxScaler(feature_range=(0, 1))
X_train[X_train.columns] = mm_scaler.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = mm_scaler.transform(X_test[X_test.columns])


# selecting Kbest features
#X_new = SelectKBest(chi2, k=20).fit_transform(X_train, y_train)
selector = SelectKBest(f_classif, k=121)
selector.fit(X_train, y_train)
cols = selector.get_support(indices=True)
X_train = X_train.iloc[:,cols]
X_test = X_test.iloc[:,cols]


# log features columns and target information
ml_experiment["features_informations"] = {}
ml_experiment["features_informations"]["features"] = list(X_train.columns)
ml_experiment["features_informations"]["target"] = "predict if local min(BUY:1), max(SELL:0) or not(HOLD:2)"
ml_experiment["features_informations"]["target_window_size"] = window_size


# handling data inbalance with weighted class
sample_weights, ml_experiment["hyperparameters"]["class_weights"] = utils.get_sample_weights(y_train)


# one hot encoding labels for cnn
y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)


# transforming dataframe into set of images, each feature cell become a "pixel"
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
dim = int(np.sqrt(121))
X_train = utils.reshape_as_image(X_train, dim, dim)
X_test = utils.reshape_as_image(X_test, dim, dim)


# model training
print("model training started")
ml_experiment["begin_time"] = datetime.datetime.now().replace(microsecond=0)

cnn_model = ml.create_model_cnn(ml_experiment["hyperparameters"])

print(cnn_model.summary())

history = cnn_model.fit(X_train, y_train, epochs=ml_experiment["hyperparameters"]["epochs"],
                        batch_size=ml_experiment["hyperparameters"]["batch_size"],
                        validation_split=ml_experiment["hyperparameters"]["validation_split"],
                        sample_weight=sample_weights)

ml_experiment["training_time"] = datetime.datetime.now().replace(microsecond=0) - ml_experiment["begin_time"]


#eval = cnn_model.evaluate(X_test,  y_test, verbose=2)
#print(eval)

# make prediction and save in the test df
y_pred = cnn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis = 1)
df_test["Prediction"] = y_pred_classes

print(y_pred)
print(y_pred_classes)

print(Counter(y_test).keys()) # equals to list(set(y_tests))
print(Counter(y_test).values()) # counts the elements' frequency
print(Counter(y_pred_classes).keys()) # equals to list(set(y_pred_classes))
print(Counter(y_pred_classes).values()) # counts the elements' frequency


# evaluate predictions
#y_pred_round = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes, average=None)
confusion_m = confusion_matrix(y_test, y_pred_classes) 
target_names = ['0 SELL', '1 BUY', '2 HOLD']
classification_r = classification_report(y_test, y_pred_classes, target_names=target_names, output_dict=True)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("F1_score: {}", f1)
print(confusion_m)

ph.get_confusion_matrix_fig(confusion_m, target_names)

# log evaluations metrics
ml_experiment["model_metrics"] = {}
ml_experiment["model_metrics"]["Accuracy"] = str(accuracy)
ml_experiment["model_metrics"]["F1_score"] = str(f1)
ml_experiment["model_metrics"]["confusion_matrix"] = confusion_m
ml_experiment["model_metrics"]["classification_report"] = classification_r


# running backtest
bk_output_1 = bk.do_back_test(df_test, bk.CNN_B_S_H, ml_experiment["backtest_parameters_1"]["thresold"], 
                                ml_experiment["backtest_parameters_1"]["cash"], ml_experiment["backtest_parameters_1"]["commission"], "test_sans_frais")
ml_experiment["backtest_results_1"] = bk_output_1.to_dict()
bk_output_2 = bk.do_back_test(df_test, bk.CNN_B_S_H, ml_experiment["backtest_parameters_2"]["thresold"], 
                                ml_experiment["backtest_parameters_2"]["cash"], ml_experiment["backtest_parameters_2"]["commission"], "test")
ml_experiment["backtest_results_2"] = bk_output_2.to_dict()


# saving experiment with model
mm1.ml_experiment.update(ml_experiment)
mm1.save_experiment(cnn_model, True)
#mm1.save_history(history)


# plot evolution of loss and f1 score with each epoch
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['f1_metric'])
plt.plot(history.history['val_f1_metric'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.5, 1])
plt.legend(['train_loss', 'val_loss', 'f1', 'val_f1'], loc='upper left')
plt.show()