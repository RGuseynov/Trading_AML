import sys

import pandas as pd
import numpy as np

sys.path.append("trading_src")
import utils.utils as utils
import helpers.ml_models_managementHelper as mm
import helpers.load_dataHelper as dh
import logics.features_engineering as fe
import logics.backtest as bk
import logics.data_labeling as dl


pd.set_option('display.max_rows', 50)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_columns',10)
pd.set_option('display.width', 150)


#path = "data/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"
path = "data/bitcoin_prepared_data/1H/bitstampUSD_data_2020.csv"

df = pd.read_csv(path)

df = df.set_index(df["Timestamp"])
df = df.drop(["Timestamp"], axis=1)

df.index = pd.to_datetime(df.index)

print(df)
print(df.info())

#df["Prediction"] = 2

#bk_params = {"thresold": 0, "cash": 10000, "commission": 0}

#bk_output_1 = bk.do_back_test(df, bk.B_S_H, 0, 
#                                10000, 0, "bk_plot_sans_frais_jour")

df = fe.add_partial_TA(df)

print(df)



example_np = df.iloc[700,:64].to_numpy()

np.set_printoptions(suppress=True)

print(df.iloc[700,:64])

print(np.reshape(example_np, (8, 8)))
