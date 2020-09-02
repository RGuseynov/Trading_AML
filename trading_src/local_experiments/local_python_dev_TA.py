import glob
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

sys.path.append("trading_src")
import logics.features_engineering as fe


pd.set_option('display.max_rows', 50)
pd.set_option('display.min_rows', 50)
pd.set_option('display.max_columns',10)
pd.set_option('display.width', 150)

path = "data/bitcoin_prepared_data/1H"
all_files = glob.glob(path + "/*.csv")
df = pd.concat((pd.read_csv(f) for f in all_files))

df = df.set_index(df["Timestamp"])
df = df.drop(["Timestamp"], axis=1)
df.index = pd.to_datetime(df.index)

df = df[df.index.year >= 2016]

df_TA = fe.add_custom_TA(df)
print(df_TA.describe())
print(df_TA)




#df_TA_custom["HT_up"] = np.abs(df_TA_custom["HT"])
#df_TA_custom["HT_dawn"] = df["Close"] - (df_TA_custom["HT_up"] - df["Close"])
#df_TA_custom[["HT_up", "HT_dawn", "Close"]].plot(kind="line")
#plt.show()