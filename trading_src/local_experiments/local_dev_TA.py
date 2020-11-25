import glob
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

sys.path.append("trading_src")
import logics.features_engineering as fe
import logics.data_labeling as dl


pd.set_option('display.max_rows', 20)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_columns',8)
pd.set_option('display.width', 150)

# format d'affichage
pd.options.display.float_format = '{:,.4f}'.format

path = "data/bitcoin_prepared_data/1H"
all_files = glob.glob(path + "/*.csv")
df = pd.concat((pd.read_csv(f) for f in all_files))

df = df.set_index(df["Timestamp"])
df = df.drop(["Timestamp"], axis=1)
df.index = pd.to_datetime(df.index)
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

df = df[df.index.year >= 2016]

# correction de valeur Low extreme (une valeur a 1.5$, erreur probable ou flash crash)
df.loc[df["Low"] <= 300, "Low"] = df.loc[df["Low"] <= 300, ["Open", "Close"]].min(axis=1).values[0]

#y = dl.create_labels_base(df, 'Close')
#print(y)
#print(len(y))


#fe.add_test_TA(df)
##df = df.drop(["Open", "High", "Low", "Close", "Volume"], axis=1)
#df = df.drop(["Open", "High", "Low", "Volume"], axis=1)
#print(df.describe())
#print(df.tail(20))



#df_TA_custom["HT_up"] = np.abs(df_TA_custom["HT"])
#df_TA_custom["HT_dawn"] = df["Close"] - (df_TA_custom["HT_up"] - df["Close"])
#df_TA_custom[["HT_up", "HT_dawn", "Close"]].plot(kind="line")
#plt.show()