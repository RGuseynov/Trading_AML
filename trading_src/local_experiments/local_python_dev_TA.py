import glob
import sys

import pandas as pd

sys.path.append("trading_src")
import logics.features_engineering as fe


pd.set_option('display.max_rows', 50)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_columns',10)
pd.set_option('display.width', 150)

path = "data/bitcoin_prepared_data/1H"
all_files = glob.glob(path + "/*.csv")
df = pd.concat((pd.read_csv(f) for f in all_files))

df = df.set_index(df["Timestamp"])
df = df.drop(["Timestamp"], axis=1)
df.index = pd.to_datetime(df.index)

df_my_TA = fe.add_my_TA(df)
print(df_my_TA)

df_custom_TA = fe.add_custom_TA(df)
print(df_custom_TA)
