from azureml.core import Run

import pandas as pd 
import numpy as np 
import argparse
import os

sys.path.append("trading_src")
import utils.utils as utils
import helpers.save_plotsHelper as ph
import helpers.ml_models_managementHelper as mm
import helpers.load_dataHelper as dh
import logics.features_engineering as fe
import logics.backtest as bk
import logics.data_labeling as dl



parser = argparse.ArgumentParser()
parser.add_argument('--output_path', dest='output_path', required=True)
parser.add_argument('--output_validation_path', dest='output_validation_path', required=True)
args = parser.parse_args()
  
print("loading data...")

stocks = Run.get_context().input_datasets['stocks']
df_stocks = stocks.to_pandas_dataframe()

df_stocks.columns = ['Symbol', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']

dict_of_stocks_df = {symbol: group_df for symbol, group_df in df_stocks.groupby(['Symbol'])}
for symbol, df_stock in dict_of_stocks_df.items():
    fe.add_custom_TA(df_stock)
    dict_of_stocks_df[symbol] = df_stock.astype('float32', errors='ignore')

print("fa done")

df_stocks = pd.concat(dict_of_stocks_df.values())
dict_of_stocks_df = None

df_stocks = df_stocks.set_index(df_stocks["Timestamp"])
df_stocks = df_stocks.drop(["Timestamp"], axis=1)
df_stocks.index = pd.to_datetime(df_stocks.index)
df_stocks_train = df_stocks[(df_stocks.index.year < 2020) | ((df_stocks.index.year == 2020) & (df_stocks.index.month <= 6))]
df_stocks_test = df_stocks[(df_stocks.index.year == 2020) & (df_stocks.index.month > 6)]

df_stocks = None

window_size = 21
df_stocks_train["Label"] = dl.create_labels(df_stocks_train, window_size)
df_stocks_test["Label"] = dl.create_labels(df_stocks_test, window_size)

print("saving dataset to output")

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

df_stocks_train.to_csv(args.output_path)
df_stocks_test.to_csv(args.output_validation_path)

print(f"Wrote test to {args.output_path} and train to {args.output_path}")

