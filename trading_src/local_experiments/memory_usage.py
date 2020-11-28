import glob
import sys
import tracemalloc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

sys.path.append("trading_src")
import logics.features_engineering as fe


tracemalloc.start()

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

df = pd.read_csv("data/SP500_historical_data/SP500_1H.csv", index_col=0)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

df = df.rename(columns={"Open_price": "Open", "High_price": "High", "Low_price": "Low", "Close_price": "Close"})

df = fe.add_TA(df)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

print(df)
print(df.info(memory_usage="deep", verbose=False))

tracemalloc.stop()
