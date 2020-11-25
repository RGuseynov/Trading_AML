#import glob
#import sys

#import numpy as np
#import pandas as pd

#import matplotlib.pyplot as plt

#sys.path.append("trading_src")
#import logics.features_engineering as fe


#df = pd.read_csv("data/SP500_historical_data/SP500_1H.csv", index_col=0)

#df = df.rename(columns={"Open_price": "Open", "High_price": "High", "Low_price": "Low", "Close_price": "Close"})

#df = fe.add_TA(df)

#print(df)
#print(df.info(memory_usage="deep", verbose=False))


import tracemalloc

tracemalloc.start()
print("ok")
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.stop()