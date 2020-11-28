import numpy as np
import pandas as pd
import scipy.signal


#Bollinger Bands 
#%B quantifies a security's price relative to the upper and lower Bollinger Band.
def BBANDS(df, w=20, n_deviation=2, all=False) -> pd.DataFrame:  
    #Moving Average
    MA = df["Close"].rolling(window=w).mean()
    #Moving Standard Deviation
    MSD = df["Close"].rolling(window=w).std()
    if all:
        df["BollingerB_W" + str(n)] = b1 = n_deviation * 2 * MSD / MA
        df["BollingerB_MIDDLE_W" + str(w)] = MA 
        df["BollingerB_UPPER_W" + str(w)] = MA + (MSD * n_deviation)
        df["BollingerB_LOWER_W" + str(w)] = MA - (MSD * n_deviation)
    df["Bollinger%B_W" + str(w)] = (df['Close'] - MA + n_deviation * MSD) / (n_deviation* 2 * MSD) 
    return df


#Moving Average  
def MA(df, w=30, column="Close") -> pd.Series:  
    return df[column].rolling(window=w).mean()

#Exponential Moving Average  
def EMA(df, w=30, column="Close") -> pd.Series:  
    return df[column].ewm(span=w, min_periods=w).mean()

#Double Exponential Moving Average
def DEMA(df, w=30, column="Close") -> pd.Series:
    ema = pd.DataFrame()
    ema["ema1"] = EMA(df, w, column)
    ema["ema2"] = EMA(ema, w, "ema1")
    return 2 * ema["ema1"] - ema["ema2"]

#Triple Exponential Moving Average
def TEMA(df, w=30, column="Close") -> pd.Series:
    ema = pd.DataFrame()
    ema["ema1"] = EMA(df, w, column)
    ema["ema2"] = EMA(ema, w, "ema1")
    ema["ema3"] = EMA(ema, w, "ema2")
    return 3 * ema["ema1"] - 3 * ema["ema2"] + ema["ema3"]

# hilbert transform 
#def HT(df, column="Close"):
#    return scipy.signal.hilbert(df[column])

#Kaufman Adaptive Moving Average
def KAMA(df, time_period_er=10, time_period_fast=2, time_period_slow=30):
    #Efficiency Ratio (ER)
    Volatility = np.abs(df["Close"].diff(1)).rolling(window=time_period_er).sum()
    Change = np.abs(df["Close"].diff(time_period_er))
    ER = Change/Volatility

    #Smoothing Constant (SC)
    SC = (ER * (2/(time_period_fast + 1) - (2/(time_period_slow + 1))) + 2/(time_period_slow + 1)).pow(2)

    df["KAMA"] = np.nan
    kama_loc = df.columns.get_loc("KAMA")
    df.iloc[time_period_er, kama_loc] = df.iloc[:time_period_er, df.columns.get_loc("Close")].mean()

    for i in range(time_period_er + 1, len(df)):
        df.iloc[i, kama_loc] = df.iloc[i-1, kama_loc] + SC.iloc[i] * (df.iloc[i,df.columns.get_loc("Close")] - df.iloc[i-1, kama_loc])

    return df["KAMA"]

