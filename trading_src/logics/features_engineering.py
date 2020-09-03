import pandas as pd
import numpy as np
import talib
from talib.abstract import *


def process_ta_functions_group (df: pd.DataFrame, inputs: dict, ta_functions: list) -> pd.DataFrame:
    for func in ta_functions:
        output = globals()[func](inputs)
        # cas ou la fonction retourne une seule liste de outputs
        if len(output) == len(inputs["open"]):
            df[func] = output
        # cas ou la fonction retourne une liste de liste de outputs(ex:bande haute,bande moyenne, bande basse)
        else:
            i = 0
            for out in output:
                df[func+"_"+str(i)] = out
                i += 1
    return df


def add_TA (df: pd.DataFrame) -> pd.DataFrame:
    inputs = {
    'open': df["Open"],
    'high': df["High"],
    'low': df["Low"],
    'close': df["Close"],
    'volume': df["Volume_(BTC)"]
    }

    overlap_studies = talib.get_function_groups()['Overlap Studies']
    if "MAVP" in overlap_studies:
        overlap_studies.remove("MAVP")
    df = process_ta_functions_group(df, inputs, overlap_studies)
    df = process_ta_functions_group(df, inputs, talib.get_function_groups()['Momentum Indicators'])
    df = process_ta_functions_group(df, inputs, talib.get_function_groups()['Cycle Indicators'])

    for func in talib.get_function_groups()['Volume Indicators']:
        df[func] = globals()[func](inputs)

    for func in talib.get_function_groups()['Volatility Indicators']:
        df[func] = globals()[func](inputs)

    for func in talib.get_function_groups()['Pattern Recognition']:
        df[func] = globals()[func](inputs)

    return df


def add_partial_TA (df: pd.DataFrame) -> pd.DataFrame:
    inputs = {
    'open': df["Open"],
    'high': df["High"],
    'low': df["Low"],
    'close': df["Close"],
    'volume': df["Volume_(BTC)"]
    }

    overlap_studies = talib.get_function_groups()['Overlap Studies']
    if "MAVP" in overlap_studies:
        overlap_studies.remove("MAVP")
    df = process_ta_functions_group(df, inputs, overlap_studies)
    df = process_ta_functions_group(df, inputs, talib.get_function_groups()['Momentum Indicators'])
    df = process_ta_functions_group(df, inputs, talib.get_function_groups()['Cycle Indicators'])

    for func in talib.get_function_groups()['Volume Indicators']:
        df[func] = globals()[func](inputs)

    for func in talib.get_function_groups()['Volatility Indicators']:
        df[func] = globals()[func](inputs)

    #for func in talib.get_function_groups()['Pattern Recognition']:
    #    df[func] = globals()[func](inputs)

    return df


def add_test_TA (df: pd.DataFrame) -> pd.DataFrame:
    inputs = {
    'open': df["Open"],
    'high': df["High"],
    'low': df["Low"],
    'close': df["Close"],
    'volume': df["Volume_(BTC)"]
    }

    #df["BBANDS_upper"], df["BBANDS_middle"], df["BBANDS_lower"] = BBANDS(inputs, 20, 2.0, 2.0)

    df["KAMA"] = KAMA(inputs, timeperiod=30)
    df["MESA"] = MESA(inputs)

    return df



def add_BBands_TA (df: pd.DataFrame) -> pd.DataFrame:
    inputs = {
    'open': df["Open"],
    'high': df["High"],
    'low': df["Low"],
    'close': df["Close"],
    'volume': df["Volume_(BTC)"]
    }

    for i in range(5, 30):
        #df["BBANDS_UP"+str(i)], df["BBANDS_MID"+str(i)], df["BBANDS_LOW"+str(i)] = BBANDS(inputs, timeperiod=i, nbdevup=2.0, nbdevdn=2.0)
        #df["BBANDS_UP"+str(i)] = df["BBANDS_UP"+str(i)] / df["Close"]
        #df["BBANDS_LOW"+str(i)] = df["BBANDS_LOW"+str(i)] / df["Close"]

        upper_band, medium_band, lower_band = BBANDS(inputs, timeperiod=i, nbdevup=2.0, nbdevdn=2.0)
        df["BBANDS_WIDTH_"+str(i)] = (upper_band - lower_band) / medium_band
        
        #for j in range(1,31):
        #    df["BBANDS_WIDTH_"+str(i)+"_"+str(j)] = df["BBANDS_WIDTH_"+str(i)].shift(j) / df["BBANDS_WIDTH_"+str(i)]

        #Moving Average
        MA = df["Close"].rolling(window=i).mean()
        #Moving Standard Deviation
        MSD = df["Close"].rolling(window=i).std()
        df["BBANDS_%B_" + str(i)] = (df['Close'] - MA + 2 * MSD) / (2 * 2 * MSD)

        #for j in range(1,31):
        #    df["BBANDS_%B_"+str(i)+"_"+str(j)] = df["BBANDS_%B_"+str(i)] - df["BBANDS_%B_"+str(i)].shift(j)


    return df


def add_custom_TA (df: pd.DataFrame) -> pd.DataFrame:
    inputs = {
    'open': df["Open"],
    'high': df["High"],
    'low': df["Low"],
    'close': df["Close"],
    'volume': df["Volume_(BTC)"]
    }
    close = df["Close"]

    # OVERLAP STUDIES
    # Boollinger Bands default period=5
    #for i in range(2, 201):
    #    upper_band, medium_band, lower_band = BBANDS(close, timeperiod=i, nbdevup=2.0, nbdevdn=2.0)
    #    df["BBANDS_WIDTH_"+str(i)] = (upper_band - lower_band) / medium_band    
    #    #Moving Average
    #    MA = df["Close"].rolling(window=i).mean()
    #    #Moving Standard Deviation
    #    MSD = df["Close"].rolling(window=i).std()
    #    df["BBANDS_%B_" + str(i)] = (df['Close'] - MA + 2 * MSD) / (2 * 2 * MSD)

    # Simple Moving average default period=30  up to 200 ?
    for i in range(2, 201):
        df["SMA_"+str(i)] = SMA(close, timeperiod=i, matype=0)
    # Weighted Moving Average default timeperiod=30
    for i in range(2, 201):
        df["WMA_"+str(i)] = WMA(close, timeperiod=i)
    # Exponential Moving Average default period=30
    for i in range(2, 201):
        df["EMA_"+str(i)] = EMA(close, timeperiod=i)
    # Double Exponential Moving Average default period=30
    for i in range(2, 201):
        df["DEMA_"+str(i)] = DEMA(close, timeperiod=i)
    # Triple Exponential Moving Average default period=30
    for i in range(2, 201):
        df["TEMA_"+str(i)] = TEMA(close, timeperiod=i)
    # Triple Exponential Moving Average (T3) default period=5, vfactor=0
    for i in range(2, 201):
        df["T3_"+str(i)] = T3(close, timeperiod=i, vfactor=0.0)
    # Triangular Moving Average default period=30
    for i in range(2, 201):
        df["TRIMA_"+str(i)] = TRIMA(close, timeperiod=i)
    # Kaufman Adaptive Moving Average default period=30
    for i in range(2, 201):
        df["KAMA_"+str(i)] = KAMA(close, timeperiod=i)

    # Adaptive Moving Average default fastlimit=0, slowlimit=0  fast > slow
    for fast in np.arange(0.05, 1.0, 0.05):
        for slow in np.arange(0.05, 1.0, 0.05):
            mama, fama = MAMA(close, fastlimit=fast, slowlimit=slow)
            df["MAMA_"+"{:2.2f}".format(fast)+"_"+"{:2.2f}".format(slow)] = mama
            df["FAMA_"+"{:2.2f}".format(fast)+"_"+"{:2.2f}".format(slow)] = fama

    # Moving average with variable period default minperiod=2, maxperiod=30, matype=0
    # real = MAVP(close, periods, minperiod=2, maxperiod=30, matype=0)

    # MidPoint over period default timeperiod=14
    for i in range(2, 201):
        df["MIDPOINT_"+str(i)] = MIDPOINT(close, timeperiod=i)
    # Midpoint Price over period default timeperiod=14
    for i in range(2, 201):
        df["MIDPRICE_"+str(i)] = MIDPRICE(inputs, timeperiod=i)

    # Parabolic SAR default acceleration=0, maximum=0
    for acc in range(0, 51):
        for max in range(0, 51):
            df["SAR_"+str(acc)+"_"+str(max)] = SAR(inputs, acceleration=float(acc), maximum=float(max))

    # Parabolic SAR - Extended default startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0
    # real = SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)

    # Hilbert Transform - Instantaneous Trendline
    df["HT_TRENDLINE"] = HT_TRENDLINE(close)

    return df

