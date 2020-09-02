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



def add_custom_TA (df: pd.DataFrame) -> pd.DataFrame:
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
        
        for j in range(1,30):
            df["BBANDS_WIDTH_"+str(i)+"_"+str(j)] = df["BBANDS_WIDTH_"+str(i)].shift(j) / df["BBANDS_WIDTH_"+str(i)]

        #Moving Average
        MA = df["Close"].rolling(window=i).mean()
        #Moving Standard Deviation
        MSD = df["Close"].rolling(window=i).std()
        df["BBANDS_%B_" + str(i)] = (df['Close'] - MA + 2 * MSD) / (2 * 2 * MSD)

        for j in range(1,30):
            df["BBANDS_%B_"+str(i)+"_"+str(j)] = df["BBANDS_%B_"+str(i)] - df["BBANDS_%B_"+str(i)].shift(j)


    return df



