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


def add_TA (df: pd.DataFrame):
    inputs = {
    'open': df["Open"],
    'high': df["High"],
    'low': df["Low"],
    'close': df["Close"],
    'volume': df["Volume"]
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


def add_partial_TA (df: pd.DataFrame) -> None:
    inputs = {
    'open': df["Open"],
    'high': df["High"],
    'low': df["Low"],
    'close': df["Close"],
    'volume': df["Volume"]
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



def add_test_TA (df: pd.DataFrame):
    inputs = {
    'open': df["Open"],
    'high': df["High"],
    'low': df["Low"],
    'close': df["Close"],
    'volume': df["Volume"]
    }
    close = df["Close"]

    #df["BBANDS_upper"], df["BBANDS_middle"], df["BBANDS_lower"] = BBANDS(inputs, 20, 2.0, 2.0)

    df["MA"] = SMA(close) 
    df["normalized_MA"] = SMA(close) / close
    df["RSI"] =  df["RSI"] = RSI(inputs)




def add_BBands_TA (df: pd.DataFrame) -> pd.DataFrame:
    inputs = {
    'open': df["Open"],
    'high': df["High"],
    'low': df["Low"],
    'close': df["Close"],
    'volume': df["Volume"]
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


def add_custom_TA (df: pd.DataFrame) -> None:
    inputs = {
    'open': df["Open"],
    'high': df["High"],
    'low': df["Low"],
    'close': df["Close"],
    'volume': df["Volume"]
    }
    close = df["Close"]

    #customise timeperiod for indicators

    #default 30 
    t1_b = 10
    t1_e = 201
    t1_s = 5

    #default 14 
    t2_b = 6
    t2_e = 31
    t2_s = 2

    #default 10 
    t3_b = 6
    t3_e = 21
    t3_s = 1

    #default 5 
    t4_b = 3
    t4_e = 13
    t4_s = 1

    #defaut 12
    t_fast_b = 4
    t_fast_e = 21
    t_fast_s = 2
    #default 26
    t_slow_b = 20
    t_slow_e = 37
    t_slow_s = 2

    # OVERLAP STUDIES
    # Boollinger Bands default period=5
    for i in range(3, 52, 2):
        upper_band, medium_band, lower_band = BBANDS(close, timeperiod=i, nbdevup=2.0, nbdevdn=2.0)
        df["BBANDS_WIDTH_"+str(i)] = (upper_band - lower_band) / medium_band  
        df["BBANDS_%B_"+ str(i)] = (df['Close'] - lower_band) / (upper_band - lower_band)
        df["BBANDS_%B_"+ str(i)] =  df["BBANDS_%B_"+ str(i)].replace(-np.inf, -1.5)
        df["BBANDS_%B_"+ str(i)] =  df["BBANDS_%B_"+ str(i)].replace(np.inf, 1.5)

        ##Moving Average
        #MA = df["Close"].rolling(window=i).mean()
        ##Moving Standard Deviation
        #MSD = df["Close"].rolling(window=i).std()
        #df["BBANDS_%B_" + str(i)] = (df['Close'] - MA + 2 * MSD) / (2 * 2 * MSD)

    # Simple Moving average default period=30  up to 200 ?
    for i in range(t1_b, t1_e, t1_s):
        df["SMA_"+str(i)] = SMA(close, timeperiod=i, matype=0) / close
    # Weighted Moving Average default timeperiod=30
    for i in range(t1_b, t1_e, t1_s):
        df["WMA_"+str(i)] = WMA(close, timeperiod=i) / close
    # Exponential Moving Average default period=30
    for i in range(t1_b, t1_e, t1_s):
        df["EMA_"+str(i)] = EMA(close, timeperiod=i) / close
    # Double Exponential Moving Average default period=30
    for i in range(t1_b, t1_e, t1_s):
        df["DEMA_"+str(i)] = DEMA(close, timeperiod=i) / close
    # Triple Exponential Moving Average default period=30
    for i in range(t1_b, t1_e, t1_s):
        df["TEMA_"+str(i)] = TEMA(close, timeperiod=i) / close
    # Triple Exponential Moving Average (T3) default period=5, vfactor=0
    #for i in range(t4_b, t4_e):
    #    for j in np.arange(0.05, 1.0, 0.05):
    #        df["T3_"+str(i)+"_"+"{:2.2f}".format(j)] = T3(close, timeperiod=i, vfactor=j) / close
    # Triangular Moving Average default period=30
    for i in range(t1_b, t1_e, t1_s):
        df["TRIMA_"+str(i)] = TRIMA(close, timeperiod=i) / close
    # Kaufman Adaptive Moving Average default period=30
    for i in range(t1_b, t1_e, t1_s):
        df["KAMA_"+str(i)] = KAMA(close, timeperiod=i) / close

    ## Adaptive Moving Average default fastlimit=0, slowlimit=0  fast > slow
    #for fast in np.arange(0.05, 1.0, 0.05):
    #    for slow in np.arange(0.05, 1.0, 0.05):
    #        mama, fama = MAMA(close, fastlimit=fast, slowlimit=slow)
    #        df["MAMA_"+"{:2.2f}".format(fast)+"_"+"{:2.2f}".format(slow)] = mama / close
    #        df["FAMA_"+"{:2.2f}".format(fast)+"_"+"{:2.2f}".format(slow)] = fama / close

    ## Moving average with variable period default minperiod=2, maxperiod=30, matype=0
    #real = MAVP(close, periods, minperiod=2, maxperiod=30, matype=0)

    # MidPoint over period default timeperiod=14
    for i in range(t2_b, t2_e, t2_s):
        df["MIDPOINT_"+str(i)] = MIDPOINT(close, timeperiod=i) / close
    # Midpoint Price over period default timeperiod=14
    for i in range(t2_b, t2_e, t2_s):
        df["MIDPRICE_"+str(i)] = MIDPRICE(inputs, timeperiod=i) / close

    # Parabolic SAR default acceleration=0.02, maximum=0.2
    df["SAR"] = SAR(inputs, acceleration=0.02, maximum=0.2)

    ##Parabolic SAR - Extended default startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0
    #real = SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)

    # Hilbert Transform - Instantaneous Trendline
    df["HT_TRENDLINE"] = HT_TRENDLINE(close) / close


    #MOMENTUM INDICATORS
    # Average Directional Movement Index default timeperiod=14
    for i in range(t2_b, t2_e, t2_s):
        df["ADX_"+str(i)] = ADX(inputs, timeperiod=i)
    # Average Directional Movement Index Rating default timeperiod=14
    for i in range(t2_b, t2_e, t2_s):
        df["ADXR_"+str(i)] = ADXR(inputs, timeperiod=i)

    #Absolute Price Oscillator default fastperiod=12, slowperiod=26, matype=0  slow_period > fast_period
    for fast in range(t_fast_b,t_fast_e, t_fast_s):
        for slow in range(t_slow_b,t_slow_e, t_slow_s):
            df["APO_"+str(fast)+"_"+str(slow)] = APO(close, fastperiod=fast, slowperiod=slow, matype=0)

    # Aroon default timeperiod=14 
    for i in range(t2_b, t2_e, t2_s):
        aroondown, aroonup = AROON(inputs, timeperiod=i)
        df["AROON_DOWN_"+str(i)] = aroondown
        df["AROON_UP_"+str(i)] = aroonup
    # Aroon Oscillator timeperiod=14 difference between aroon up - aroon down 
    for i in range(t2_b, t2_e, t2_s):
        df["AROONOSC_"+str(i)] = AROONOSC(inputs, timeperiod=i)

    # Balance Of Power
    df["BOP"] = BOP(inputs)

    #Commodity Channel Index default timeperiod=14
    for i in range(t2_b, t2_e, t2_s):
        df["CCI_"+str(i)] = CCI(inputs, timeperiod=i)

    #Chande Momentum Oscillator timeperiod=14
    for i in range(t2_b, t2_e, t2_s):
        df["CMO_"+str(i)] = CMO(close, timeperiod=i)

    #Directional Movement Index timeperiod=14
    for i in range(t2_b, t2_e, t2_s):
        df["DX_"+str(i)] = DX(inputs, timeperiod=i)

    #Moving Average Convergence/Divergence fastperiod=12, slowperiod=26, signalperiod=9 slow_period > fast_period
    for fast in range(t_fast_b, t_fast_e, t_fast_s):
        for slow in range(t_slow_b, t_slow_e, t_slow_s):
            for signal in range(9,10):
                macd, macdsignal, macdhist = MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
                df["MACD_"+str(fast)+"_"+str(slow)+"_"+str(signal)] = macd
                df["MACD_SIGNAL_"+str(fast)+"_"+str(slow)+"_"+str(signal)] = macdsignal
                df["MACD_HIST_"+str(fast)+"_"+str(slow)+"_"+str(signal)] = macdhist

    #MACD with controllable MA type fastperiod=12, slowperiod=26, signalperiod=9 slow_period > fast_period
    for fast in range(t_fast_b, t_fast_e, t_fast_s):
        for slow in range(t_slow_b, t_slow_e, t_slow_s):
            for signal in range(9,10):
                macd, macdsignal, macdhist = MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
                df["MACD_"+str(fast)+"_"+str(slow)+"_"+str(signal)] = macd
                df["MACD_SIGNAL_"+str(fast)+"_"+str(slow)+"_"+str(signal)] = macdsignal
                df["MACD_HIST_"+str(fast)+"_"+str(slow)+"_"+str(signal)] = macdhist

    #Money Flow Index timeperiod=14
    for i in range(t2_b, t2_e, t2_s):
        df["MFI_"+str(i)] = MFI(inputs, timeperiod=i)

    #Minus Directional Indicator timeperiod=14
    for i in range(t2_b, t2_e, t2_s):
        df["MINUS_DI_"+str(i)] = MINUS_DI(inputs, timeperiod=i)

    #Plus Directional Indicator timeperiod=14
    for i in range(t2_b, t2_e, t2_s):
        df["PLUS_DI_"+str(i)] = PLUS_DI(inputs, timeperiod=i)

    #Minus Directional Movement timeperiod=14  (need maybe some normalisation if NN)
    for i in range(t2_b, t2_e, t2_s):
        df["MINUS_DM_"+str(i)] = MINUS_DM(inputs, timeperiod=i)

    #Minus Directional Movement timeperiod=14  (need maybe some normalisation if NN)
    for i in range(t2_b, t2_e, t2_s):
        df["PLUS_DM_"+str(i)] = PLUS_DM(inputs, timeperiod=i)

    #Momentum timeperiod=10
    for i in range(t3_b, t3_e, t3_s):
        df["MOM_"+str(i)] = MOM(close, timeperiod=i)

    #Percentage Price Oscillator fastperiod=12, slowperiod=26 slow_period > fast_period
    for fast in range(t_fast_b, t_fast_e, t_fast_s):
        for slow in range(t_slow_b, t_slow_e, t_slow_s):
            df["PPO_"+str(fast)+"_"+str(slow)] = PPO(close, fastperiod=fast, slowperiod=slow, matype=0)

    #Rate of change : ((price/prevPrice)-1)*100  timeperiod=10
    for i in range(t3_b, t3_e, t3_s):
        df["ROC_"+str(i)] = ROC(close, timeperiod=i)

    #Rate of change Percentage: (price-prevPrice)/prevPrice  timeperiod=10
    for i in range(t3_b, t3_e, t3_s):
        df["ROCP_"+str(i)] = ROCP(close, timeperiod=i)

    #Rate of change ratio: (price/prevPrice)  timeperiod=10
    for i in range(t3_b, t3_e, t3_s):
        df["ROCR_"+str(i)] = ROCR(close, timeperiod=i)

    #Rate of change ratio 100 scale: (price/prevPrice)*100  timeperiod=10
    for i in range(t3_b, t3_e, t3_s):
        df["ROCR100_"+str(i)] = ROCR100(close, timeperiod=i)

    #Relative Strength Index timeperiod=14
    for i in range(t2_b, t2_e, t2_s):
        df["RSI_"+str(i)] = RSI(inputs, timeperiod=i)

    #Stochastic fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
    for fastk_p in range(4, 7):
        for slowk_p in range(2, 5):
            for slowd_p in range(2, 5):
                lowk, slowd = STOCH(inputs, fastk_period=fastk_p, slowk_period=slowk_p, slowk_matype=0, slowd_period=slowd_p, slowd_matype=0)
                df["STOCH_LOWK_"+str(fastk_p)+"_"+str(slowk_p)+"_"+str(slowd_p)] = lowk
                df["STOCH_SLOWD_"+str(fastk_p)+"_"+str(slowk_p)+"_"+str(slowd_p)] = slowd

    #Stochastic Fast fastk_period=5, fastd_period=3, fastd_matype=0
    for fastk_p in range(4, 8):
        for fastd_p in range(2, 5):
            fastk, fastd = STOCHF(inputs, fastk_period=fastk_p, fastd_period=fastd_p, fastd_matype=0)
            df["STOCHF_FASTK_"+str(fastk_p)+"_"+str(slowk_p)] = fastk
            df["STOCHF_FASTD_"+str(fastk_p)+"_"+str(slowk_p)] = fastd

    #Stochastic Relative Strength Index timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
    for timeperiod in range(t2_b, t2_e, t2_s):
        for fastk_p in range(t4_b, t4_e, t4_s):
            for fastd_p in range(2, 5):
                fastk, fastd = STOCHRSI(close, timeperiod=timeperiod, fastk_period=fastk_p, fastd_period=fastd_p, fastd_matype=0)
                df["STOCHRSI_FASTK_"+str(timeperiod)+"_"+str(fastk_p)+"_"+str(fastd_p)] = fastk
                df["STOCHRSI_FASTD_"+str(timeperiod)+"_"+str(fastk_p)+"_"+str(fastd_p)] = fastd

    #1-day Rate-Of-Change (ROC) of a Triple Smooth EMA timeperiod=30
    for i in range(t1_b, t1_e, t1_s):
        df["TRIX_"+str(i)] = TRIX(close, timeperiod=i)

    #Ultimate Oscillator timeperiod1=7, timeperiod2=14, timeperiod3=28
    for i1 in range(3, 12, 2):
        for i2 in range(10, 21, 2):
            for i3 in range(20, 41, 2):
                df["ULTOSC_"+str(i1)+"_"+str(i2)+"_"+str(i3)] = ULTOSC(inputs, timeperiod1=i1, timeperiod2=i2, timeperiod3=i3)

    #Williams' %R timeperiod=14
    for i in range(t2_b, t2_e, t2_s):
        df["WILLR_"+str(i)] = WILLR(inputs, timeperiod=i)


    ##VOLUME INDICATORS
    #AD - Chaikin A/D Line
    df["AD"] = AD(inputs)

    #ADOSC - Chaikin A/D Oscillator fastperiod=3, slowperiod=10
    for fast in range(2,7):
        for slow in range(6,21):
            df["ADOSC_"+str(fast)+"_"+str(slow)] = ADOSC(inputs, fast, slow)

    #OBV - On Balance Volume
    df["OBV"] = OBV(inputs)


    ##VOLATILITY INDICATORS
    #ATR - Average True Range timeperiod=14
    #for i in range(t2_b, t2_e):
    #    df["ATR_"+str(i)] = ATR(inputs, timeperiod=i)

    #better normalised indicator than non
    ##NATR - Normalized Average True Range
    for i in range (t2_b, t2_e, t2_s):
        df["NATR_"+str(i)] = NATR(inputs, timeperiod=i)

    ##TRANGE - True Range
    #df["TRANGE"] = TRANGE(inputs)


    ##PATTERN RECOGNITION
    #for func in talib.get_function_groups()['Pattern Recognition']:
    #    df[func] = globals()[func](inputs)


