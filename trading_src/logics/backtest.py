from backtesting import Backtest, Strategy

from backtesting.lib import crossover
from backtesting.test import SMA

import xgboost as xgb
import pandas as pd

import pickle


# library exemple
class SmaCross(Strategy):
    n1 = 10
    n2 = 30

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        if (not self.position and crossover(self.sma1, self.sma2)):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()


class BinaryPrediction(Strategy):
    prediction = None
    thresold = None

    def init(self):      
        self.prediction = self.I(lambda x:x, self.prediction)

    def next(self):
        if (not self.position and self.prediction > 0.5 + self.thresold):
            self.buy()
        elif self.prediction < 0.5 - self.thresold:
            self.position.close()


class B_S_H(Strategy):
    prediction = None
    thresold = None

    def init(self):      
        self.prediction = self.I(lambda x:x, self.prediction)

    def next(self):
        if (not self.position and self.prediction == 1):
            self.buy()
        elif self.prediction == 0:
            self.position.close()


def do_back_test(test_data, strategy, thresold=0, cash=10000, commission=0.001, filename=None):
    test_data = test_data[["Open", "High", "Low", "Close", "Volume", "Prediction"]]
    #test_data = test_data.rename(columns={"Volume_(BTC)": "Volume"}) 
    
    bt = Backtest(test_data, strategy, cash=cash, commission=commission)
    output = bt.run(prediction=test_data["Prediction"], thresold=thresold)
    if filename :
        bt.plot(filename=filename, open_browser=False)
    output = output.drop("_strategy")
    output = output.drop("_equity_curve")
    output = output.drop("_trades")
    return output



if __name__ == "__main__":

    #model = pickle.load(open("Models/2019_BINLOG_1MIN_TO_5MIN/iteration_17/model.pkl", 'rb'))
    model = pickle.load(open("Models/ALL_XGBOOST_BINLOG_1MIN_TO_5MIN_Later.pkl", 'rb'))

    df = pd.read_csv("../CommonFiles/TA_EngeeniredData/bitstampUSD_1-min_data_2020.csv", index_col="Timestamp", parse_dates=['Timestamp'])
    #df = df.loc["2020-01-01 00:00:00":"2020-02-01 00:00:00",["Open", "High", "Low", "Close", "Volume_(BTC)"]]
    #df = df.loc["2020-04-01 00:00:00":"2020-05-01 00:00:00"]
    #print(df)
    #df = df.resample("5T").first()
    # conversion du df en matrix xgb pour le model
    dfMatrix = xgb.DMatrix(df)
    df["Prediction"] = model.predict(dfMatrix)
    df = df[["Open", "High", "Low", "Close", "Volume", "Prediction"]]
    #df.rename(columns={"Volume_(BTC)": "Volume"}, inplace=True) 
    print(df)

    bt = Backtest(df, BinaryPrediction, cash=10000, commission=.000)

    output = bt.run(prediction=df["Prediction"], thresold=0.1)
    print(output)
    bt.plot(filename=None, open_browser=False)

    #outp = output.drop("_strategy")
    #outp = outp.to_json()