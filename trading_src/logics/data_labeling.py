import pandas as pd
import numpy as np


def add_B_S_H(row):
    if row["Close"] == row["temp_min"]:
        return 1
    elif row["Close"] == row["temp_max"]:
        return 0
    else:
        return 2


def create_labels(df, window_size=11):
    df["temp_max"] = df["Close"].rolling(window_size, center=True).max()
    df["temp_min"] = df["Close"].rolling(window_size, center=True).min()
    df["temp_y"] = df[["Close", "temp_max", "temp_min"]].apply(add_B_S_H, axis=1)
    y = df["temp_y"].copy()
    df.drop(['temp_max', 'temp_min', 'temp_y'], axis=1, inplace=True)
    return y


# copied
def create_labels_base(df, col_name, window_size=11):
        """
        Data is labeled as per the logic in research paper
        Label code : BUY => 1, SELL => 0, HOLD => 2
        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy
        returns : numpy array with integer codes for labels with
                  size = total-(window_size)+1
        """

        #self.log("creating label with original paper strategy")
        row_counter = 0
        total_rows = len(df)
        labels = np.zeros(total_rows)
        labels[:] = np.nan
        print("Calculating labels")

        while row_counter < total_rows:
            if row_counter >= window_size - 1:
                window_begin = row_counter - (window_size - 1)
                window_end = row_counter
                window_middle = int((window_begin + window_end) / 2)

                min_ = np.inf
                min_index = -1
                max_ = -np.inf
                max_index = -1
                for i in range(window_begin, window_end + 1):
                    price = df.iloc[i][col_name]
                    if price < min_:
                        min_ = price
                        min_index = i
                    if price > max_:
                        max_ = price
                        max_index = i

                if max_index == window_middle:
                    labels[window_middle] = 0
                elif min_index == window_middle:
                    labels[window_middle] = 1
                else:
                    labels[window_middle] = 2

            row_counter = row_counter + 1

        return labels