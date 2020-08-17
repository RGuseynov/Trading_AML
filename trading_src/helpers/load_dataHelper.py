import glob

import pandas as pd


def load_years(data_folder:str, timeframe_folder:str, years:list) -> pd.DataFrame:
    path = 'data/' + data_folder + "/" + timeframe_folder
    df_list = []
    for year in years:
        file = glob.glob(path + "/*" + str(year) + ".csv")[0]
        df = pd.read_csv(file, index_col="Timestamp", parse_dates=['Timestamp'])
        df_list.append(df)
    df = pd.concat(df_list, axis=0, ignore_index=False)
    return df


if __name__ == "__main__" :
    df = load_years("bitcoin_prepared_data", "30T", [2018])
    print(df)
