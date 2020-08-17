import pandas as pd
import os


# traitement inital donnee brut BTC
def remplissage_nan_initial(df: pd.DataFrame) -> pd.DataFrame:
    #remplissage des minutes manquantes(sans activites) avec les donnees du dernier prix
    df["Close"].fillna(method="ffill", inplace=True)
    df["Open"].fillna(df["Close"], inplace=True)
    df["High"].fillna(df["Close"], inplace=True)
    df["Low"].fillna(df["Close"], inplace=True)
    #df.loc[:,["Open", "High", "Low"]].fillna(df["Close"], inplace=True)
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    return df


# traitement inital donnee brut BTC
def traitement_initial():
    # Utiliser la date comme index, convertir unix time to datetime, garder que les dates apres 2014, ajouter les minutes manquantes
    file_path = "data/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"
    print(os.getcwd())
    df = pd.read_csv(file_path, index_col="Timestamp")
    df.index = pd.to_datetime(df.index, unit="s")
    df = df[["Close", "Open", "High", "Low", "Volume_(BTC)"]]
    # permet d'assurer d'avoir un index bien continu avec un timeframe d'une minute, les minutes manquantes sont remplis avec des nan
    df = df.resample("1T").mean().copy()

    # suite aux observations: division du dataset en 2 car discontinuite de 5 jours entre le 05/01/15 et le 09/01/15
    # et on garde que les valeurs apres 2014 car avant trop de discontinuites et donnees moins pertinentes
    df_1 = df.loc['2014-01-01T00:00':'2015-01-05T09:12',:].copy()
    df_2 = df.loc['2015-01-09T21:05':,:].copy()

    df_1 = remplissage_nan_initial(df_1)
    df_2 = remplissage_nan_initial(df_2)

    # save csv with differents timeframes
    resample_times = ["1T", "5T", "30T", "1H", "4H", "1D"]
    if not os.path.exists("data/bitcoin_prepared_data"):
        os.mkdir("data/bitcoin_prepared_data")

    for time in resample_times:
        base_path = "data/bitcoin_prepared_data/" + time
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        
        df_1 = df_1.resample(time).agg({'Open': "first", 'High': "max", "Low": "min", "Close": "last", "Volume_(BTC)": "sum"})
        df_2 = df_2.resample(time).agg({'Open': "first", 'High': "max", "Low": "min", "Close": "last", "Volume_(BTC)": "sum"})
        
        df_1.to_csv(base_path + "/bitstampUSD_data_2014-01-01_to_2015-01-05.csv")

        # create an unique csv for df_2
        df_2.to_csv(base_path + "/bitstampUSD_data_2015-01-05_to_2020-04-22.csv")
        # or one csv per year
        for year in range(2015, 2021):
            df_2[df_2.index.year == year].to_csv(base_path + "/bitstampUSD_data_" + str(year) + ".csv")



if __name__ == "__main__":
    traitement_initial()