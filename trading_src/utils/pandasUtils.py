import pandas as pd


# permet de creer une colonne indiquant la difference de temps(time_gap) sur l'index(de type date)
# avec la ligne precedente 
# sert a detecter et mesurer les discontinuitees temporels
def compute_times_gaps(df:pd.DataFrame, new_column_name:string = "Time_gaps") -> pd.DataFrame:   
    # drop missing values for calculating time_gaps beetween non missing values
    df = df.dropna()

    times_gaps = df.iloc[1:].index - df.iloc[:-1].index
    df.loc[1:, new_column_name] = times_gaps
    return df


# filtre un df sur une colonne de type TimeDelta avec une condition
# (valeur superieur a un threshold de 15min ici)
def filter_timeDelta(df:pd.DataFrame, column_name:string = "Time_gaps", threshold:pd.Timedelta = pd.Timedelta("15T")) -> pd.DataFrame:

    return df[df[column_name] > threshold]


# retourne les valeurs comprise entre une date et un TimeDelta
def get_date_since(df:pd.DataFrame, begin:pd.Timestamp, delta:pd.Timedelta) -> pd.DataFrame:
    #begin = pd.Timestamp('2015-01-05T09:12')

    end = begin + delta
    return df[(df.index >= begin) & (df.index <= end)]

