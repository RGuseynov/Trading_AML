import pandas as pd


# control l'affichage des resultats de pandas
pd.set_option('display.max_rows', 50)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_columns',10)
pd.set_option('display.min_columns', 10)
pd.set_option('display.width', 150)

# format d'affichage
pd.options.display.float_format = '{:,.2f}'.format

# info sur le dataframe et les colonnes
print(df.describe())
print(df.info())
print(df.count())
print(df[column].value_counts())

# effacement de l'index qui devient une colonne
df.reset_index(level=0, inplace=True)

# resample le dataframe en utilisant "Timestamp" comme index avec une frequence d'une ligne = 1M
df = df.resample("1T", on="Timestamp").mean()

# nombre de valeurs par annee et plot du resultat
df["Timestamp"].groupby(df["Timestamp"].dt.year).count().plot(kind="bar", title="Values per Year")

# Conte le nombre de lignes dont les valeurs sont toutes NaN
df["group_no"] = df.isnull().all(axis=1).cumsum()

# Permet d'avoir la difference de timestampe entre chaque ligne(avec le temps comme index)
times_gaps = df.iloc[1:].index - df.iloc[:-1].index
# cr�er une nouvelle colonne avec cette difference dont la derniere sera NaT
df.loc[:-1,"Time_gaps"] = times_gaps

# objet de type timeDelta avec pandas
threshold = pd.Timedelta("1H")

# selection date
begin = pd.Timestamp('2017-01-01T12')
end = pd.Timestamp('2018-01-01T12')
print(df[(df.index > begin) & (df.index < end)])

# resample avec fonction d'aggregation
df = df.resample('1H').agg({"Volume_(BTC)": np.sum, 'High': np.max})
# ou
df = df.resample("5T").agg({'Open': "first", 'High': "max", "Low": "min", "Close": "last", "Volume_(BTC)": "sum"})

#remplacement de valeur avec un filtre par valeur min le long des lignes
df_TA.loc[df_TA["Low"] <= 300, "Low"] = df_TA[["Open", "Close"]].min(axis=1)


# connaitre l'usage memorie du dataframe sans le detail des colonnes
print(df.info(memory_usage="deep", verbose=False))