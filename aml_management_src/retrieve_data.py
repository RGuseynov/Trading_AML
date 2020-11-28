import azureml.core
from azureml.core import Workspace, Datastore, Dataset


# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# Get the dataset
data_stocks = Dataset.get_by_name(ws, name='Stocks_1H')
data_stocks_df = data_stocks.to_pandas_dataframe()

try:
    data_stocks_df.to_csv("data/SP500_historical_data/SP500_1H.csv")
except Exception as e:
    print(e)
    data_stocks_df.to_csv("SP500_1H.csv")

