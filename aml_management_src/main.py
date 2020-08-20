import azureml.core
from azureml.core import Workspace, Datastore, Dataset


# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# Retrieve key from keyvault to create datastore (connection)
#keyvault = ws.get_default_keyvault()
#retrieved_secret = keyvault.get_secret(name="tradingaml-storage")


#blob_datastore = Datastore.register_azure_blob_container(workspace = ws,
#					datastore_name = "financial_timeseries_ohlc",					
#					container_name = "financial-timeseries-ohlc",
#					account_name = "tradingaml",
#					account_key = retrieved_secret)
# datastore_name : nom à donner au magasin de données sur AML
# container_name : nom du container Azure
# account_name : nom du compte de stockage Azure
# account_key : clé du compte de stockage Azure


blob_datastore = Datastore.get(ws, 'financial_timeseries_ohlc')


##Create a file dataset from the path on the datastore (this may take a short while)
#file_data_set = Dataset.File.from_files(path=(blob_datastore, 'bitcoin_clean_ohlcv/1H/*.csv'))

## Get the files in the dataset
#for file_path in file_data_set.to_path():
#    print(file_path)

## Register the file dataset
#file_data_set = file_data_set.register(workspace=ws, 
#                           name='bitcoin 1H file dataset',
#                           description='bitcoin 1H files',
#                           tags = {'format':'CSV'},
#                           create_new_version=True)

#print('Datasets registered')



tabular_data_set = Dataset.Tabular.from_delimited_files(path=(blob_datastore, 'bitcoin_clean_ohlcv/1H/*.csv'))

tabular_data_set = tabular_data_set.register(workspace=ws, 
                           name='bitcoin 1H tabular dataset',
                           description='bitcoin 1H tabular',
                           tags = {'format':'CSV'},
                           create_new_version=True)

print('Datasets registered')


## Get specific datastore
#aml_datastore = Datastore.get(ws, 'nom_dun_datastore_existant')
## Set the default datastore
#ws.set_default_datastore('nom_dun_datastore_existant')
## Get the default datastore
#default_ds = ws.get_default_datastore()
