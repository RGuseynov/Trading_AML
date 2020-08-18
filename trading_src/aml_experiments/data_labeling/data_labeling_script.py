from azureml.core import Run
import pandas as pd
import os
import sys
import glob

from azureml.core import Workspace, Datastore, Dataset
from azureml.core import Experiment, RunConfiguration, ScriptRunConfig

sys.path.append("trading_src")
import logics.data_labeling as dl


# Get the experiment run context
run = Run.get_context()

print("loading data")

# load the diabetes dataset
#data = pd.read_csv('diabetes.csv')
#download_location = run.input_datasets['input_1']

download_location = sys.argv[1]

print(download_location)

all_files = glob.glob(download_location + "/*.csv")
df = pd.concat((pd.read_csv(f) for f in all_files))

print(df)


window_size = 11
df["y"] = dl.create_labels(df, window_size)

print(df)
run.log("window_size", window_size)


# Count the rows and log the result
row_count = (len(df))
run.log('observations', row_count)
print('Analyzing {} rows of data'.format(row_count))
      
# Save a sample of the data in the outputs folder (which gets uploaded automatically)
os.makedirs('outputs', exist_ok=True)
df.sample(100).to_csv("outputs/sample.csv", index=False, header=True)


os.makedirs('data', exist_ok=True)
local_path = 'data/labeled.csv'
df.to_csv(local_path)

# get the experiment and workspace from within the script run context
exp = run.experiment
ws = exp.workspace

# get the datastore to upload prepared data
datastore = Datastore.get(ws, 'financial_timeseries_ohlc')

# upload the local file from src_dir to the target_path in datastore
datastore.upload(src_dir='data', target_path='data_test')

# create a dataset referencing the cloud location
dataset = Dataset.Tabular.from_delimited_files(datastore.path('data_test/labeled.csv'))

tab_data_set = dataset.register(workspace=ws, 
                           name='labeled',
                           description='labeled test data',
                           tags = {'format':'CSV'},
                           create_new_version=True)


# Complete the run
run.complete()

