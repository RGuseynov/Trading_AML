import azureml.core
from azureml.core import Workspace, Datastore, Dataset
from azureml.core import Experiment, RunConfiguration, ScriptRunConfig
from azureml.core import Environment

import pandas as pd

import glob


# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# get dataset
dataset = Dataset.get_by_name(ws, name='bitcoin 1H file dataset')
dataset_input = dataset.as_download()

# Create a Python environment for the experiment
bitcoin_env = Environment("data_labeling_test_env")
bitcoin_env.python.user_managed_dependencies = True # for local env

# create a new RunConfig object
experiment_run_config = RunConfiguration()
experiment_run_config.environment = bitcoin_env

# Create a script config
src = ScriptRunConfig(source_directory="trading_src", 
                      script='aml_experiments/data_labeling/data_labeling_script.py',
                      run_config=experiment_run_config,
                      arguments=[dataset_input]) 

# submit the experiment
experiment = Experiment(workspace = ws, name = 'data_labeling_test_experiment')
run = experiment.submit(config=src)
run.wait_for_completion()
