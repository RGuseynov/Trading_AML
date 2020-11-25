import azureml.core
from azureml.core import Environment, Experiment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.train.estimator import Estimator
from azureml.core import Model
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import TrainingOutput

from azureml.train.automl import AutoMLConfig
from azureml.pipeline.steps import AutoMLStep
from azureml.pipeline.core import Pipeline


# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

cluster_name = "aml-cluster"
# Verify that cluster exists
try:
    pipeline_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If not, create it
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2', max_nodes=4)
    pipeline_cluster = ComputeTarget.create(ws, cluster_name, compute_config)

pipeline_cluster.wait_for_completion(show_output=True)

# Get python environment
registered_env = Environment.get(ws, 'xgboost-env')

# Create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()
# Use the compute you created above. 
pipeline_run_config.target = pipeline_cluster
# Assign the environment to the run configuration
pipeline_run_config.environment = registered_env

print ("Run configuration created.")

# Get the training dataset
data_stocks = Dataset.get_by_name(ws, name='Stocks_1H')
#Get bitcoin dataset
data_bitcoin = ws.datasets.get("bitcoin 1H tabular dataset")

datastore = ws.get_default_datastore()
prepped_data_path = PipelineData("stocks_train", datastore).as_dataset()
prepped_validation_data_path = PipelineData("stocks_validation", datastore).as_dataset()

dataprep_step = PythonScriptStep(
    name="dataprep", 
    script_name="aml_experiments/auto_ML_pipeline/dataprep.py", 
    compute_target=pipeline_cluster, 
    runconfig=pipeline_run_config,
    arguments=["--output_path", prepped_data_path, "--output_validation_path", prepped_validation_data_path],
    inputs=[data_stocks.as_named_input('stocks')],
    outputs=[prepped_data_path, prepped_validation_data_path],
    allow_reuse=True,
    source_directory="trading_src"
)

# type(prepped_data_path) == PipelineOutputFileDataset
# type(prepped_data) == PipelineOutputTabularDataset
#prepped_data = prepped_data_path.parse_parquet_files(file_extension=None)
prepped_data = prepped_data_path.parse_delimited_files(file_extension=None)
prepped_validation_data = prepped_validation_data_path.parse_delimited_files(file_extension=None)


metrics_data = PipelineData(name='metrics_data',
                            datastore=datastore,
                            pipeline_output_name='metrics_output',
                            training_output=TrainingOutput(type='Metrics'))

model_data = PipelineData(name='best_model_data',
                          datastore=datastore,
                          pipeline_output_name='model_output',
                          training_output=TrainingOutput(type='Model'))


# Change iterations to a reasonable number (50) to get better accuracy
automl_settings = {
    "iteration_timeout_minutes" : 10,
    "iterations" : 10,
    "experiment_timeout_hours" : 0.25,
    "primary_metric" : 'AUC_weighted'
}

automl_config = AutoMLConfig(task = 'classification',
                             path = '.',
                             debug_log = 'automated_ml_errors.log',
                             compute_target = pipeline_cluster,
                             run_configuration = pipeline_run_config,
                             featurization = 'auto',
                             training_data = prepped_data,
                             validation_data = prepped_validation_data,
                             label_column_name = 'Label',
                             **automl_settings)

train_step = AutoMLStep(name='AutoML_Classification',
    automl_config=automl_config,
    outputs=[metrics_data,model_data],
    enable_default_model_output=False,
    enable_default_metrics_output=False,
    allow_reuse=True)


pipeline = Pipeline(ws, [dataprep_step, train_step])

experiment = Experiment(workspace=ws, name='stocks_BSH_autoML')

print("pipeline submitted")
run = experiment.submit(pipeline)
run.wait_for_completion()






## Create an estimator
#estimator = Estimator(source_directory="trading_src",
#              inputs=[data_stocks.as_named_input('stocks'), data_bitcoin.as_named_input('bitcoin')],
#              compute_target = training_cluster, # Use the compute target created previously
#              environment_definition = registered_env,
#              entry_script='aml_experiments/xgb_BSH_Stocks_v2/xgb_BSH_Stocks_v2_script.py')

## Create an experiment
#experiment = Experiment(workspace = ws, name = 'xgb-BSH-training-stocks-v2')
##experiment = Experiment(workspace = ws, name = 'test-TALIB-env')

## Run the experiment
#run = experiment.submit(config=estimator)

#run.wait_for_completion()

