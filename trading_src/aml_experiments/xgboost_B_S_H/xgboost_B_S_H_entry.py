import azureml.core
from azureml.core import Environment, Experiment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.train.estimator import Estimator
from azureml.core import Model
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

cluster_name = "aml-cluster"
# Verify that cluster exists
try:
    training_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If not, create it
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2', max_nodes=4)
    training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)

training_cluster.wait_for_completion(show_output=True)

# Get python environment
registered_env = Environment.get(ws, 'trading-xgboost-ta-3.6-env')

# Get the training dataset
data = ws.datasets.get("bitcoin 1H tabular dataset")

# Create an estimator
estimator = Estimator(source_directory="trading_src",
              inputs=[data.as_named_input('bitcoin')],
              compute_target = cluster_name, # Use the compute target created previously
              environment_definition = registered_env,
              entry_script='aml_experiments/xgboost_B_S_H/xgboost_B_S_H_script.py')

# Create an experiment
experiment = Experiment(workspace = ws, name = 'xgboost-training-test')

# Run the experiment
run = experiment.submit(config=estimator)

run.wait_for_completion()

# Register the model
run.register_model(model_path='outputs/xgboost_model_test.pkl', model_name='xgboost_model_test',
                   tags={'Training context':'Azure ML compute'}, properties={'return no fee': run.get_metrics()['return no fee'], 
                                                                             'return with fee': run.get_metrics()['return with fee']})

# List registered models
for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')
