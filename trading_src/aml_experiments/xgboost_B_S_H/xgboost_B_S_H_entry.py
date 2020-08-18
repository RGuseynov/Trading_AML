from azureml.core import Environment, Experiment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.train.estimator import Estimator
from azureml.widgets import RunDetails

# Create a Python environment for the experiment
diabetes_env = Environment("diabetes-experiment-env")
diabetes_env.python.user_managed_dependencies = False # Let Azure ML manage dependencies
diabetes_env.docker.enabled = True # Use a docker container

# Create a set of package dependencies
diabetes_packages = CondaDependencies.create(conda_packages=['scikit-learn','ipykernel','matplotlib', 'pandas'],
                                             pip_packages=['azureml-sdk','pyarrow'])

# Add the dependencies to the environment
diabetes_env.python.conda_dependencies = diabetes_packages

# Register the environment (just in case previous lab wasn't completed)
diabetes_env.register(workspace=ws)
registered_env = Environment.get(ws, 'diabetes-experiment-env')

# Get the training dataset
diabetes_ds = ws.datasets.get("diabetes dataset")

# Create an estimator
estimator = Estimator(source_directory=experiment_folder,
              inputs=[diabetes_ds.as_named_input('diabetes')],
              compute_target = cluster_name, # Use the compute target created previously
              environment_definition = registered_env,
              entry_script='diabetes_training.py')

# Create an experiment
experiment = Experiment(workspace = ws, name = 'diabetes-training')

# Run the experiment
run = experiment.submit(config=estimator)
# Show the run details while running
RunDetails(run).show()


from azureml.core import Model

# Register the model
run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',
                   tags={'Training context':'Azure ML compute'}, properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']})

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
