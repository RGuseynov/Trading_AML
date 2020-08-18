import azureml.core
from azureml.core import Workspace

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies


# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))


whl_url = Environment.add_private_pip_wheel(workspace=ws,file_path = "TA_Lib-0.4.18-cp38-cp38-win_amd64.whl")
myenv = Environment(name="trading-xgboost-env")
pip_packages=['azureml-sdk','pandas','xgboost','scikit-learn','matplotlib','seaborn','Backtesting',]
conda_dep = CondaDependencies.create(pip_packages=pip_packages, python_version="3.8")
conda_dep.add_pip_package(whl_url)
myenv.python.conda_dependencies=conda_dep

print(myenv.name, 'defined.')

# Register the environment
myenv.register(workspace=ws)


envs = Environment.list(workspace=ws)
for env in envs:
    print("Name",env)
    print("packages", envs[env].python.conda_dependencies.serialize_to_string())
#if env.startswith("AzureML"):

## Create a Python environment for the experiment
#xgboost_env = Environment("trading-xgboost-env")
#xgboost_env.python.user_managed_dependencies = False # Let Azure ML manage dependencies
#xgboost_env.docker.enabled = True # Use a docker container

## Create a set of package dependencies (conda or pip as required)
#diabetes_packages = CondaDependencies.create(conda_packages=['scikit-learn','ipykernel','matplotlib', 'pandas'],
#                                          pip_packages=['azureml-sdk','pyarrow'])

## Add the dependencies to the environment
#diabetes_env.python.conda_dependencies = diabetes_packages

#print(diabetes_env.name, 'defined.')


# autre exemple
#myenv = Environment(name="myenv")
#conda_dep = CondaDependencies()
#conda_dep.add_conda_package("scikit-learn")
#conda_dep.add_pip_package("pillow==5.4.1")
#myenv.python.conda_dependencies=conda_dep


# Register the environment
#diabetes_env.register(workspace=ws)


#registered_env = Environment.get(ws, 'diabetes-experiment-env')
