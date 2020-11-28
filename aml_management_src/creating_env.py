import azureml.core
from azureml.core import Workspace

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies


# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

myenv = Environment(name="xgboost-env")
pip_packages=['azureml-sdk','pandas','xgboost','scikit-learn','matplotlib','seaborn','Backtesting',
              'talib-binary','numpy','graphviz']
conda_dep = CondaDependencies.create(pip_packages=pip_packages, python_version="3.6")

## not necessary with talib-binary
##whl_url = Environment.add_private_pip_wheel(workspace=ws, file_path="TA_Lib-0.4.18-cp38-cp38-win_amd64.whl", exist_ok= True)
#whl_url = "https://tradingws4129079722.blob.core.windows.net/azureml/Environment/azureml-private-packages/TA_Lib-0.4.10-cp36-cp36m-manylinux1_x86_64.whl"
##conda_dep.add_pip_package(whl_url)

myenv.python.conda_dependencies=conda_dep
print(myenv.name, 'defined.')

# Register the environment
myenv.register(workspace=ws)


# list registered environments
envs = Environment.list(workspace=ws)
for env in envs:
    if not env.startswith("AzureML"):
        print("Name",env)
        print("packages", envs[env].python.conda_dependencies.serialize_to_string())


# exemple
#myenv = Environment(name="myenv")
#conda_dep = CondaDependencies()
#conda_dep.add_conda_package("scikit-learn")
#conda_dep.add_pip_package("pillow==5.4.1")
#myenv.python.conda_dependencies=conda_dep

# Register the environment
#diabetes_env.register(workspace=ws)

#registered_env = Environment.get(ws, 'diabetes-experiment-env')
