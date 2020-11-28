import azureml.core
from azureml.core import Workspace

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies


# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# ajout new package to existing environment
myenv = Environment.get(ws, "test-TALIB-env-3.6-env")
conda_dep = myenv.python.conda_dependencies
conda_dep.add_pip_package("graphviz")
myenv.python.conda_dependencies = conda_dep

print(myenv.name, 'defined.')
myenv.register(workspace=ws)
