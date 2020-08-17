from azureml.core import Run
import pandas as pd
import os
import sys
import glob


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

# Count the rows and log the result
row_count = (len(df))
run.log('observations', row_count)
print('Analyzing {} rows of data'.format(row_count))
      
# Save a sample of the data in the outputs folder (which gets uploaded automatically)
os.makedirs('outputs', exist_ok=True)
df.sample(100).to_csv("outputs/sample.csv", index=False, header=True)

# Complete the run
run.complete()
