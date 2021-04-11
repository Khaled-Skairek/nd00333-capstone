#!/usr/bin/env python
# coding: utf-8

# # Automated ML
# 
# TODO: Import Dependencies. In the cell below, import all the dependencies that you will need to complete the project.

# In[1]:


from azureml.core import Workspace, Experiment, Dataset
from azureml.train.automl import AutoMLConfig
from azureml.widgets import RunDetails
import joblib


# ## Dataset
# 
# The dataset used is the heart failure prediction dataset available at Kaggle.com.
# Each entry of the dataset conatins information (features) about individual. The task is binary classification; to predict if individual is going to have heart failure or not.

# In[2]:


ws = Workspace.from_config()

# choose a name for experiment
experiment_name = 'AutoML_experiment'

experiment = Experiment(ws, experiment_name)


# In[6]:


from azureml.core.compute import ComputeTarget, AmlCompute

# TODO: Create compute cluster
# Use vm_size = "Standard_D2_V2" in your provisioning configuration.
# max_nodes should be no greater than 4.

from azureml.core.compute_target import ComputeTargetException 

# Choose a name for your CPU cluster 
cpu_cluster_name = "my-cluster"

# Verify that cluster does not exist already 
try: 
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.') 
except ComputeTargetException: 
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2', 
                                                           max_nodes=4) 
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config) 

cpu_cluster.wait_for_completion(show_output=True)


# In[7]:


dataset = Dataset.get_by_name(ws, name='Heart-failure-prediction') 
dataset.to_pandas_dataframe() 


# ## AutoML Configuration
# 
# ### AutoMLSettings
# Due to the available resources (time and space), the auto ml is limited with timeout and maximum allowed parallel computations. The used metric for post-thresholding is  AUC_weighted which optimizes better for small datasets.
# 
# ### AutoMLConfig
# Since the task is classification, we need to provide the type, the dataset, and the labeled column (DEATH_EVENT). Early stopping is enabled to save time as well.

# In[8]:


automl_settings = {
    "experiment_timeout_minutes": 20, 
    "max_concurrent_iterations": 5, 
    "primary_metric" : 'AUC_weighted' 
}

automl_config = AutoMLConfig(task="classification", 
                             training_data=dataset,
                             compute_target=cpu_cluster, 
                             label_column_name="DEATH_EVENT",    
                             path=".", 
                             enable_early_stopping=True, 
                             featurization='auto', 
                             debug_log="automl_errors.log", 
                             **automl_settings)


# In[9]:


# Submit your experiment
remote_run = experiment.submit(automl_config)


# ## Run Details
# 

# In[10]:


RunDetails(remote_run).show() 


# In[12]:


remote_run.wait_for_completion() 


# ## Best Model
# 
# TODO: In the cell below, get the best model from the automl experiments and display all the properties of the model.
# 
# 

# In[14]:


best_run, fitted_model = remote_run.get_output() 
print (best_run) 
print (fitted_model) 


# In[15]:


#Save the best model
joblib.dump(fitted_model, "best_model_auto_ml.model") 


# ## Model Deployment
# 
# Remember you have to deploy only one of the two models you trained.. Perform the steps in the rest of this notebook only if you wish to deploy this model.
# 
# TODO: In the cell below, register the model, create an inference config and deploy the model as a web service.

# In[35]:


from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration

model = Model.register(workspace=ws,
                       model_name='my-autoML-model',                # Name of the registered model in your workspace.
                       model_path='./best_model_auto_ml.model',     # Local file to upload and register as a model.
                       resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                       description='VotingEnsemble') # ,
                       # tags={'area': 'diabetes', 'type': 'classification'})

print('Name:', model.name)
print('Version:', model.version)


# In[31]:


print(ws.environments['AzureML-VowpalWabbit-8.8.0'])


# In[39]:


from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core import Environment

environment = Environment.get(workspace=ws, name="AzureML-AutoML")
service_name = 'heart-failure-prediction'
inference_config = InferenceConfig(entry_script='score.py', environment=environment)
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
service = Model.deploy(workspace=ws,
                      name=service_name,
                      models=[model],
                      inference_config=inference_config,
                      deployment_config=aci_config,
                      overwrite=True)

print(service.get_logs())

service.wait_for_deployment(show_output=True)


# TODO: In the cell below, send a request to the web service you deployed to test it.

# In[38]:


import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = service.scoring_uri
# If the service is authenticated, set the key or token
key = ''

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "age": 17,
            "anaemia": 1,
            "creatinine_phosphokinase": 600,
            "diabetes": 1,
            "ejection_fraction": 30,
            "high_blood_pressure": 0,
            "platelets": 263000,
            "serum_creatinine": 1.2,
            "serum_sodium": 130,
            "sex": 1,
            "smoking": 0,
            "time": 15,
          },
          {
            "age": 35,
            "anaemia": 0,
            "creatinine_phosphokinase": 500,
            "diabetes": 0,
            "ejection_fraction": 30,
            "high_blood_pressure": 0,
            "platelets": 280000,
            "serum_creatinine": 2.1,
            "serum_sodium": 150,
            "sex": 0,
            "smoking": 0,
            "time": 10,
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
# headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())


# TODO: In the cell below, print the logs of the web service and delete the service

# In[ ]:


service.get_logs(num_lines=100)
# service.delete()

# AmlCompute.delete(cpu_cluster)

