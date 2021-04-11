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

experiment=Experiment(ws, experiment_name)


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


# In[ ]:


remote_run.wait_for_completion() 


# ## Best Model
# 
# TODO: In the cell below, get the best model from the automl experiments and display all the properties of the model.
# 
# 

# In[ ]:


best_run, fitted_model = remote_run.get_output() 
print (best_run) 
print (fitted_model) 


# In[ ]:


#Save the best model
joblib.dump(fitted_model, "best_model_auto_ml.model") 


# ## Model Deployment
# 
# Remember you have to deploy only one of the two models you trained.. Perform the steps in the rest of this notebook only if you wish to deploy this model.
# 
# TODO: In the cell below, register the model, create an inference config and deploy the model as a web service.

# In[ ]:





# TODO: In the cell below, send a request to the web service you deployed to test it.

# In[ ]:





# TODO: In the cell below, print the logs of the web service and delete the service

# In[ ]:




