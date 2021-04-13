#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter Tuning using HyperDrive
# 
# TODO: Import Dependencies. In the cell below, import all the dependencies that you will need to complete the project.

# In[1]:


from azureml.core import Workspace, Experiment
from azureml.train.sklearn import SKLearn 
from azureml.train.hyperdrive.run import PrimaryMetricGoal 
from azureml.train.hyperdrive.policy import BanditPolicy 
from azureml.train.hyperdrive.sampling import RandomParameterSampling 
from azureml.train.hyperdrive.runconfig import HyperDriveConfig 
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
from azureml.widgets import RunDetails 


# ## Dataset
# 
# TODO: Get data. In the cell below, write code to access the data you will be using in this project. Remember that the dataset needs to be external.

# In[2]:


ws = Workspace.from_config()
experiment_name = 'Hyper-drive-experiment'
experiment=Experiment(ws, experiment_name)


# In[3]:


from azureml.core.compute import ComputeTarget, AmlCompute

# TODO: Create compute cluster
# Use vm_size = "Standard_D2_V2" in your provisioning configuration.
# max_nodes should be no greater than 4.

from azureml.core.compute_target import ComputeTargetException 

# Choose a name for your CPU cluster 
cpu_cluster_name = "notebook142329"

# Verify that cluster does not exist already 
try: 
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.') 
except ComputeTargetException: 
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2', 
                                                           max_nodes=4) 
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config) 

cpu_cluster.wait_for_completion(show_output=True)


# ## Hyperdrive Configuration
# 
# TODO: Explain the model you are using and the reason for chosing the different hyperparameters, termination policy and config settings.

# In[8]:


# Create an early termination policy. This is not required if you are using Bayesian sampling.
early_termination_policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1) 

# Create the different params that you will be using during training
param_sampling = RandomParameterSampling({ 
    "--learning_rate": uniform(0.01, 0.3),
    "--epochs": choice(4, 8, 16, 32),
    "--neurons": choice(48, 60, 72, 84, 96)}) 
                              

# Create your estimator and hyperdrive config
estimator = SKLearn(source_directory=".", compute_target=cpu_cluster, entry_script='entry.py', pip_packages=['azureml-dataprep', 'tensorflow']) 

hyperdrive_run_config = HyperDriveConfig(estimator=estimator,  
                                     hyperparameter_sampling=param_sampling,  
                                     policy=early_termination_policy,  
                                     primary_metric_name='accuracy',  
                                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,  
                                     max_total_runs=64, 
                                     max_concurrent_runs=2)


# In[5]:


# Submit your experiment
hdr = experiment.submit(config=hyperdrive_run_config)


# ## Run Details
# 
# OPTIONAL: Write about the different models trained and their performance. Why do you think some models did better than others?
# 
# TODO: In the cell below, use the `RunDetails` widget to show the different experiments.

# In[6]:


RunDetails(hdr).show()


# In[9]:


hdr.wait_for_completion() 


# ## Best Model
# 
# TODO: In the cell below, get the best model from the hyperdrive experiments and display all the properties of the model.

# In[7]:


best_run = hdr.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()

print('Best run id:', best_run.id) 
print('\n Accuracy:', best_run_metrics['accuracy'])  
print('\n Learning rate:', best_run_metrics[r'learning_rate'])  
print('\n Number of neurons in hidden layer', best_run_metrics[r'neurons'])


# ## Model Deployment
# 
# Remember you have to deploy only one of the two models you trained.. Perform the steps in the rest of this notebook only if you wish to deploy this model.
# 
# TODO: In the cell below, register the model, create an inference config and deploy the model as a web service.

# TODO: In the cell below, send a request to the web service you deployed to test it.

# TODO: In the cell below, print the logs of the web service and delete the service
