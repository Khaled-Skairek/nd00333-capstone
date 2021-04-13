# Heart failure prediction using auto ML and hyper-drive

Two powerful features of Azure ML are used to get the best classification model. The classification model gets several properties(features) of an individual as input and predicts if that individual would die due to heart failure or not.

## Project Set Up and Installation
The project uses an external dataset from Kaggle. This dataset has to be registered in Azure Studio under the name "Heart-failure-prediction" so it can be used in jupyter notebook. A copy of the dataset (as csv) is provided in the repo under data folder.

## Dataset

### Overview
The dataset that was used in the project is the heart failure prediction dataset provided from Kaggle. A description of the dataset can be found under the link 
https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
We are going to use the heart failure prediction dataset to train a classification model which predicts if an individual is going to die or not. The dataset, in each entry, provides several properties(features) about individual and a binary label which tells if that individual died or not due to heart failure. The features are (as Kaggle describes them):  
**Age**  
**Anaemia**: Decrease of red blood cells or hemoglobin(boolean)  
**creatinine_phosphokinase**: Level of the CPK enzyme in the blood (mcg/L)  
**diabetes**: If the patient has diabetes (boolean)  
**ejection_fraction**: Percentage of blood leaving the heart at each contraction (percentage)  
**high_blood_pressure**: If the patient has hypertension (boolean)  
**platelets**: Platelets in the blood (kiloplatelets/mL)  
**serum_creatinine**: Level of serum creatinine in the blood (mg/dL)  
**serum_sodium**: Level of serum sodium in the blood (mEq/L)  
**sex**: Woman or man (binary)  
**smoking**: If the patient smokes or not (boolean)  
**time**: Follow-up period (days)  
**DEATH_EVENT**: If the patient deceased during the follow-up period (boolean)  

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
Here is a link to the screencasting in which I describe the following aspects:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

https://youtu.be/BIaTe5ZHJTg


