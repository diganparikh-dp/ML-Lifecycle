# Databricks notebook source
# MAGIC %md
# MAGIC # End-To-End MLOps Text Classification example using transfer learning and MLflow
# MAGIC __WORK_IN_PROGRESS__
# MAGIC 
# MAGIC **PART 6/7 - ML Engineer/DevOps: Notebook to push MLflow artifact to AKS** _(scheduled as job to be triggered by Azure DevOps)_
# MAGIC 1. Pull custom artifacts from central model registry
# MAGIC 2. Push to AKS
# MAGIC * Create Job and retrieve `JobID` (DO ONCE) - `332066`

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/diganparikh-dp/Images/blob/main/ML%20End%202%20End%20Workflow/MLOps%20end2end%20-%20Corvel_ML3.jpg?raw=true" width=860/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment setup
# MAGIC Define parameters, install requirement and define context

# COMMAND ----------

# DBTITLE 1,Create parameters as input 'widgets'
dbutils.widgets.removeAll()
dbutils.widgets.text("MODEL_NAME","DocType_Test", "Model Name")
dbutils.widgets.text("MLFLOW_CENTRAL_URI","databricks://ml-scope:dp", "Central Model Registry URI")
dbutils.widgets.text("event_message","databricks://ml-scope:dp", "Central Model Registry URI")
dbutils.widgets.dropdown("stage","Staging", ["None", "Archived", "Staging", "Production"], "Transition to:")

# COMMAND ----------

# DBTITLE 1,Set MLFlow to point to Central Server
registry_uri = dbutils.widgets.get("MLFLOW_CENTRAL_URI")
mlflow.set_registry_uri(registry_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch Model In Transition

# COMMAND ----------

# After receiving payload from webhooks, use MLflow client to retrieve model details and lineage
def fetch_webhook_data(): 
    try:
        registry_event = json.loads(dbutils.widgets.get('event_message'))
        model_name = registry_event['model_name']
        model_version = registry_event['version']
        
        if 'to_stage' in registry_event:
            stage = registry_event['to_stage']
        else:
            dbutils.notebook.exit("Invalid stage requested")
    except:
        #If it's not in a job but interactive demo, we get the last version from the registry
        model_name = dbutils.widgets.get("MODEL_NAME")
        model_version = client.get_registered_model(model_name).latest_versions[0].version
        stage = dbutils.widgets.get("stage")
    return(model_name, model_version, stage)

# COMMAND ----------

# Get the model in transition, its name and version from the metadata received by the webhook
model_name, latest_model_version, stage = fetch_webhook_data()

# COMMAND ----------

# MAGIC %md
# MAGIC ## [Push to AML/AKS](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models)
# MAGIC 
# MAGIC <img src="https://docs.microsoft.com/en-us/azure/machine-learning/media/how-to-deploy-mlflow-models/mlflow-diagram-deploy.png" width=700/>

# COMMAND ----------

import json
  
# Data to be written
deploy_config ={
    "computeType": "aks",
    "computeTargetName": "aks-mlflow"
}

# Serializing json 
json_object = json.dumps(deploy_config)
  
# Writing to sample.json
with open(f"{SAVE_DIR}/deployment_config.json", "w") as outfile:
    outfile.write(json_object)

# COMMAND ----------

from mlflow.deployments import get_deploy_client

# set the tracking uri as the deployment client
client = get_deploy_client(mlflow.get_tracking_uri())
model_uri = f"models:/{model_name}/{stage}'

# set the model path 
model_path = "model"

# set the deployment config
deployment_config_path = f"{SAVE_DIR}/deployment_config.json"
test_config = {'deploy-config-file': deployment_config_path}

# define the model path and the name is the service name
# the model gets registered automatically and a name is autogenerated using the "name" parameter below 
client.create_deployment(model_uri=model_uri,
                         config=test_config,
                         name=f"{USE_CASE}-aks-deployment")
