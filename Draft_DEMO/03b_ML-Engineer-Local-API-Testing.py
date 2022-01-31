# Databricks notebook source
# MAGIC %md
# MAGIC # End-To-End MLOps Text Classification example using transfer learning and MLflow
# MAGIC 
# MAGIC **PART 3b/7 - ML Engineer/DevOps: Local API Testing **
# MAGIC 1. REST API local testing (using Model Serving) _OPTIONAL_

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/diganparikh-dp/Images/blob/main/ML%20End%202%20End%20Workflow/MLOps%20end2end%20-%20Corvel_ML2.jpg?raw=true" width=860/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment setup
# MAGIC Define parameters, install requirement and define context

# COMMAND ----------

# DBTITLE 1,Create parameters as input 'widgets'
dbutils.widgets.removeAll()
dbutils.widgets.text("MODEL_NAME","DocType_PyFunc_Test", "Model Name")
dbutils.widgets.dropdown("stage","Staging", ["None", "Archived", "Staging", "Production"], "Stage to Test:")

# COMMAND ----------

# DBTITLE 1,Get latest version number
import mlflow
client = mlflow.tracking.MlflowClient()

model_name = dbutils.widgets.get("MODEL_NAME")

# Get latest version
# latest_model = client.search_model_versions(f"name='{model_name}'")
# latest_model_version = latest_model[0].version


# OR Stage
latest_model_version = dbutils.widgets.get("stage")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Local API Interactive Testing _(OPTIONAL)_:
# MAGIC First ensure that Model Serving is enabled [here](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#mlflow/models/DocType_PyFunc_Test/serving)
# MAGIC 
# MAGIC Example call:
# MAGIC ```
# MAGIC [
# MAGIC {
# MAGIC "Request" : ["Hello World"]
# MAGIC }
# MAGIC ]
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test model serving
# MAGIC Using python API call

# COMMAND ----------

# DBTITLE 1,Create Helper Calls
import os
import requests
import numpy as np
import pandas as pd

def score_model(data: pd.DataFrame,
                model_name: str,
                version: str,
                instance=dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('browserHostName'),
                token=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
               ):
    url = f'https://{instance}/model/{model_name}/{version}/invocations'
    headers = {'Authorization': f'Bearer {token}'}
    data_json = data.to_dict(orient='split')
    response = requests.request(method='POST', headers=headers, url=url, json=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    
    return pd.DataFrame(response.json())

test_df = pd.DataFrame({'Request':['hello world']})

# COMMAND ----------

out_df = score_model(test_df, model_name=model_name, version=latest_model_version)

# COMMAND ----------

display(out_df)

# COMMAND ----------


