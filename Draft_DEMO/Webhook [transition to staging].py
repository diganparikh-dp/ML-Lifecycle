# Databricks notebook source
model_name = 'DocType_Test'
cmr_host = "https://e2-demo-west.cloud.databricks.com"
cmr_token = dbutils.secrets.get(scope='mlops_webinar',key='CMR-token')
auth_header = {"Authorization": f"Bearer {cmr_token}"}

# COMMAND ----------

import json
import requests

model_name = 'DocType_PyFunc_Test'
cmr_host = "https://e2-demo-field-eng.cloud.databricks.com"
cmr_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
auth_header = {'Authorization': f'Bearer {cmr_token}'}

list_endpoint = f"{cmr_host}/api/2.0/mlflow/registry-webhooks/list"
list_webhook_params = {
  'model_name': model_name
}
response = requests.get( list_endpoint, headers=auth_header, data=json.dumps(list_webhook_params) )
response.content
