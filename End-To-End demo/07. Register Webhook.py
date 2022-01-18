# Databricks notebook source
model_name = 'DocType_PyFunc_Test'
cmr_host = "https://e2-demo-west.cloud.databricks.com"
cmr_token = dbutils.secrets.get(scope='mlops_webinar',key='CMR-token')
auth_header = {"Authorization": f"Bearer {cmr_token}"}

# COMMAND ----------

# MAGIC %md ## List Existing WebHooks

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

# COMMAND ----------

# MAGIC %md ## Create a new WebHook

# COMMAND ----------

create_webhook_endpoint = f"{cmr_host}/api/2.0/mlflow/registry-webhooks/create"

job_id = 150888
releases_job_spec = {
  'job_id': job_id,
  'access_token': cmr_token
}
create_webhook_doc = {
  'model_name': model_name,
  'events': 'MODEL_VERSION_TRANSITIONED_STAGE',
  'description': f"{model_name} CI-CD WebHook",
  'job_spec': releases_job_spec
}
response = requests.post( create_webhook_endpoint, headers=auth_header, data=json.dumps(create_webhook_doc) )
response.content

# COMMAND ----------

# MAGIC %md ## Test WebHook

# COMMAND ----------

test_webhook_endpoint = f"{cmr_host}/api/2.0/mlflow/registry-webhooks/test"

test_webhook_doc = {
  'id': '01bef2971d7348a7b534f1e7ee8ca4e3',
  'event': 'MODEL_VERSION_TRANSITIONED_STAGE'
}
response = requests.post( test_webhook_endpoint, headers=auth_header, data=json.dumps(test_webhook_doc) )
response.content

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delete a WebHook

# COMMAND ----------

delete_webhook_endpoint = f"{cmr_host}/api/2.0/mlflow/registry-webhooks/delete"

del_webhook_doc = { 'id': '217120f15732498fbc21ce443b8a075a' }
response = requests.delete( delete_webhook_endpoint, headers=auth_header, data=json.dumps(del_webhook_doc) )
response.content

# COMMAND ----------


