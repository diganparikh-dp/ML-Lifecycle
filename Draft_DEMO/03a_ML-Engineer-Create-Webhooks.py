# Databricks notebook source
# MAGIC %md
# MAGIC # End-To-End MLOps Text Classification example using transfer learning and MLflow
# MAGIC __WORK_IN_PROGRESS__
# MAGIC 
# MAGIC **PART 3a/7 - ML Engineer: Create Model Webhooks**
# MAGIC 1. Create model **webhooks** 
# MAGIC 
# MAGIC _P.S: This notebook need to be run only once (interactively or automated) every time a new model gets created in order to append the webhooks into it_

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment setup
# MAGIC Define parameters, install requirement and define context

# COMMAND ----------

# DBTITLE 1,Create parameters as input 'widgets'
dbutils.widgets.removeAll()
dbutils.widgets.text("MODEL_NAME","DocType_Test", "Model Name")
dbutils.widgets.text("MLFLOW_CENTRAL_URI","databricks://ml-scope:dp", "Central Model Registry URI")
dbutils.widgets.text("mlops_job_id","330465", "Job ID (Databricks MLOps Validation):")
dbutils.widgets.text("azure_job_id","332018", "Job ID (Azure DevOps release pipeline):")

# COMMAND ----------

model_name = dbutils.widgets.get("MODEL_NAME")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Webhooks for given model
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn-mlflow-webhook.png" width=1000 >
# MAGIC <br><br>
# MAGIC 
# MAGIC ### Supported Events
# MAGIC * Registered model created
# MAGIC * Model version created
# MAGIC * Transition request created
# MAGIC * Accept/Reject transition request
# MAGIC * Comment on a model version
# MAGIC 
# MAGIC ### Types of webhooks
# MAGIC * HTTP webhook -- send triggers to endpoints of your choosing such as **slack**, AWS Lambda, **Azure DevOps/Functions**, or GCP Cloud Functions
# MAGIC * Job webhook -- trigger a job within the Databricks workspace

# COMMAND ----------

# DBTITLE 1,Import libs/packages
import mlflow

# COMMAND ----------

# DBTITLE 1,Pre-Requisite: Point/Use Central MLFlow Server (ONLY IF NOT RUNNING FROM CENTRAL WORKSPACE)
registry_uri = dbutils.widgets.get("MLFLOW_CENTRAL_URI")
mlflow.set_registry_uri(registry_uri)

# COMMAND ----------

# DBTITLE 1,Pre-Requisite: Define/Get host url and access token of given workspace (i.e. Dev/QA/Prod)
# Get host url and access token for workspace to create webhooks on
client_ = mlflow.tracking.client.MlflowClient()

host_creds = client_._tracking_client.store.get_host_creds()
host = host_creds.host
token = host_creds.token

# COMMAND ----------

#LIST HOOK
import json
import requests

# model_name = 'DocType_PyFunc_Test'
cmr_host = "https://e2-demo-field-eng.cloud.databricks.com"
cmr_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
#cmr_token = dbutils.secrets.get(scope='ml-scope',key='dp-token')
auth_header = {'Authorization': f'Bearer {cmr_token}'}

list_endpoint = f"{cmr_host}/api/2.0/mlflow/registry-webhooks/list"
list_webhook_params = {
  'model_name': model_name
}
response = requests.get( list_endpoint, headers=auth_header, data=json.dumps(list_webhook_params) )
response.content

# COMMAND ----------

#model_name = 'DocType_Test'
cmr_host = "https://e2-demo-west.cloud.databricks.com"
cmr_token = dbutils.secrets.get(scope='ml-scope',key='dp-token')
auth_header = {"Authorization": f"Bearer {cmr_token}"}

list_endpoint = f"{cmr_host}/api/2.0/mlflow/registry-webhooks/list"
list_webhook_params = {
  'model_name': model_name
}
response = requests.get( list_endpoint, headers=auth_header, data=json.dumps(list_webhook_params) )
response.content

# COMMAND ----------

# DBTITLE 1,Create Helper Function
from mlflow.utils.rest_utils import http_request
import json

def mlflow_call_endpoint(endpoint, method, body='{}'):
    if method == 'GET':
        response = http_request(
            host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
    else:
        response = http_request(
            host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body))
    
    return response.json()

# Manage webhooks

# List
def list_webhooks(model_name):
    list_model_webhooks = json.dumps({"model_name": model_name})
    response = mlflow_call_endpoint("registry-webhooks/list", method = "GET", body = list_model_webhooks)
    
    return(response)

# Delete
def delete_webhooks(webhook_id):
    # Remove a webhook
    response = mlflow_call_endpoint("registry-webhooks/delete", method="DELETE",
                     body = json.dumps({'id': webhook_id}))
    
    return(response)

def reset_webhooks(model_name):
    whs = list_webhooks(model_name)
    if 'webhooks' in whs:
        for wh in whs['webhooks']:
            delete_webhooks(wh['id'])

# COMMAND ----------

list_webhooks("DocType_Test")

# COMMAND ----------

#CREATE HOOK
create_webhook_endpoint = f"{cmr_host}/api/2.0/mlflow/registry-webhooks/create"

job_id = 333539
releases_job_spec = {
  'job_id': job_id,
  'access_token': cmr_token
}
create_webhook_doc = {
  'model_name': model_name,
  'events': 'TRANSITION_REQUEST_CREATED',
  'description': f"{model_name} CI-CD WebHook",
  'job_spec': releases_job_spec
}
response = requests.post( create_webhook_endpoint, headers=auth_header, data=json.dumps(create_webhook_doc) )
response.content

# COMMAND ----------

#TEST HOOK
test_webhook_endpoint = f"{cmr_host}/api/2.0/mlflow/registry-webhooks/test"

test_webhook_doc = {
  'id': '5c5443241c4d4d9b8e944769f5a40e7e',
  'event': 'MODEL_VERSION_TRANSITIONED_STAGE'
}
response = requests.post( test_webhook_endpoint, headers=auth_header, data=json.dumps(test_webhook_doc) )
response.content

# COMMAND ----------

# Reset (for Demo purposes)
reset_webhooks(model_name = model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Webhook #1: Trigger MLOps validation test job
# MAGIC When ***model transition request*** happens: trigger a [databricks notebook job](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/1390462475054837/command/1390462475054839) for model validation

# COMMAND ----------

# DBTITLE 1,Helper function to create databricks job webhook
def create_job_webhook(model_name, job_id, host_in=host, token_in=token, events=["TRANSITION_REQUEST_CREATED"], description=""):
    trigger_job = json.dumps({
      "model_name": model_name,
      "events": events,
      "description": description,
      "status": "ACTIVE",
      "job_spec": {
        "job_id": str(job_id),
        "workspace_url": host_in,
        "access_token": token_in
      }
    })
    response = mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_job)
    return(response)

# COMMAND ----------

# DBTITLE 1,Create webhook to trigger MLOps validation job
mlops_job_id = dbutils.widgets.get("mlops_job_id") # This is our 04_ML-Engineer-MLOps-Validation notebook

# Add the webhook to trigger job:
create_job_webhook(model_name = model_name,
                   job_id = mlops_job_id,
                   description="Trigger a databricks validation job when a model transition is requested."
                  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Webhook #2: Slack notification
# MAGIC Notify MLOps slack channel **when model transition gets accepted**

# COMMAND ----------

# DBTITLE 1,Helper function to create slack notification webhook
def create_notification_webhook(model_name, slack_url, events=["MODEL_VERSION_TRANSITIONED_STAGE"], description=""):
    trigger_slack = json.dumps({
        "model_name": model_name,
        "events": events,
        "description": description,
        "status": "ACTIVE",
        "http_url_spec": {
            "url": slack_url
        }
    })
    response = mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_slack)
    
    return(response)

# COMMAND ----------

# DBTITLE 1,Get Slack webhook URL
try:
  # HINT: Always recommended to use/redact using Databrick's SECRET
    slack_webhook = dbutils.secrets.get(scope="my-databricks-scope", key="slack-webhook-url")
except:
    print("NO SLACK URL - copy/paste manually")
    slack_webhook = None

# COMMAND ----------

# DBTITLE 1,Create slack notification webhook
create_notification_webhook(model_name = model_name,
                            slack_url = slack_webhook,
                            description="Notify the MLOps team that a model transition has been accepted."
                           )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Webhook #3: Trigger Azure DevOps pipeline job
# MAGIC When model transition is accepted run a [databricks notebook job](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/1390462475087299/command/1390462475087314) which will trigger an Azure DevOps release pipeline

# COMMAND ----------

azure_job_id = dbutils.widgets.get("azure_job_id") # This is our 04_ML-Engineer-MLOps-Validation notebook

# Add the webhook to trigger job:
create_job_webhook(model_name = model_name,
                   job_id = azure_job_id,
                   events="MODEL_VERSION_TRANSITIONED_STAGE",
                   description="Trigger a databricks job which triggers an Azure DevOps pipeline when a model transition is requested.")
