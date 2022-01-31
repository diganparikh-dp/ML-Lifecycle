# Databricks notebook source
# MAGIC %md
# MAGIC # End-To-End MLOps Text Classification example using transfer learning and MLflow
# MAGIC __WORK_IN_PROGRESS__
# MAGIC 
# MAGIC **PART 3a/7 - ML Engineer: Create Model Webhooks**
# MAGIC 1. Validation job at transition request
# MAGIC 2. Slack notification at transition acceptance
# MAGIC 3. AzureDevOps job at transition acceptance
# MAGIC 
# MAGIC _P.S: This notebook need to be run only once (interactively or automated) every time a new model gets created in order to append the webhooks into it_

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/diganparikh-dp/Images/blob/main/ML%20End%202%20End%20Workflow/MLOps%20end2end%20-%20Corvel_ML.jpg?raw=true" width=860/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment setup
# MAGIC Define parameters, install requirement and define context

# COMMAND ----------

# DBTITLE 1,Create parameters as input 'widgets'
dbutils.widgets.removeAll()
dbutils.widgets.text("MODEL_NAME","DocType_Test", "Model Name")
dbutils.widgets.text("MLFLOW_HOST_URL","https://e2-demo-field-eng.cloud.databricks.com", "Model Registry URL")
dbutils.widgets.text("mlops_job_id","341239", "Job ID (Databricks MLOps Validation):")
dbutils.widgets.text("azure_job_id","341258", "Job ID (Azure DevOps release pipeline):")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get model name, Model Registry URL and Access Token info

# COMMAND ----------

model_name = dbutils.widgets.get("MODEL_NAME")
mlflow_host_url = dbutils.widgets.get("MLFLOW_HOST_URL")

# COMMAND ----------

# token = dbutils.secrets.get(scope="ml-scope", key="dp-token") # PAT for Central Model Registry
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get() # Local PAT

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

# DBTITLE 1,Create Helper Function for MLflow API calls
import json
import requests

def mlflow_call_endpoint(endpoint="", method="POST", body="{}", mlflow_host_url="", token=None):
    
    if token:
        auth_header = {"Authorization": f"Bearer {token}"}
    else:
        auth_header = {}

    list_endpoint = f"{mlflow_host_url}/api/2.0/mlflow/{endpoint}"
    
    if method == "GET":
        response = requests.get(list_endpoint, headers=auth_header, data=json.dumps(body))
    elif method == "POST":
        response = requests.post(list_endpoint, headers=auth_header, data=json.dumps(body))
    else:
        return {"Invalid Method"}

    return response.text

# COMMAND ----------

# DBTITLE 1,Create Helper Function for webhooks management
# List
def list_webhooks(model_name):
    list_model_webhooks = {"model_name": model_name}
    response = mlflow_call_endpoint("registry-webhooks/list", method = "GET", body = list_model_webhooks, mlflow_host_url=mlflow_host_url, token=token)
    
    if isinstance(response, str):
      response = json.loads(response)
    return(response)

# Delete
def delete_webhooks(webhook_id):
    # Remove a webhook
    response = mlflow_call_endpoint("registry-webhooks/delete", method="DELETE",
                     body = {'id': webhook_id}, mlflow_host_url=mlflow_host_url, token=token)
    
    return(response)

def reset_webhooks(model_name):
    whs = list_webhooks(model_name)
    if 'webhooks' in whs:
        for wh in whs['webhooks']:
            delete_webhooks(wh['id'])

# COMMAND ----------

# Reset (for Demo purposes)
reset_webhooks(model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Webhook #1: Trigger MLOps validation test job
# MAGIC When ***model transition request*** happens: trigger a [databricks notebook job](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/1390462475054837/command/1390462475054839) for model validation

# COMMAND ----------

# DBTITLE 1,Helper function to create databricks job webhook
def create_job_webhook(model_name, job_id, mlflow_host_url_in=mlflow_host_url, token_in=token, events=["TRANSITION_REQUEST_CREATED"], description=""):
    trigger_job = {
      "model_name": model_name,
      "events": events,
      "description": description,
      "status": "ACTIVE",
      "job_spec": {
        "job_id": str(job_id),
        "workspace_url": mlflow_host_url_in,
        "access_token": token_in
      }
    }
    response = mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_job, mlflow_host_url=mlflow_host_url_in, token=token_in)
    return(response)

# COMMAND ----------

# DBTITLE 1,Create webhook to trigger MLOps validation job
mlops_job_id = dbutils.widgets.get("mlops_job_id") # This is our 04_ML-Engineer-MLOps-Validation notebook

# Add the webhook to trigger job:
create_job_webhook(model_name = model_name,
                   job_id = mlops_job_id,
                   description="Trigger a databricks validation job when model transition is requested."
                  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Webhook #2: Slack notification
# MAGIC Notify MLOps slack channel **when model transition gets accepted**

# COMMAND ----------

# DBTITLE 1,Helper function to create slack notification webhook
def create_notification_webhook(model_name, slack_url, events=["MODEL_VERSION_TRANSITIONED_STAGE"], description=""):
    trigger_slack = {
        "model_name": model_name,
        "events": events,
        "description": description,
        "status": "ACTIVE",
        "http_url_spec": {
            "url": slack_url
        }
    }
    response = mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_slack, mlflow_host_url=mlflow_host_url, token=token)
    
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
                            events=["MODEL_VERSION_TRANSITIONED_STAGE"],
                            description="Notify the MLOps team that model transition has been accepted."
                           )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Webhook #3: Trigger Azure DevOps pipeline job
# MAGIC When model transition is accepted run a [databricks notebook job](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/1390462475087299/command/1390462475087314) which will trigger an Azure DevOps release pipeline

# COMMAND ----------

azure_job_id = dbutils.widgets.get("azure_job_id") # This is our 05_ML-Engineer-Trigger-AzureDevOps notebook

# Add the webhook to trigger job:
create_job_webhook(model_name = model_name,
                   job_id = azure_job_id,
                   events="MODEL_VERSION_TRANSITIONED_STAGE",
                   description="Start a databricks job which triggers an Azure DevOps pipeline when model transition is accepted."
                  )

# COMMAND ----------

list_webhooks(model_name)

# COMMAND ----------


