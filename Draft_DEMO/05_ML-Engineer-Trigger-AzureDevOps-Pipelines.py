# Databricks notebook source
# MAGIC %md
# MAGIC # End-To-End MLOps Text Classification example using transfer learning and MLflow
# MAGIC __WORK_IN_PROGRESS__
# MAGIC 
# MAGIC **PART 5/7 - ML Engineer/DevOps: Trigger Azure DevOps pipeline notebook** _(scheduled as job to be triggered during Model Transition)_
# MAGIC 1. Parse payload
# MAGIC 2. Trigger Azure DevOps via REST API
# MAGIC * Create Job and retrieve `JobID` _(DO ONCE)_ `332018`

# COMMAND ----------

# MAGIC %md
# MAGIC **TO-DO**
# MAGIC _add diagram_

# COMMAND ----------

# DBTITLE 1,Notebook params
dbutils.widgets.text("event_message",'{}',"Webhook payload")
dbutils.widgets.text("org_id",'tristannixon-databricks',"Organization ID")
dbutils.widgets.text("project_id",'MLOps%20Webinar',"Project ID")
dbutils.widgets.text("QA_pipeline_id",'4',"Pipeline ID QA (Azure DevOps)")
dbutils.widgets.text("production_pipeline_id",'3',"Pipeline ID Production (Azure DevOps)")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Parse webhook payload

# COMMAND ----------

import json
 
event_message = dbutils.widgets.get("event_message")
event_message_dict = json.loads(event_message)

# COMMAND ----------

# DBTITLE 1,Contextual parameters definition
import base64

org_id = dbutils.widgets.get("org_id")
project_id = dbutils.widgets.get("project_id")
devops_token = dbutils.secrets.get(scope='mlops',key='devops_Token')

releases_uri = f"https://vsrm.dev.azure.com/{org_id}/{project_id}/_apis/release/releases?api-version=6.0"
encoded_token = base64.b64encode(bytes(f":{devops_token}", 'utf-8')).decode("utf-8")
devops_auth = {'Authorization': f"Basic {encoded_token}",
               'Content-Type': 'application/json'}

# COMMAND ----------

# MAGIC %md ## Trigger Release pipeline
# MAGIC Based on stage transition request

# COMMAND ----------

# DBTITLE 1,Create helper calls
import requests

def trigger_release_pipeline( pipeline_def_id ):
  create_release_doc = { 'definitionId': pipeline_def_id }
  response = requests.post(releases_uri, headers=devops_auth, data = json.dumps(create_release_doc) )
  return response.content

# COMMAND ----------

if event_message_dict['event'] == 'MODEL_VERSION_TRANSITIONED_STAGE':
  if event_message_dict['to_stage'] == 'Staging':
    print("Running Staging pipeline")
    results = trigger_release_pipeline(int(dbutils.widgets.get("QA_pipeline_id")))
  elif event_message_dict['to_stage'] == 'Production' :
    print("Running Production pipeline")
    results = trigger_release_pipeline(int(dbutils.widgets.get("production_pipeline_id")))
  else:
    print("Invalid requested stage (expecting Staging/Production only)")
    results = None
    dbutils.notebook.exit("Invalid requested stage (expecting Staging/Production only)")
  
  print(results)
else:
  print("Wrong trigger, exiting job")
  dbutils.notebook.exit("Wrong trigger, exiting job")
