# Databricks notebook source
# MAGIC %md
# MAGIC # End-To-End MLOps Text Classification example using transfer learning and MLflow
# MAGIC __TO-DO__: update diagram
# MAGIC 
# MAGIC **PART 4/7 - ML Engineer: Model Validation Test notebook** _(scheduled as job to be triggered during Model Transition Request)_
# MAGIC 1. Pull custom artifacts from central model registry
# MAGIC 2. Schema Validation check
# MAGIC 3. QA check
# MAGIC 4. Accept Transition _(to requested `stage`)_
# MAGIC * Create Job and retrieve `JobID` _(DO ONCE)_ `330465`

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/diganparikh-dp/Images/main/Corvel%20Future%20Diagram.png" width=860/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment setup
# MAGIC Define parameters, install requirement and define context

# COMMAND ----------

import mlflow
import json

# COMMAND ----------

# DBTITLE 1,Create parameters as input 'widgets'
dbutils.widgets.removeAll()
dbutils.widgets.text("INPUT_DATA", "/mnt/oetrta/diganparikh/corvel/corvel_contents/iter6.14_pocsample.csv", "Test file")
dbutils.widgets.text("MODEL_NAME","DocType_Test", "Model Name")
dbutils.widgets.text("MLFLOW_CENTRAL_URI","databricks://ml-scope:dp", "Central Model Registry URI")
dbutils.widgets.text("event_message","{}", "Webhook payload")
dbutils.widgets.dropdown("stage","Staging", ["None", "Archived", "Staging", "Production"], "Transition to:")

# COMMAND ----------

# DBTITLE 1,Set MLFlow to point to Central Server
registry_uri = dbutils.widgets.get("MLFLOW_CENTRAL_URI")
mlflow.set_registry_uri(registry_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch Model In Transition

# COMMAND ----------

# DBTITLE 1,Helper function to parse webhook payload
# Instantiate MLflow client
client = mlflow.tracking.MlflowClient()

def fetch_webhook_data(): 
    """
    Parse payload from webhooks and use MLflow client to retrieve model details and lineage
    """
    try:
        registry_event = json.loads(dbutils.widgets.get('event_message'))
        model_name = registry_event['model_name']
        model_version = registry_event['version']
        if 'to_stage' in registry_event:
            stage = registry_event['to_stage']
        else:
            print("Invalid trigger, exiting notebook...")
            dbutils.notebook.exit()
        print("Parsing webhook payload")
    except:
        #If it's not in a job but interactive run, retrieve last version from the registry
        model_name = dbutils.widgets.get("MODEL_NAME")
        model_version = client.search_model_versions(f"name='{model_name}'")[-1].version
        stage = dbutils.widgets.get("stage")
        print("Parsing notebook parameters (not payload)")

    return(model_name, model_version, stage)

# COMMAND ----------

# Get the model in transition, its name and version from the metadata received by the webhook
model_name, latest_model_version, stage = fetch_webhook_data()
model_uri = f"models:/{model_name}/{latest_model_version}"
print(f"Validating '{model_name}', version {latest_model_version} for {stage} from {mlflow.get_registry_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schema validation check: perform inference against pandas DataFrame

# COMMAND ----------

# DBTITLE 0,inferencing against pandas df... works fine but goes through the nltk installation process
import pandas as pd

test_df = pd.DataFrame({'Request':["City of Fort Matthews-EC-TT  CarelQ  Automated to Georgia - Site 11  A CORVEL NETWORK  CarelQ Transportation  Invoice Date: 01/01/2020  Corvel Scan Date: 02/02/2020  Transportation /Translation Invoice :  123456  Account Group:  Patient  Claim #  Date of Service  ItemId  Item Name  Quantity  Rate  Charge  Elden , Isiah, Rili  1111-WC-  1-12-18  TRANS-AMB ROUND  TRANSPORTATION- ROUND TRIP TOTAL  $336.07  $336.07  18-0000088  TRIP  9AM FR 121 HELM ST FORT Matthews CA TO ADVANCED  Winterhill Lodge 2002 361 BALTHAM ST PORT CHARLOTTE CA  FORT MAT  John Wick  1111 -WC  05/12/2015  TRANS- AMB WAIT  TRANSPORTATION - WAIT TIME, AMBULATORY  $49.39  $249.55  18-0008888  TIME  Total Charges :  $895.62  This is not a medical bill . Thank you for your business !  Make payable to:  (321)555-4600  CarelQ  PO Box 1000 S. Main East   TIN: (123) 555 - 4148", "Hi This is a Test, what is the label?"]})

try:
    # Load model as a PyFuncModel
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Predict on a Pandas DataFrame
    out_df = loaded_model.predict(test_df)
    display(out_df)

    schema_check = "Schema OK"
except:
    schema_check = "Schema validation failed"
    out_df = pd.DataFrame()
    
print(schema_check)

# COMMAND ----------

# MAGIC %md
# MAGIC ## QA test: Batch Inference on test dataset
# MAGIC _can be a file, table and/or Feature Store_

# COMMAND ----------

# DBTITLE 1,Create test dataframe
from pyspark.sql.types import StructType, StringType, DoubleType, ArrayType, StructField

input_data = dbutils.widgets.get("INPUT_DATA")

this_schema = StructType() \
      .add("Request",StringType(),True) \
      .add("label",StringType(),True) \
      .add("dcn",StringType(),True) \
      .add("page",DoubleType(),True)

pocsampleDF = (spark.read
               .format("csv")
               .option("header", False)
               .option("multiLine", True)
               .option("escape", '"')
               .schema(this_schema)
               .load(input_data)
              )

n_samples = pocsampleDF.count()

# Repartition
pocsampleDF = pocsampleDF.repartition(int(n_samples)) # ONLY FOR PURPOSE OF POC
print(f"Testing {n_samples} samples from {input_data}")

# COMMAND ----------

# DBTITLE 1,Batch Inference using Spark UDF
from pyspark.sql.functions import col

try:
    # Load model as a Spark UDF.
    loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type = ArrayType(StringType()))
    
    # Predict on a Spark DataFrame.
    columns = list(['Request'])
    sparkUDF_test = (pocsampleDF
                     .withColumn('predictions', loaded_model(*columns))
                     .withColumn("pred_label", col("predictions").getItem(0))
                     .withColumn("pred_score", col("predictions").getItem(1))
                     .withColumn("pred_elapsed_time", col("predictions").getItem(2))
                     .drop("predictions")
                    )
    display(sparkUDF_test)
    QA_check = "QA OK"

except:
    QA_check = "QA failed"
    schema = StructType([
      StructField('pred_label', StringType(), True),
      StructField('pred_score', DoubleType(), True),
      StructField('pred_elapsed_time', DoubleType(), True)
      ])
    sparkUDF_test = spark.createDataFrame([], schema)
    
print(QA_check)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Other checks such as:
# MAGIC * signature check
# MAGIC * description check
# MAGIC * artifact check
# MAGIC * concept/data drift detection
# MAGIC * etc.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Accept/Reject Transition request

# COMMAND ----------

# DBTITLE 1,Create Helper Calls
from mlflow.utils.rest_utils import http_request
import json

def mlflow_call_endpoint(endpoint, method, body='{}'):
    
    # Get host url and access token for workspace to create webhooks on
    client_ = mlflow.tracking.client.MlflowClient()
    host_creds = client_._tracking_client.store.get_host_creds()
    
    if method == 'GET':
        response = http_request(
            host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
    else:
        response = http_request(
            host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body))
    
    return response.json()

# Accept or reject transition request
def accept_transition(model_name, version, stage, comment):
    approve_request_body = {'name': model_name,
                            'version': version,
                            'stage': stage,
                            'archive_existing_versions': 'true',
                            'comment': comment}
  
    mlflow_call_endpoint('transition-requests/approve', 'POST', json.dumps(approve_request_body))

def reject_transition(model_name, version, stage, comment):
    reject_request_body = {'name': model_name,
                           'version': version,
                           'stage': stage, 
                           'comment': comment}
    
    mlflow_call_endpoint('transition-requests/reject', 'POST', json.dumps(reject_request_body))

# COMMAND ----------

# If any checks failed, reject and move to Archived
if all([not(out_df.empty), not(sparkUDF_test.rdd.isEmpty())]): 
    print(f"Accepting transition to {stage}...")
    accept_transition(model_name,
                   latest_model_version,
                   stage=stage,
                   comment=f'All tests passed!  Moving to {stage}.')
else:
    print(f"Rejecting transition to {stage}...")
    reject_transition(model_name,
                   latest_model_version,
                   stage=stage,
                   comment='Tests failed, moving to archived.  Check the job run to see what happened.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create/Schedule as manual job
# MAGIC __DO ONCE__
# MAGIC [Here](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#job/330465)
