# Databricks notebook source
# MAGIC %md
# MAGIC # End-To-End MLOps Text Classification example using transfer learning and MLflow
# MAGIC __TO-DO__: update diagram
# MAGIC 
# MAGIC **PART 3b/7 - ML Engineer/DevOps: Local API Testing **
# MAGIC 1. REST API local testing (using Model Serving) _OPTIONAL_
# MAGIC 
# MAGIC _P.S: This notebook can also be triggered automatically every time a Data-Scientist pushes a new BERT or LSTM model version to the local registry_

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/diganparikh-dp/Images/main/Corvel%20Future%20Diagram.png" width=860/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment setup
# MAGIC Define parameters, install requirement and define context

# COMMAND ----------

# DBTITLE 1,Create parameters as input 'widgets'
dbutils.widgets.removeAll()
dbutils.widgets.text("SAVE_DIR","/dbfs/mnt/oetrta/diganparikh/corvel/corvel_contents", "Global path/URI (ADLS)")
dbutils.widgets.text("USE_CASE", 'symbeo_doctyping', "Use-Case Name")
dbutils.widgets.text("PRE_TRAINED_MODEL_NAME","emilyalsentzer/Bio_ClinicalBERT", "Pre-Trained BERT model to load")
dbutils.widgets.text("MODEL_NAME","DocType_Test", "Model Name")
dbutils.widgets.text("MLFLOW_CENTRAL_URI","databricks://ml-scope:dp", "Central Model Registry URI")
dbutils.widgets.dropdown("stage","Staging", ["None", "Archived", "Staging", "Production"], "Transition to:")

# COMMAND ----------

# DBTITLE 1,Import libs/packages of choice
from datetime import datetime

import os, json, sys, time, random
import nltk
from nltk.corpus import stopwords

import mlflow

import keras
import pandas as pd

import sklearn
import joblib

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from textwrap import wrap

import tensorflow
from tensorflow.keras.models import load_model
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

import re
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# DBTITLE 1,Get latest version number
model_name = dbutils.widgets.get("MODEL_NAME")
latest_model = client.search_model_versions(f"name='{model_name}'")
latest_model_version = latest_model[0].version

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
