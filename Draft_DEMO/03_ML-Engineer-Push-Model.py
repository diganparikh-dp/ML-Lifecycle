# Databricks notebook source
# MAGIC %md
# MAGIC # End-To-End MLOps Text Classification example using transfer learning and MLflow
# MAGIC __TO-DO__: update diagram & add link to Azure DevOps project page
# MAGIC 
# MAGIC **PART 3/7 - ML Engineer: Push Model to Central Registry **
# MAGIC 1. Pull independent artifacts from local model registry (BERT & LSTM)
# MAGIC 2. Local/Functionnal Testing
# MAGIC 3. Log custom artifact to MLflow's central model registry
# MAGIC 4. Request Transition to Staging/Production
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
dbutils.widgets.text("MLFLOW_URI_PAT","databricks://ml-scope:dp", "Model Registry PAT")
dbutils.widgets.text("MLFLOW_HOST_URL","https://e2-demo-west.cloud.databricks.com", "Central Model Registry URL")
dbutils.widgets.dropdown("stage","Staging", ["None", "Archived", "Staging", "Production"], "Transition to:")

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install transformers

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

# MAGIC %md
# MAGIC ### Define artifacts location

# COMMAND ----------

SAVE_DIR = dbutils.widgets.get("SAVE_DIR")
USE_CASE = dbutils.widgets.get("USE_CASE")
artifacts_global = {
    "NLTK_DATA_PATH" : os.path.join(SAVE_DIR,"nltk_data"),
    "BERT_MODEL_PATH" : os.path.join(SAVE_DIR,f"{USE_CASE}_bert.bin"),
    "LABEL_ENCODER_PATH" : os.path.join(SAVE_DIR,f"{USE_CASE}_label_encoder.pkl"),
    "LSTM_MODEL_PATH" : os.path.join(SAVE_DIR,f"{USE_CASE}_lstm.h5")
}

# COMMAND ----------

# DBTITLE 1,Get latest model versions
from mlflow.tracking import MlflowClient

client_local = MlflowClient()

bert_model_name = f"CORVEL_BERT_{USE_CASE}"
lstm_model_name = f"CORVEL_LSTM_{USE_CASE}"

# Get latest model versions
bert_latest_model_version = client_local.get_latest_versions(bert_model_name)[0].version
# lstm_latest_model_version = client_local.get_latest_versions(lstm_model_name)[0].version

# COMMAND ----------

# DBTITLE 1,Pull latest models from local MLflow
bert_model_loaded = mlflow.pytorch.load_model(f"models:/{bert_model_name}/{bert_latest_model_version}")
# lstm_model_loaded = mlflow.pytorch.load_model(f"models:/{lstm_model_name}/{lstm_latest_model_version}") # Dummy Model for Now

# OR Assume 'Production' version is the gold standard
# bert_model_loaded = mlflow.pytorch.load_model(f"models:/{bert_model_name}/Production")
# lstm_model_loaded = mlflow.pytorch.load_model(f"models:/{lstm_model_name}/Production") # Dummy Model for Now

print(f"Pulling BERT version {bert_latest_model_version} from Dev Registry")
# print(f"Pulling LSTM version {lstm_latest_model_version} from Dev Registry")

# COMMAND ----------

# DBTITLE 1,Save to temp/global location for creating custom MLflow artifact
# torch.save(bert_model_loaded.state_dict(), artifacts_global["BERT_MODEL_PATH"])
# lstm_model_loaded.save(artifacts_global["LSTM_MODEL_PATH"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set/Create CONDA ENV
# MAGIC Critical step to ensure environment reproducibility

# COMMAND ----------

conda_env = mlflow.pyfunc.get_default_conda_env()
conda_env['dependencies'][-1]['pip'] += [
    f'joblib=={joblib.__version__}',
    f'nltk=={nltk.__version__}',
    f'keras=={keras.__version__}',
    'sklearn',
    f'tensorflow=={tensorflow.__version__}',
    'torch==1.9.1',
    f'transformers=={transformers.__version__}'
    ]
conda_env

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Custom PythonModel Wrapper
# MAGIC Create custom python model with `context` artifacts

# COMMAND ----------

# DBTITLE 1,Define Wrapper Classes
PRE_TRAINED_MODEL_NAME = dbutils.widgets.get("PRE_TRAINED_MODEL_NAME")

class SentimentClassifier(nn.Module):
    """
    Wrapper on top of BERT model
    """
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

class DocTypeWrapper(mlflow.pyfunc.PythonModel):
    """
    Main custom model class
    """
    ###############
    # Property
    ###############
    device :str = None
    bert_model = None
    lstm_model = None
    label_encoder = None
    tokenizer = None
    RANDOM_SEED : int = 42
    MAX_LEN : int = 200
    OVERLAP : int = 50
    NO_CLASS_NAMES : int = 25
    NUM_FEATURES : int = 768   
  
    #lstm settings
    BATCH_SIZE_VAL : int = 1
    BATCHES_PER_EPOCH_VAL: int = 1
    
    # Artifacts global uri/location
    artifacts_global : dict = artifacts_global
    
    def load_context(self, context):
        '''
        this section of code is tasked with setting up the model.
        '''
        
        np.random.seed(self.RANDOM_SEED)
        torch.manual_seed(self.RANDOM_SEED)

        import os
        from nltk.corpus import stopwords

        # need to set up GPU
        self.device = 'cpu'
        
        # Loading all necessary models
        self.tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME) # define the tokenizer
        self.bert_model = SentimentClassifier(self.NO_CLASS_NAMES) # load BERT model
        if context:
            # At Runtime/Inference
            bert_path = os.path.abspath(context.artifacts["BERT_MODEL_PATH"])
            label_encoder_path = os.path.abspath(context.artifacts["LABEL_ENCODER_PATH"])
            lstm_path = os.path.abspath(context.artifacts["LSTM_MODEL_PATH"])
            nltk_data_path = os.path.abspath(context.artifacts["NLTK_DATA_PATH"])

        else:
            # During training/interactive 
            bert_path = self.artifacts_global['BERT_MODEL_PATH']
            label_encoder_path = self.artifacts_global["LABEL_ENCODER_PATH"]
            lstm_path = self.artifacts_global["LSTM_MODEL_PATH"]
            nltk_data_path = self.artifacts_global["NLTK_DATA_PATH"]
            
        print(f'start loading bert model')
        self.bert_model = self.bert_model.to(self.device) # send the model to device
        self.bert_model.load_state_dict(torch.load(bert_path, map_location=torch.device('cpu')), strict=False)
        print(f'end loading bert model')
        
        print(f'start loading label encoder')
        self.label_encoder = joblib.load(label_encoder_path) # load encoder from file
        print(f'end loading encoder')
        
        print(f'start loading lstm model')
        self.lstm_model = load_model(lstm_path) # load LSTM model
        print(f'end loading lstm model')
        
        print(f'Download & set NLTK/stopwords data')
        nltk.data.path.append(nltk_data_path)
        nltk.download('stopwords', download_dir=nltk_data_path)
        os.environ["NLTK_DATA"] = nltk_data_path
        self._stop_words = set(stopwords.words('english'))

    def predict(self, context, model_inputDF):
        '''
        Pass in a document ocr text and the model will try and determine the type, along with accuracy.
        
        TO-DO:
        * VECTORIZE THIS FUNCTION TO ACCEPT a pandas dataframe by default in the Future
        
        '''
        start_time = time.time()
        sentence = self.clean_text(model_inputDF.iloc[0][0]) # Extract first element (string) - WORKAROUND FOR NOW
        samples = self.get_split(sentence)
        samples_torched = [self.input_fn(t, self.tokenizer) for t in samples]
        output_vecs = self.bert_predict(samples_torched) # this is single element - specific to bert

        res = torch.cat(output_vecs).cpu().numpy() # returns an array of array[int]

        # lstm predict will return percentages per category
        preds_lstm = self.lstm_model.predict(self.val_generator(res), steps = self.BATCHES_PER_EPOCH_VAL)[0]
        pred_prob = max(preds_lstm)
        # get index where preds_lstm is value
        pred_label = np.where(preds_lstm==pred_prob) 
        pred_label_new = self.label_encoder.inverse_transform(pred_label)[0]
        end_time = time.time()

        return pd.DataFrame({'Label':[pred_label_new.upper()], 
                             'Score':[float(pred_prob)], 
                             'ElapsedTime':[int((end_time-start_time) * 1000)]
                            })

    def clean_text(self, sample):
        '''
        custom text cleaning code.
        '''
        sample = sample.lower()
        sample = re.sub('[,\.!?]', '', sample)
        sample = re.sub('[^a-zA-Z]', ' ', sample)
        sample = re.sub(r'\s+',' ', sample)
        sample = sample.replace('corvel scan date', '')
        sample = ' '.join([word for word in sample.split() if word not in self._stop_words])

        return sample
  
    def get_split(self, text1):
        '''
        splits word.
        '''
        l_total = []
        l_parcial = []
        if len(text1.split())//150 >0:
            n = len(text1.split())//150
        else: 
            n = 1
        for w in range(n):
            if w == 0:
                l_parcial = text1.split()[:self.MAX_LEN]
                l_total.append(" ".join(l_parcial))
            else:
                l_parcial = text1.split()[w*150:w*150 + self.MAX_LEN]
                l_total.append(" ".join(l_parcial))
        return l_total
  
    def bert_predict(self, samples):
        '''
        bert model prediction method
        note: need to revisit to se if we can adapt to pass across a dataframe instead of singular predict
        '''  
        self.bert_model.eval() # this should be done on initialization.
        output_vectors = []
        for s in samples:

            input_ids = torch.unsqueeze(s['input_ids'],0).to(self.device)
            attention_mask = torch.unsqueeze(s['attention_mask'],0).to(self.device)
            with torch.no_grad():
                _,pooled_outputs = self.bert_model.bert(input_ids=input_ids,attention_mask=attention_mask)

            output_vectors.append(pooled_outputs)
        return output_vectors
  
    def input_fn(self, sentence, tokenizer): 
        '''
        generate json of torch.Tensor properties : input_ids, attention_mask
        '''  
        encoding = tokenizer.encode_plus(
                        sentence,
                        add_special_tokens=True,
                        max_length=self.MAX_LEN,
                        return_token_type_ids=False,
                        pad_to_max_length=True,
                        return_attention_mask=True,
                        truncation=True,
                        return_tensors='pt',
                        )

        return {"input_ids":encoding['input_ids'].flatten(),"attention_mask":encoding['attention_mask'].flatten()}
  
    def val_generator(self, sample):
        '''
        returns a generator. sample is a numpy.ndarray
        '''
        x_list = sample
        # Generate batches
        for b in range(self.BATCHES_PER_EPOCH_VAL):
            longest_index = (b + 1) * self.BATCH_SIZE_VAL - 1
            timesteps = len(max(sample[:(b + 1) * self.BATCH_SIZE_VAL][-31:], key=len))
            x_train = np.full((self.BATCH_SIZE_VAL, timesteps, self.NUM_FEATURES), -99.)
            for i in range(self.BATCH_SIZE_VAL):
                li = b * self.BATCH_SIZE_VAL + i
                x_train[i, 0:len(x_list[li]), :] = x_list[li]
            yield x_train

# COMMAND ----------

# MAGIC %md
# MAGIC ### Local Functionnal Testing
# MAGIC **Predict on single pandas dataframe for schema validation**

# COMMAND ----------

# DBTITLE 1,Instantiate Default Model
trained_wrappedModel = DocTypeWrapper()

# Load baseline label encorder and pre-trained BERT, LSTM models from global URI/Location
trained_wrappedModel.load_context(None)

# COMMAND ----------

test_df = pd.DataFrame({'Request':['hello world']})
out_df = trained_wrappedModel.predict(None, test_df)
display(out_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log model to MLflow's Central Model Registry

# COMMAND ----------

# DBTITLE 1,Infer Signature first
from mlflow.models.signature import infer_signature
signature = mlflow.models.infer_signature(
    test_df,
    out_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set MLflow to point to desired server for pushing model
# MAGIC _using the [Personnal Access Tokens](https://docs.databricks.com/dev-tools/api/latest/authentication.html#token-management) created_
# MAGIC * DEV ('databricks')
# MAGIC * QA
# MAGIC * PROD ('Central Model Registry PAT')

# COMMAND ----------

# DBTITLE 1,Point to desired  MLFlow Registry
registry_uri = dbutils.widgets.get("MLFLOW_URI_PAT") # 'databricks' for local
mlflow.set_registry_uri(registry_uri)

# COMMAND ----------

# DBTITLE 1,Log new version
model_name = dbutils.widgets.get("MODEL_NAME")
with mlflow.start_run(run_name=f"{USE_CASE}_{model_name}"):
    
    mlflow.pyfunc.log_model(
        "model",
        python_model=DocTypeWrapper(),           
        conda_env=conda_env,    
        registered_model_name=model_name,
        artifacts=trained_wrappedModel.artifacts_global,
        signature= signature
      )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Local API Testing _(OPTIONAL)_
# MAGIC Enable Model Serving and test payload/response and/or use [test notebook](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/1390462475107692/command/1390462475107722)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Request Transition to new stage:
# MAGIC `None`
# MAGIC `Archived`
# MAGIC `Staging`
# MAGIC `Production`
# MAGIC 
# MAGIC Pick last version and request transition to stage parameter
# MAGIC 
# MAGIC **Assumes Webhooks were already created [here](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/1390462475087728/command/1390462475087764)**

# COMMAND ----------

# DBTITLE 1,Helper call for model transition request
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

def request_transition(model_name, version, stage, mlflow_host_url="", token=None):
    transition_request = {
        'name': model_name,
        'version': version,
        'stage': stage,
        'archive_existing_versions': 'true'
    }
    
    return mlflow_call_endpoint('transition-requests/create', 'POST', transition_request, mlflow_host_url, token)

# COMMAND ----------

token = dbutils.secrets.get(scope="ml-scope", key="dp-token") # PAT for Central Model Registry
# token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get() # Local

request_transition(
    model_name=model_name,
    version=latest_model_version,
    stage=dbutils.widgets.get("stage"),
    mlflow_host_url=dbutils.widget.get("MLFLOW_HOST_URL"),
    token=token
)

# COMMAND ----------

# MAGIC %md
# MAGIC **This will trigger the following sequential actions:**
# MAGIC 1. [Validation job](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#job/330465) has kicked
# MAGIC 2. Receive slack notification if validation was succesful
# MAGIC 3. [AzureDevOps](add link) pipeline [job](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#job/332018) has kicked
# MAGIC 4. [MLflow artifact was pushed to AKS](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#job/332066)
