# Databricks notebook source
# MAGIC %md
# MAGIC # End-To-End MLOps Text Classification example using transfer learning and MLflow
# MAGIC 
# MAGIC **PART 3/4 - Model Deployment **
# MAGIC * Log custom artifact to MLflow
# MAGIC * Expose as REST/API
# MAGIC * _Push to AKS_

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
dbutils.widgets.text("INPUT_DATA", "/mnt/oetrta/diganparikh/corvel/corvel_contents/iter6.14_pocsample.csv", "Path to input OCR + label file")
dbutils.widgets.text("USE_CASE", 'symbeo_doctyping', "Use-Case Name")
dbutils.widgets.text("PRE_TRAINED_MODEL_NAME","emilyalsentzer/Bio_ClinicalBERT", "Pre-Trained BERT model to load")

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

# DBTITLE 1,Pull models from MLflow and save to temp/global artifact location for creating custom artifact
# bert_model_name = f"CORVEL_BERT_{USE_CASE}"
# lstm_model_name = f"CORVEL_LSTM_{USE_CASE}"
model_name = 'DocType_PyFunc_Test'

bert_model_loaded = mlflow.pytorch.load_model(f"models:/{model_name}/Production")
#lstm_model_loaded = mlflow.pytorch.load_model(f"models:/{lstm_model_name}/Production")

torch.save(bert_model_loaded.state_dict(), artifacts_global["BERT_MODEL_PATH"])
lstm_model_loaded.save(artifacts_global["LSTM_MODEL_PATH"])

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
# MAGIC ### Test custom model
# MAGIC **Predict on single pandas dataframe**

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
# MAGIC ## Log model to MLflow Model Registry

# COMMAND ----------

# DBTITLE 1,Infer Signature first
from mlflow.models.signature import infer_signature
signature = mlflow.models.infer_signature(
    test_df,
    out_df)

# COMMAND ----------

model_name = "DocType_PyFunc_Test"
with mlflow.start_run(run_name='DocType_PyFunc'):
    
    mlflow.pyfunc.log_model(
        model_name,
        python_model=DocTypeWrapper(),           
        conda_env=conda_env,    
        registered_model_name="DocType_PyFunc_Test",
        artifacts=trained_wrappedModel.artifacts_global,
        signature= signature
      )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transition Stages
# MAGIC `None`
# MAGIC `Archived`
# MAGIC `Staging`
# MAGIC `Production`
# MAGIC 
# MAGIC Pick last version and mark as 'Production'

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
latest_model = client.search_model_versions(f"name='{model_name}'")
latest_model_version = latest_model[0].version

client.transition_model_version_stage(
    name=model_name,
    version=int(latest_model_version),
    stage="Production",
    archive_existing_versions=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable Model Serving
# MAGIC [Here](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#mlflow/models/DocType_PyFunc_Test/serving)
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

def score_model(data: pd.DataFrame, model_name: str, version: str):
  token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
  instance = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('browserHostName')
  url = f'https://{instance}/model/{model_name}/{version}/invocations'
  headers = {'Authorization': f'Bearer {token}'}
  data_json = data.to_dict(orient='split')
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return pd.DataFrame(response.json())

test_df = pd.DataFrame({'Request':['hello world']})

# COMMAND ----------

model_name = "DocType_PyFunc_Test"
latest_model_version = "1"
out_df = score_model(test_df, model_name, "1")

# COMMAND ----------

display(out_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load from MLflow and perform inference against pandas DataFrame
# MAGIC for functionnal testing purposes

# COMMAND ----------

# DBTITLE 0,inferencing against pandas df... works fine but goes through the nltk installation process
test_df = pd.DataFrame({'Request':["City of Fort Matthews-EC-TT  CarelQ  Automated to Georgia - Site 11  A CORVEL NETWORK  CarelQ Transportation  Invoice Date: 01/01/2020  Corvel Scan Date: 02/02/2020  Transportation /Translation Invoice :  123456  Account Group:  Patient  Claim #  Date of Service  ItemId  Item Name  Quantity  Rate  Charge  Elden , Isiah, Rili  1111-WC-  1-12-18  TRANS-AMB ROUND  TRANSPORTATION- ROUND TRIP TOTAL  $336.07  $336.07  18-0000088  TRIP  9AM FR 121 HELM ST FORT Matthews CA TO ADVANCED  Winterhill Lodge 2002 361 BALTHAM ST PORT CHARLOTTE CA  FORT MAT  John Wick  1111 -WC  05/12/2015  TRANS- AMB WAIT  TRANSPORTATION - WAIT TIME, AMBULATORY  $49.39  $249.55  18-0008888  TIME  Total Charges :  $895.62  This is not a medical bill . Thank you for your business !  Make payable to:  (321)555-4600  CarelQ  PO Box 1000 S. Main East   TIN: (123) 555 - 4148", "Hi This is a Test, what is the label?"]})

import mlflow
logged_model = f"models:/{model_name}/Production'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(test_df)

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
with open("deployment_config.json", "w") as outfile:
    outfile.write(json_object)

# COMMAND ----------

from mlflow.deployments import get_deploy_client

# set the tracking uri as the deployment client
client = get_deploy_client(mlflow.get_tracking_uri())
model_uri = f"models:/{model_name}/Production'

# set the model path 
model_path = "model"

# set the deployment config
deployment_config_path = "deployment_config.json"
test_config = {'deploy-config-file': deployment_config_path}

# define the model path and the name is the service name
# the model gets registered automatically and a name is autogenerated using the "name" parameter below 
client.create_deployment(model_uri=model_uri,
                         config=test_config,
                         name=f"{USE_CASE}-aks-deployment")
