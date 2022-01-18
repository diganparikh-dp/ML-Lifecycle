# Databricks notebook source
# MAGIC %md
# MAGIC # End-To-End MLOps Text Classification example using transfer learning and MLflow
# MAGIC _demo for Corvel_
# MAGIC 
# MAGIC **PART 2b/4 - Model Training ** (_WORK_IN_PROGRESS_)
# MAGIC * Fine-Tune an LSTM model and log artifacts to MLflow/Model Registry
# MAGIC * Store outputs to DELTA

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
dbutils.widgets.text("MAX_LEN", "200", "Max Text Length (BERT preprocessing)")
dbutils.widgets.text("NO_LSTM_RUNS", "2", "Number of Runs (per training iteration)")
dbutils.widgets.text("LSTM_EPOCHS", "5", "Number of Epochs (training)")
dbutils.widgets.text("dbName","Corvel_E2E_Demo","Database to use")
dbutils.widgets.text("PRE_TRAINED_MODEL_NAME","emilyalsentzer/Bio_ClinicalBERT", "Pre-Trained BERT model to load")
dbutils.widgets.text("RANDOM_SEED","42", "Random Seed")
dbutils.widgets.text("BATCH_SIZE","4", "Batch Size (BERT preprocessing)")

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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

from collections import defaultdict

import re
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define database to use and artifacts location

# COMMAND ----------

SAVE_DIR = dbutils.widgets.get("SAVE_DIR")
USE_CASE = dbutils.widgets.get("USE_CASE")
artifacts_global = {
    "LABEL_ENCODER_PATH" : os.path.join(SAVE_DIR,f"{USE_CASE}_label_encoder.pkl"),
    "LSTM_MODEL_PATH" : os.path.join(SAVE_DIR,f"{USE_CASE}_lstm.h5")
}
dbName = dbutils.widgets.get("dbName")
spark.sql(f"USE {dbName}")
print(f"Using {dbName} database")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read clean/Gold dataset
# MAGIC From `CORVEL_GOLD` table

# COMMAND ----------

trainingDF = spark.read.table("CORVEL_GOLD")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Label Encoding Classes
# MAGIC Using `spark.ml.feature`'s `StringIndexer`

# COMMAND ----------

from pyspark.ml.feature import IndexToString, StringIndexer
mlflow.autolog(disable=True)
enc_spark = StringIndexer(inputCol="label", outputCol="label_new", handleInvalid='skip').fit(trainingDF)
class_names = enc_spark.labels
trainingDF = enc_spark.transform(trainingDF) \
                      .drop("label") \
                      .withColumnRenamed("label_new", "label")

# COMMAND ----------

# MAGIC %md
# MAGIC Save label encoder as sklearn encoder for real-time "model-serving" purposes (OPTIONAL)

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

enc_sk = LabelEncoder().fit(np.array(class_names))

# Save label encoder
with open(artifacts_global["LABEL_ENCODER_PATH"],'wb') as f:
    pickle.dump(enc_sk,f)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper methods/classes used for BERT pre-processing

# COMMAND ----------

# DBTITLE 1,BERT Helper Classes
class GPReviewDataset(Dataset):
    '''
    Dataset class which stores features in BERT input format.
    
    Args:
        doc:       
        targets:   true labels converted into numeric feature through sklearn LabelEncoder
        tokenizer: BertTokenizer or AutoTokenizer object
        max_len:   maximum input length of and input example used for padding and truncation
        
    Returns:
    '''
    
    def __init__(self, 
                 doc, 
                 targets, 
                 tokenizer, 
                 max_len:int):
        
        self.doc = doc
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.doc)
  
    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        doc = str(self.doc[int(item)])
        target = self.targets[int(item)]

        encoding = self.tokenizer.encode_plus(
                    doc,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=False,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    truncation=True, #  This line is required to ignore truncation warnings otherwise defaults to longest first strategy, also results with and without are exactly same
                    return_tensors='pt',
                    )
        
        return {
            'doc_text': doc,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
            }

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes_in, PRE_TRAINED_MODEL_NAME_in):
        super(SentimentClassifier, self).__init__()
        
        if PRE_TRAINED_MODEL_NAME_in=='bert-base-uncased':
            self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,return_dict=False)
        elif PRE_TRAINED_MODEL_NAME_in=='bert-base-cased':
            self.bert = BertModel.from_pretrained("bert-base-cased",return_dict=False)
        elif PRE_TRAINED_MODEL_NAME_in=='emilyalsentzer/Bio_ClinicalBERT':
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",return_dict=False)
        elif PRE_TRAINED_MODEL_NAME_in=='emilyalsentzer/Bio_Discharge_Summary_BERT':
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT",return_dict=False)

        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes_in)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)
      
def defining_bert_tokenizer(PRE_TRAINED_MODEL_NAME:str):
    '''
    Tokenizer which pre-processes text for BERT model.
    
    Args:
        PRE_TRAINED_MODEL_NAME: name of pre-trained BERT model
    
    Returns:
        BertTokenizer or AutoTokenizer object depending on PRE_TRAINED_MODEL_NAME,
        both have the same behavior
    '''
    
    if PRE_TRAINED_MODEL_NAME=='bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif PRE_TRAINED_MODEL_NAME=='bert-base-cased':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    elif PRE_TRAINED_MODEL_NAME=='emilyalsentzer/Bio_ClinicalBERT':
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    elif PRE_TRAINED_MODEL_NAME=='emilyalsentzer/Bio_Discharge_Summary_BERT':
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")
    return tokenizer

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper methods for LSTM

# COMMAND ----------

# DBTITLE 1,Helper methods to calculate vectors for LSTM
def build_lstm_feature_vectors(df_train_in, df_val_in, df_test_in, MAX_LEN_in, BATCH_SIZE_in, bert_model_in, device_in):
    """
    Helper method
    """
    

# COMMAND ----------

def concat_vecs_for_lstm_light(df_in:pd.DataFrame):
    '''
    For LSTM input, BERT output vectors need to be aggregated together to represent original dataset.
    
    Args:
        df_new:pd.DataFrame
        df:pd.DataFrame
        ind_l:list
    
    Returns:
        np.ndarray
    '''
    
    # Reorder
    df_in_ = df_in.sort_values(by=['index'])
    
    # Stack
    return np.vstack(tuple(zip(df_in[:,'emb'])))

# COMMAND ----------

# Cross-Validation
TRAIN_TEST_SPLIT = 0.1
silverDF = spark.read.table("CORVEL_SILVER")
df_train, df_test = train_test_split(silverDF.select("dcn", "page", "text_chunk", "index", "label").toPandas(), test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
bert_model_name = f"CORVEL_BERT_{USE_CASE}"
bert_model_loaded = mlflow.pytorch.load_model(f"models:/{bert_model_name}/Production")

# COMMAND ----------

# MAGIC %md
# MAGIC ## LEgacy Calculation of LSTM vectors

# COMMAND ----------

DATA_COLUMN = 'text_processed'
LABEL_COLUMN = 'label'
OVERLAP = 50

# Cross-Validation
TRAIN_TEST_SPLIT = 0.1
silverDF = spark.read.table("CORVEL_SILVER")
silverDF = enc_spark.transform(silverDF) \
                      .drop(LABEL_COLUMN) \
                      .withColumnRenamed(f"{LABEL_COLUMN}_new", LABEL_COLUMN)
display(silverDF)

# COMMAND ----------

def create_data_loader(df:pd.DataFrame, 
                       tokenizer,
                       max_len:int, 
                       batch_size:int):
    '''
    Data loader which will feed input features to BERT during training process.
    
    Args:
        df:         dataset
        tokenizer:  transformers.AutoTokenizer object
        max_len:    maximum input length of a document
        batch_size: input batch size for training
    
    Returns:
        dataloader: DataLoader
    '''
    
    ds = GPReviewDataset(
        #doc=df.encounter_text.to_numpy()
        doc=np.array(df.iloc[:,0]),# Text
        #targets=df.doc_class.to_numpy()
        targets=np.array(df.iloc[:,1]),# Label
        tokenizer=tokenizer,
        max_len=max_len
      )
    
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        #num_workers=4
        num_workers=0
        )
    
    return dataloader

def concat_vecs_for_lstm(df_new:pd.DataFrame,
                         df:pd.DataFrame,
                         ind_l:list):
    '''
    For LSTM input, BERT output vectors need to be aggregated together to represent original dataset.
    
    Args:
        df_new:pd.DataFrame
        df:pd.DataFrame
        ind_l:list
    
    Returns:
        train_l_final:list
        label_l_final:list
    '''
    
    aux = -1
    len_l = 0
    train_x = {}
    tr_emb = list(df_new.emb)
    
    for l, emb in zip(ind_l, tr_emb):
        if l in train_x.keys():
            train_x[l] = np.vstack([train_x[l], emb])
        else:
            train_x[l] = [emb]

    train_l_final = []
    label_l_final = []
    
    for k in train_x.keys():
        train_l_final.append(train_x[k])
        label_l_final.append(df.loc[k]['label'])
    
    return train_l_final, label_l_final

def get_bert_vectors(data_loader:DataLoader, 
                     model:SentimentClassifier, 
                     device):
    '''
    Function to get BERT vector representation from the models' last hidden state.
    
    Args:
        data_loader:DataLoader
        model:SentimentClassifier
        device:torch.device object if CUDA or "cpu" if GPU is not available
    
    Returns:
        res:np.ndarray
    '''
    # testing vectors
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    outputs = []

    with torch.no_grad():
        for d in data_loader:

            texts = d["doc_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            _,pooled_outputs = model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
            )

            review_texts.extend(texts)
            #predictions.extend(preds)
            #prediction_probs.extend(probs)
            outputs.append(pooled_outputs)
            real_values.extend(targets)

    res = torch.cat(outputs).cpu().numpy()
    return res

def get_split(text1:str,
              maxlen:int,
              overlap:int,
              full_split=False):
    '''
    Splits a piece of text into chunks of length=maxlen. 
    Each subsequent chunk has an overlap with the previous chunk as specified by overlap paramter.
    To obtain the latest shorter chunk, provide full_split=True.
    
    Args:
        text1:str
        maxlen:int
        overlap:int
        full_split=False
    
    Returns:
        l_total:list
    '''
    
    l_total = []
    l_parcial = []
    if len(text1.split())//(maxlen-overlap) > 0:
        if(full_split):
            n = (len(text1.split())//(maxlen-overlap)) + 1 
        else:
            n = len(text1.split())//(maxlen-overlap)
    else: 
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text1.split()[:maxlen]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = text1.split()[w*(maxlen-overlap):w*(maxlen-overlap) + maxlen]
            l_total.append(" ".join(l_parcial))
    return l_total

def expand_chunk_splits(df:pd.DataFrame):
    '''
    Expands equal-length chunks belonging to the same original text piece into
    independent samples for BERT fine-tuning.
    
    Args:
        df:pd.DataFrame
    
    Returns:
        tr_l:list
        lbl_l:list
        ind_l:list
    '''
    
    tr_l = []
    lbl_l = []
    ind_l = []
    for idx,row in df.iterrows():
        for l in row['text_split']:
            tr_l.append(l)
            lbl_l.append(row['label'])
            ind_l.append(idx)
            
    return tr_l, lbl_l, ind_l

# COMMAND ----------

# DBTITLE 1,Legacy calculation
df_train, df_test = train_test_split(silverDF.select(DATA_COLUMN, LABEL_COLUMN).toPandas(), test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
# bert_model_name = f"CORVEL_BERT_{USE_CASE}"
# bert_model_loaded = mlflow.pytorch.load_model(f"models:/{bert_model_name}/Production")

df_train['text_split'] = df_train[DATA_COLUMN].apply(get_split,args=(MAX_LEN,OVERLAP,))
df_test['text_split'] = df_test[DATA_COLUMN].apply(get_split,args=(MAX_LEN,OVERLAP,))
df_val['text_split'] = df_val[DATA_COLUMN].apply(get_split,args=(MAX_LEN,OVERLAP,))

train_l,label_l,index_l = expand_chunk_splits(df_train)
test_l,test_label_l,test_index_l = expand_chunk_splits(df_test)
val_l,val_label_l,val_index_l = expand_chunk_splits(df_val)

df_train_new = pd.DataFrame({DATA_COLUMN:train_l, LABEL_COLUMN:label_l})
df_test_new = pd.DataFrame({DATA_COLUMN:test_l, LABEL_COLUMN:test_label_l})
df_val_new = pd.DataFrame({DATA_COLUMN:val_l, LABEL_COLUMN:val_label_l})

# Prepare data loaders
train_data_loader = create_data_loader(df_train_new, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val_new, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test_new, tokenizer, MAX_LEN, BATCH_SIZE)

test_res = get_bert_vectors(test_data_loader,bert_model_loaded,device)
train_res = get_bert_vectors(train_data_loader,bert_model_loaded,device)
val_res = get_bert_vectors(val_data_loader,bert_model_loaded,device)

df_test_new['emb'] = list(test_res)
df_train_new['emb'] = list(train_res)
df_val_new['emb'] = list(val_res)

# train set
train_l_final, label_l_final = concat_vecs_for_lstm(df_train_new,df_train,index_l)
df_train_new1 = pd.DataFrame({'emb': train_l_final, 'label': label_l_final})

# validation set
val_l_final,vlabel_l_final = concat_vecs_for_lstm(df_val_new,df_val,val_index_l)
df_val_new1 = pd.DataFrame({'emb': val_l_final, 'label': vlabel_l_final})

# test set
test_l_final,tlabel_l_final = concat_vecs_for_lstm(df_test_new,df_test,test_index_l)
df_test_new1 = pd.DataFrame({'emb': test_l_final, 'label': tlabel_l_final})

# COMMAND ----------

df_test_new1.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set (Static/Dynamic) Hyper Parameters

# COMMAND ----------

# DBTITLE 1,BERT & LSTM Settings
# BERT-related
RANDOM_SEED = int(dbutils.widgets.get("RANDOM_SEED"))
PRE_TRAINED_MODEL_NAME = dbutils.widgets.get("PRE_TRAINED_MODEL_NAME")
MAX_LEN = int(dbutils.widgets.get("MAX_LEN"))
BATCH_SIZE = int(dbutils.widgets.get("BATCH_SIZE"))

# LSTM params
USE_MID_LEVEL = True
LSTM_CELLS = 500
NODES_MID_LEVEL = 60
NO_LSTM_RUNS = int(dbutils.widgets.get("NO_LSTM_RUNS"))
LSTM_EPOCHS = int(dbutils.widgets.get("LSTM_EPOCHS"))
NUM_FEATURES = 768

BATCH_SIZE_TRAIN = 14 # batch_size (ADJUSTED TO FIT CURRENT POC DATASET)
BATCHES_PER_EPOCH = 4 #batches_per_epoch (ADJUSTED TO FIT CURRENT POC DATASET)
        
BATCH_SIZE_VAL = 1  # batch_size_val (ADJUSTED TO FIT CURRENT POC DATASET)
BATCHES_PER_EPOCH_VAL = 3 #batches_per_epoch_val (ADJUSTED TO FIT CURRENT POC DATASET)

BATCH_SIZE_TEST = 1  # batch_size_val (ADJUSTED TO FIT CURRENT POC DATASET)
BATCHES_PER_EPOCH_TEST = 4 # batches_per_epoch_val (ADJUSTED TO FIT CURRENT POC DATASET)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = defining_bert_tokenizer(PRE_TRAINED_MODEL_NAME)

# COMMAND ----------

len(df_val_new1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and log runs to MLflow

# COMMAND ----------

# DBTITLE 1,Create/Set MLflow experiment for LSTM training
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
bert_experiment_name = f"/Users/{current_user}/Corvel_LSTM_experiments_test"
# bert_experiment = mlflow.create_experiment(bert_experiment_name) - DO ONCE
bert_experiment = mlflow.get_experiment_by_name(bert_experiment_name)
bert_artifact_path = "LSTM-model"
mlflow.set_experiment(experiment_id=bert_experiment.experiment_id)

# COMMAND ----------

# DBTITLE 1,Enable Auto-Logging
mlflow.autolog() # Auto-Logs all standard hyper-params (for standard models)

# COMMAND ----------

# DBTITLE 1,Initialize Parent Run
with mlflow.start_run(experiment_id=bert_experiment.experiment_id) as parent_run:
    # Log params for parent run
    mlflow.log_params({
        "Parent": True,
        'EPOCH' : EPOCHS,
        'LR': LR,
        'BATCH_SIZE': BATCH_SIZE,
        'PRE_TRAINED_MODEL_NAME': PRE_TRAINED_MODEL_NAME
    })
    mlflow.set_tag("Use-Case", USE_CASE)
    
    # Cross-Validation
    TRAIN_TEST_SPLIT = 0.1
    df_train, df_test = train_test_split(trainingDF.select("text_chunk", "label").toPandas(), test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
    
    # Prepare data loaders
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
    
    # Initialize model and data
    data = next(iter(train_data_loader))
    model = SentimentClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
    model = model.to(device)

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False) # here also 3e-5 and 5e-5 worth trying
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    # Log each children/epoch run
    for epoch in range(1, EPOCHS+1):
        with mlflow.start_run(run_name='BERT EPOCH', nested=True) as child_run:
            # Log custom params
            mlflow.log_params({
                'EPOCH': epoch,
                'LR': LR,
                'BATCH_SIZE': BATCH_SIZE,
                'PRE_TRAINED_MODEL_NAME': PRE_TRAINED_MODEL_NAME
            })

            # Train
            train_acc, train_loss = train_epoch(model,train_data_loader,loss_fn, optimizer, device, scheduler, len(df_train))

            # Validate
            val_acc, val_loss = eval_model(model,val_data_loader,loss_fn, device, len(df_val))

            # Log custom metrics
            mlflow.log_metrics({
                "train_acc": float(train_acc),
                "train_loss": float(train_loss),
                "val_acc": float(val_acc),
                "val_loss": float(val_loss)
            })
            
    # Evaluate on Test set
    test_acc, _ = eval_model(model,test_data_loader,loss_fn,device,len(df_test))
    mlflow.log_metric("test_acc", float(test_acc))

    # Log state dictionnary
    mlflow.pytorch.log_state_dict(model.state_dict(), bert_artifact_path)

    # Plot confusion matrix
    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model,test_data_loader,device)
    conf_m = pd.DataFrame([enc_sk.inverse_transform(y_test.numpy()),enc_sk.inverse_transform(y_pred.numpy())]).T
    conf_m.columns = ['y_Actual','y_Predicted']
    confusion_matrix = pd.crosstab(conf_m['y_Predicted'],conf_m['y_Actual'], rownames=['Predicted'], colnames=['Actual'])
    fig = plt.figure(figsize=(15,10))
    sns.heatmap((confusion_matrix/np.sum(confusion_matrix)).T, annot=True,fmt='.0%', cmap='Blues');
    plt.title('BERT confusion matrix (Test Set)');
    mlflow.log_figure(fig, "confusion_matrix.png")

    # Log model to MLflow
    mlflow.pytorch.log_model(model, bert_artifact_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Training progress can be tracked interactively using [Tensorboard](https://databricks.com/tensorflow/visualisation)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Search best run for BERT
# MAGIC Using MLflow client API to find best run and load artifact

# COMMAND ----------

from mlflow.entities import ViewType
from mlflow.tracking.client import MlflowClient
client = MlflowClient()

best_bert_model = client.search_runs(
  experiment_ids=bert_experiment.experiment_id,
  filter_string="metrics.test_acc >= 0",
  run_view_type=ViewType.ACTIVE_ONLY,
  max_results=1,
  order_by=["metrics.test_acc DESC"]
)[0]

best_bert_score = best_bert_model.data.metrics["test_acc"]
print(f'Test Accuracy of Best BERT Run: {best_bert_score}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Promote Best Run to Registry
# MAGIC Mark reference model as 'Production'
# MAGIC 
# MAGIC (this piece can be automated using **Webhooks** &/or [Multi-Task-Jobs](https://docs.databricks.com/data-engineering/jobs/index.html))
# MAGIC 
# MAGIC ```
# MAGIC artifact_path = "BERT-model" (default = "model")
# MAGIC model_name = "CORVEL-BERT"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/{artifact_path}"
# MAGIC OR
# MAGIC model_uri = f"models:/{model_name}/Version_or_Stage
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```

# COMMAND ----------

bert_model_name = f"CORVEL_BERT_{USE_CASE}"
bert_best_model_uri = f"runs:/{best_bert_model.info.run_id}/{bert_artifact_path}"
bert_model_details = mlflow.register_model(bert_best_model_uri, bert_model_name)
client.transition_model_version_stage(
    name=bert_model_name,
    version=int(bert_model_details.version),
    stage="Production",
    archive_existing_versions=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load model from MLflow

# COMMAND ----------

# DBTITLE 0,Load MLflow artifact 
# bert_model_loaded = mlflow.pytorch.load_model(bert_best_model_uri, dst_path="/dbfs/Users/amine.elhelou@databricks.com/Customers/Corvel")
bert_model_loaded = mlflow.pytorch.load_model(f"models:/{bert_model_name}/Production")

# Save binary to global artifacts location (OPTIONAL)
torch.save(bert_model_loaded.state_dict(), artifacts_global["BERT_MODEL_PATH"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fine-Tune LSTM model
# MAGIC _In_Progress_

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Helper Functions and setup parameters

# COMMAND ----------

import tensorflow #as tf
from tensorflow.keras import Sequential, optimizers, layers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Masking, Embedding, Dense, Input, concatenate, Layer, Lambda, Dropout, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import Sequence

call_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.95, patience=3, verbose=2,
                            mode='auto', min_delta=0.01, cooldown=0, min_lr=0)

def define_lstm_model(lstm_cells:int,
                      nodes_mid_level:int,
                      use_mid_level:bool,
                      class_no:int,
                      num_features:int):
    '''
    Definition of Keras LSTM model with Tensorflow backend.
    
    Args:
        lstm_cells:int,
        nodes_mid_level:int,
        use_mid_level:bool,
        class_no:int,
        num_features:int
    
    Returns:
        lstm_model:keras.models.Model
    '''
    
    text_input = Input(shape=(None,num_features,), dtype='float32', name='text')
    l_mask = layers.Masking(mask_value=-99.)(text_input)
    encoded_text = layers.LSTM(lstm_cells,)(l_mask)

    if use_mid_level:
        out_dense = layers.Dense(nodes_mid_level, activation='relu')(encoded_text)
        out = layers.Dense(class_no, activation='softmax')(out_dense)
    else:
        #out = layers.Dense(class_no, activation='softmax')(encoded_text)
        out = tensorflow.layers.Dense(class_no, activation='softmax')(encoded_text)
    
    #lstm_model = keras.models.Model(text_input, out)
    lstm_model = tensorflow.keras.models.Model(text_input, out)
    lstm_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['acc'])
    
    return lstm_model

# COMMAND ----------

# MAGIC %md
# MAGIC #### Infer embeddings and push to Feature-Store
# MAGIC _WORK_IN_PROGRESS_

# COMMAND ----------

# Create batch generators for LSTM model -> need to automate this part -> calculate batch_size and batches_per_epoch
# automatically
# print_or_log('Training LSTM model',logger, log=True, do_print=True)

res = [0,0]
best_model = None
best_acc = 0
lstm_models = []
i = 0

for i in range(NO_LSTM_RUNS):
    i = i + 1
#     print_or_log('LSTM run {}',logger, *[i], log=True, do_print=True)

    num_sequences = len(df_train_new1['emb'].to_list())

#     assert BATCH_SIZE_TRAIN * BATCHES_PER_EPOCH == num_sequences

    def train_generator(df):
        x_list = df['emb'].to_list()
        y_list =  df.label.to_list()
        # Generate batches
        while True:
            for b in range(BATCHES_PER_EPOCH):
                longest_index = (b + 1) * BATCH_SIZE_TRAIN - 1
                timesteps = len(max(df['emb'].to_list()[:(b + 1) * BATCH_SIZE_TRAIN][-BATCH_SIZE_TRAIN:], key=len))
                x_train = np.full((BATCH_SIZE_TRAIN, timesteps, NUM_FEATURES), -99.)
                y_train = np.zeros((BATCH_SIZE_TRAIN,  1))
                for i in range(BATCH_SIZE_TRAIN):
                    li = b * BATCH_SIZE_TRAIN + i
                    x_train[i, 0:len(x_list[li]), :] = x_list[li]
                    y_train[i] = y_list[li]

                yield x_train, y_train


    num_sequences_val = len(df_val_new1['emb'].to_list())

#     assert BATCH_SIZE_VAL * BATCHES_PER_EPOCH_VAL == num_sequences_val

    def val_generator(df):
        x_list = df['emb'].to_list()
        y_list =  df.label.to_list()
        # Generate batches
        while True:
            for b in range(BATCHES_PER_EPOCH_VAL):
                longest_index = (b + 1) * BATCH_SIZE_VAL - 1
                timesteps = len(max(df['emb'].to_list()[:(b + 1) * BATCH_SIZE_VAL][-31:], key=len))
                x_train = np.full((BATCH_SIZE_VAL, timesteps, NUM_FEATURES), -99.)
                y_train = np.zeros((BATCH_SIZE_VAL,  1))
                for i in range(BATCH_SIZE_VAL):
                    li = b * BATCH_SIZE_VAL + i
                    x_train[i, 0:len(x_list[li]), :] = x_list[li]
                    y_train[i] = y_list[li]

                yield x_train, y_train

    lstm_model = define_lstm_model(LSTM_CELLS,NODES_MID_LEVEL,USE_MID_LEVEL,len(class_names),NUM_FEATURES)
    lstm_model.fit(train_generator(df_train_new1), steps_per_epoch=BATCHES_PER_EPOCH, epochs=LSTM_EPOCHS,
                        validation_data=val_generator(df_val_new1), 
                        validation_steps=BATCHES_PER_EPOCH_VAL, callbacks =[call_reduce])

    # evaluate LSTM model

    num_sequences_val = len(df_test_new1['emb'].to_list())

#     assert BATCH_SIZE_VAL * BATCHES_PER_EPOCH_VAL == num_sequences_val

    res = lstm_model.evaluate_generator(val_generator(df_test_new1), steps = BATCHES_PER_EPOCH_VAL)
#     print_or_log('Test loss: {} Accuracy: {}',logger, *[res[0],res[1]], log=True, do_print=True)

    # temp change
    lstm_models.append(lstm_model)

    curr_acc = res[1]

#     if (curr_acc > best_acc):
#         best_acc = curr_acc
#         min_loss = res[0]
#         best_model = lstm_model

# lstm_model = best_model #lstm_models[0]
# run.log('LSTM test loss',float(min_loss))
# run.log('LSTM test accuracy',float(best_acc))

# with open(LSTM_CONFIG_PATH, 'w') as f:
#     with redirect_stdout(f):
#         lstm_model.summary()

# Save LSTM model
# lstm_model.save(LSTM_SAVE_PATH)  # creates a HDF5 file 'my_model.h5'
# print_or_log('LSTM model saved to {}',logger, *[LSTM_SAVE_PATH], log=True, do_print=True)

# Get LSTM predicitions for test set
# print_or_log('Evaluating LSTM model',logger, log=True, do_print=True)

preds_lstm = lstm_model.predict(val_generator(df_test_new1),steps = BATCHES_PER_EPOCH_VAL)
pred_labels = []

for pred in preds_lstm:
    pred_labels.append(np.where(pred==max(pred))[0][0])

# Produce classifiation report
# print_or_log('Producing classification report',logger, log=True, do_print=True)

names = enc_sk.inverse_transform(list(range(len(class_names))))
lstm_class_report = classification_report(df_test_new1.label, pred_labels,
                            labels=list(range(len(class_names))),
                            target_names=names,digits=4,output_dict=True)


lstm_report_df = pd.DataFrame.from_dict(lstm_class_report).T.reset_index()
lstm_report_df.columns = ['Doctype/agg','Precision','Recall','F1-score','Support']
flat_lstm_class_report = {}

for col in lstm_report_df.columns:
    flat_lstm_class_report[col] = list(lstm_report_df[col].values)

# run.log_table('LSTM classification report',flat_lstm_class_report)

# Plot confusion matrix
# print_or_log('Producing confusion matrix',logger, log=True, do_print=True)

conf_m = pd.DataFrame([enc.inverse_transform(df_test_new1.label),enc.inverse_transform(pred_labels)]).T
conf_m.columns = ['y_Actual','y_Predicted']

confusion_matrix = pd.crosstab(conf_m['y_Predicted'],conf_m['y_Actual'], rownames=['Predicted'], colnames=['Actual'])
plt.figure(figsize=(15,10))
sns.heatmap((confusion_matrix/np.sum(confusion_matrix)).T, annot=True,fmt='.0%', cmap='Blues')
plt.title('LSTM confusion matrix')
# plt.savefig(LSTM_CONF_M_FILE)

# COMMAND ----------

print_or_log('Doctyping model for {} finished',logger, *[USE_CASE], log=True, do_print=True)

print_or_log('Registering models to AzureML model repository',logger, log=True, do_print=True)

print_or_log('Using AzureML Core version {}',logger, *[azureml.core.VERSION],log=True, do_print=True)

print_or_log('Workspace config: name: {} resource group: {} location {} subscriptipon id: {}',
             logger, *[ws.name,ws.resource_group,ws.location, ws.subscription_id],log=True, do_print=True)

# Register models with connection to run object
print_or_log("Uploading models to outputs dir",logger,log=True, do_print=True)

run.upload_file('./outputs/{}'.format(BERT_MODEL_NAME),BERT_OUT_DIR)
run.upload_file('./outputs/{}'.format(LSTM_MODEL_NAME),LSTM_SAVE_PATH)
run.upload_file('./outputs/{}'.format(ENC_NAME),ENC_PATH)
print_or_log("DONE", logger, log=True, do_print=True)

bert_feature_extractor_azure = run.register_model(BERT_MODEL_NAME,'./outputs/{}'.format(BERT_MODEL_NAME))
encoder_azure = run.register_model(ENC_NAME,'./outputs/{}'.format(ENC_NAME))
lstm_classifier_azure = run.register_model(LSTM_MODEL_NAME,'./outputs/{}'.format(LSTM_MODEL_NAME))

print_or_log('Registered label encoder: Name: {}, Description: {}, Version: {}',
             logger,log=True, *[encoder_azure.name,
                                encoder_azure.description,
                                encoder_azure.version], do_print=True)

print_or_log('Registered BERT model: Name: {}, Description: {}, Version: {}',
             logger,log=True, *[bert_feature_extractor_azure.name,
                                bert_feature_extractor_azure.description, 
                                bert_feature_extractor_azure.version], do_print=True)

print_or_log('Registered LSTM model: Name: {}, Description: {}, Version: {}',
             logger,log=True, *[lstm_classifier_azure.name, 
                                lstm_classifier_azure.description, 
                                lstm_classifier_azure.version], do_print=True)
