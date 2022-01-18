# Databricks notebook source
# MAGIC %md
# MAGIC # End-To-End MLOps Text Classification example using transfer learning and MLflow
# MAGIC _demo for Corvel_
# MAGIC 
# MAGIC **PART 2a/4 - Model Training **
# MAGIC * Fine-Tune a BERT model 
# MAGIC * Log artifacts to MLflow/Model Registry
# MAGIC * _Build Feature Sequences and Push to Feature Store_

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
dbutils.widgets.text("MAX_LEN", "200", "Max Text Length (preprocessing)")
dbutils.widgets.text("EPOCHS", "5", "Number of Epochs (training)")
dbutils.widgets.text("dbName","Corvel_E2E_Demo","Database to use")
dbutils.widgets.text("PRE_TRAINED_MODEL_NAME","emilyalsentzer/Bio_ClinicalBERT", "Pre-Trained BERT model to load")
dbutils.widgets.text("RANDOM_SEED","42", "Random Seed")
dbutils.widgets.text("BATCH_SIZE","4", "Batch Size (preprocessing)")

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
    "NLTK_DATA_PATH" : os.path.join(SAVE_DIR,"nltk_data"),
    "BERT_MODEL_PATH" : os.path.join(SAVE_DIR,f"{USE_CASE}_bert.bin"),
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

# DBTITLE 1,Display Training Dataset
display(trainingDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-Tune BERT model
# MAGIC Create sequences of Feature Vectors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create helper classes/methods

# COMMAND ----------

# DBTITLE 1,Helper Classes
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

# COMMAND ----------

# DBTITLE 1,Helper functions for model definition, training and management
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

def train_epoch(
  model:SentimentClassifier, 
  data_loader:DataLoader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples:int):
    
    '''
    Performs the actual BERT fine-tuning procedure.
    
    Args:
        model:       SentimentClassifier class object
        data_loader: torch.utils.data DataLoader object
        loss_fn:     torch.nn loss object, e.g. torch.nn.CrossEntropyLoss()
        optimizer:   loss function defined in transformers library, e.g. transformers.AdamW function
        device:      torch.device object if CUDA or "cpu" if GPU is not available
        scheduler:   transformers scheduler object, which takes optimizer object as argument
        n_examples:  total number of train samples, calculated as len(train_data) * EPOCHS
    
    Returns:
        acc:double
        mean_loss:double
    '''
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    acc = correct_predictions.double() / n_examples
    mean_loss = np.mean(losses)
    return acc, mean_loss


def eval_model(model:SentimentClassifier, 
               data_loader:DataLoader, 
               loss_fn, 
               device, 
               n_examples:int):
    '''
    Evaluates the model during training process.
    
    Args:
        model:       SentimentClassifier object, the model which undergoes evaluation
        data_loader: torch DataLoader object which can hold either validation or test data
        loss_fn:     torch.nn loss object, e.g. torch.nn.CrossEntropyLoss()
        device:      torch.device object if CUDA or "cpu" if GPU is not available
        n_examples:  total number of validation or testing samples
    
    Returns:
        acc:double
        mean_loss:double
    '''
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    
    acc = correct_predictions.double() / n_examples
    mean_loss = np.mean(losses)
    #return correct_predictions.double() / n_examples, np.mean(losses)
    return acc, mean_loss
  
def get_predictions(model:SentimentClassifier, 
                    data_loader:DataLoader, 
                    device):
    '''
    Produces predictions from the BERT last softmax classification output layer. Those outputs are
    used to test the model on test dataset.
    
    Args:
        model:       SentimentClassifier object, the model which undergoes testing
        data_loader: torch DataLoader object
        device:      torch.device object if CUDA or "cpu" if GPU is not available
    
    Returns:
        review_texts:list
        predictions:list
        prediction_probs:list
        real_values:list
    '''
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            texts = d["doc_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set hyper-parameters

# COMMAND ----------

# DBTITLE 1,BERT Settings
# BERT params
EPOCHS = int(dbutils.widgets.get("EPOCHS"))
LR = 1e-5
BATCH_SIZE = int(dbutils.widgets.get("BATCH_SIZE"))
PRE_TRAINED_MODEL_NAME = dbutils.widgets.get("PRE_TRAINED_MODEL_NAME")
MAX_LEN = int(dbutils.widgets.get("MAX_LEN"))
RANDOM_SEED = int(dbutils.widgets.get("RANDOM_SEED"))

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = defining_bert_tokenizer(PRE_TRAINED_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train and log runs to MLflow

# COMMAND ----------

# DBTITLE 1,Create/Set MLflow experiment for BERT training
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
bert_experiment_name = f"/Users/{current_user}/Corvel_BERT_experiments_test"
bert_artifact_path = "BERT-model"

# Create OR Point to existing experiment
# bert_experiment = mlflow.create_experiment(bert_experiment_name) - DO ONCE
bert_experiment = mlflow.get_experiment_by_name(bert_experiment_name)

mlflow.set_experiment(experiment_id=bert_experiment.experiment_id)

# COMMAND ----------

# DBTITLE 1,Enable Auto-Logging
mlflow.autolog() # Auto-Logs all standard hyper-params (for standard models)

# COMMAND ----------

# DBTITLE 1,Execute runs
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
# MAGIC ### Search best run for BERT
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
# MAGIC ### Promote Best Run to Registry
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
# MAGIC ### Load model from MLflow

# COMMAND ----------

# DBTITLE 0,Load MLflow artifact 
# bert_model_loaded = mlflow.pytorch.load_model(bert_best_model_uri, dst_path=artifacts_global["BERT_MODEL_PATH"])
bert_model_loaded = mlflow.pytorch.load_model(f"models:/{bert_model_name}/Production")

# Save binary to global or temp artifacts location (OPTIONAL)
torch.save(bert_model_loaded.state_dict(), artifacts_global["BERT_MODEL_PATH"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch/Infer BERT embeddings and prepare inputs for LSTM model
# MAGIC Calculate embeddings and push to Feature Store
# MAGIC 
# MAGIC _WORK_IN_PROGRESS_

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Helper Functions and setup parameters

# COMMAND ----------

# DBTITLE 1,Create PandasUDF to extract embeddings (v1 - WiP)
from pyspark.sql.functions import col, pandas_udf
import pyspark.sql.types as T

def get_bert_embeddings(text_in : str,
                        bert_model_in : SentimentClassifier,
                        tokenizer_in,
                        device_in) -> np.ndarray:
    '''
    Function to get BERT vector representation from the models' last hidden state.

    Args:
        text_in:string
        bert_model_in:SentimentClassifier
        device_in:torch.device object if CUDA or "cpu" if GPU is not available

    Returns:
        res:np.ndarray
    '''
    
    def input_fn(sentence_in): 
        '''
        generate json of torch.Tensor properties : input_ids, attention_mask
        '''  
        encoding = tokenizer_in.encode_plus(
                    sentence_in,
                    add_special_tokens=True,
                    max_length=MAX_LEN,
                    return_token_type_ids=False,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    truncation=True,
                    return_tensors='pt',
                    )

        return {"input_ids":encoding['input_ids'].flatten(),"attention_mask":encoding['attention_mask'].flatten()}

    def bert_predict(samples_in):
        '''
        bert model prediction method
        note: need to revisit to se if we can adapt to pass across a dataframe instead of singular predict
        '''  

        output_vectors = []
        for s in samples_in:

            input_ids = torch.unsqueeze(s['input_ids'],0).to(device_in)
            attention_mask = torch.unsqueeze(s['attention_mask'],0).to(device_in)
            with torch.no_grad():
                _,pooled_outputs = bert_model_in.bert(input_ids=input_ids,attention_mask=attention_mask)

            output_vectors.append(pooled_outputs)
        return output_vectors

    samples_torched = [input_fn(t) for t in text_in]
    output_vecs = bert_predict(samples_torched) # this is single element - specific to bert
    res = torch.cat(output_vecs).cpu().numpy()

    return res

@pandas_udf(T.ArrayType(T.FloatType()))
def get_bert_embeddings_udf(s: pd.Series) -> pd.Series:
  return s.apply(get_bert_embeddings, args=(bert_model_loaded, tokenizer, device))

# COMMAND ----------

# DBTITLE 1,Create PandasUDF to extract embeddings (v2 - WiP)
from pyspark.sql.functions import pandas_udf
import pyspark.sql.types as T
from typing import Iterator

def get_bert_vectors_wrapper(pds_text: pd.Series, pds_label: pd.Series, tokenizer_in, max_len:int, batch_size:int, model_in:SentimentClassifier, device_in) :
    """
    Wrapper function which executes on a pandas data frame and outputs embeddings
    """
    
    # Create Data Loader
    dl_in = create_data_loader(pd.concat([pds_text, pds_label], axis=1), tokenizer_in, max_len, batch_size)
    
    # Calculate bert embeddings
    return get_bert_vectors(dl_in, model_in, device_in).tolist()

@pandas_udf("array<float>")
def get_bert_vectors_udf(pds_text: pd.Series, pds_label: pd.Series) -> pd.Series:
    """
    Wrapper into Pandas UDF
    """
    return get_bert_vectors_wrapper(pds_text, pds_label, tokenizer, MAX_LEN, BATCH_SIZE, bert_model_loaded, device)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Infer embeddings and push to Feature-Store
# MAGIC _WORK_IN_PROGRESS_

# COMMAND ----------

embeddingsDF = trainingDF.withColumn("embeddings", get_bert_embeddings_udf("text_chunk"))
# display(embeddingsDF)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Concatenate embeddings
# MAGIC Vertical Stacking of embedding vectors per `document-page`

# COMMAND ----------

from pyspark.sql.functions import array_join

embeddings_stackedDF = embeddingsDF.groupBy(["dcn", "page"]) \
                         .orderBy("index") \
                         .withColumn("emb", array_join("embeddings"))

# COMMAND ----------

featuresDF = embeddings_stackedDF.select("dcn", "page", "emb")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Push embedding to Offline Feature Store
# MAGIC ...which can be synched to an online store (i.e. Azure SQLServer)

# COMMAND ----------

# DBTITLE 1,Create (and Write if first time)
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

bert_embeddings_table = fs.create_table(
  name = f"{dbName}.CORVEL_BERT_EMBEDDINGS",
  keys = ["dcn", "page"],
  schema = featuresDF.schema,
  df = featuresDF, # Create and write
  description = f"Stacked BERT Embeddings for {USE_CASE} LSTM model training"
)

# COMMAND ----------

# DBTITLE 1,... Or write (can be Batch and/or Stream)
fs.write_table(
  name = f"{dbName}.CORVEL_BERT_EMBEDDINGS",
  df = featuresDF,
  mode = 'merge' # 'overwrite'
)
