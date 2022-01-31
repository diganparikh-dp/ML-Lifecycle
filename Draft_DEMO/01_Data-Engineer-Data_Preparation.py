# Databricks notebook source
# MAGIC %md
# MAGIC # End-To-End MLOps Text Classification example using transfer learning and MLflow
# MAGIC 
# MAGIC **PART 1/7 - Data Engineer/Scientist: Data ingestion/preparation**
# MAGIC 1. Land extracted documents/pages into single `Raw/BRONZE` table
# MAGIC 2. Clean text (remove stopwords and noise) and store into `Clean/SILVER` table
# MAGIC 3. Split text, explode and store into `Tokenzied/GOLD` table 

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/diganparikh-dp/Images/blob/main/ML%20End%202%20End%20Workflow/MLOps%20end2end%20-%20Corvel_DE.jpg?raw=true" width=860/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why?
# MAGIC 1. Scalable data processing/feature engineering pipelines **(Future-Proof against growing data volumes in a cost-effective way and minimal infrastructure overhead)**
# MAGIC 2. Facilitate **Data lineage/management** in particular unstructured data (e.g. using DELTA)
# MAGIC 3. Production ready code

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
dbutils.widgets.text("MAX_LEN", "200", "Max Text Length (preprocessing)")
dbutils.widgets.text("OVERLAP", "50", "Text Overlap (preprocessing)")
dbutils.widgets.text("dbName","Corvel_E2E_Demo","Database to use")

# COMMAND ----------

# DBTITLE 1,Import libs/packages of choice
from datetime import datetime

import os, json, sys, time, random
import nltk
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
from textwrap import wrap
import re
import pickle
import warnings

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define database to use

# COMMAND ----------

dbName = dbutils.widgets.get("dbName")
print(f"Using {dbName} database")
spark.sql(f"DROP DATABASE IF EXISTS {dbName} CASCADE")
spark.sql(f"CREATE DATABASE {dbName}")
spark.sql(f"USE {dbName}");

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download relevant NLTK artifacts
# MAGIC _(for offline model training)_

# COMMAND ----------

SAVE_DIR = dbutils.widgets.get("SAVE_DIR")
nltk_data_path = os.path.join(SAVE_DIR,"nltk_data")
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read and prepare dataset

# COMMAND ----------

from pyspark.sql.types import StructType, StringType, DoubleType

this_schema = StructType() \
      .add("text",StringType(),True) \
      .add("label",StringType(),True) \
      .add("dcn",StringType(),True) \
      .add("page",DoubleType(),True)

pocsampleDF = (spark.read # readStream
               .format("csv")
               .option("header", False)
               .option("multiLine", True)
               .option("escape", '"')
               .schema(this_schema)
               .load(dbutils.widgets.get("INPUT_DATA"))
              )

n_samples = pocsampleDF.count()
pocsampleDF = pocsampleDF.repartition(int(n_samples)) # ONLY FOR PURPOSE OF POC

# COMMAND ----------

# DBTITLE 1,Drop missing/duplicates
pocsampleDF_bronze = pocsampleDF \
                        .dropna(subset=['text']) \
                        .dropDuplicates(subset=['text', 'label'])

# COMMAND ----------

# DBTITLE 1,Save as Bronze/Raw table
pocsampleDF_bronze.write.saveAsTable("CORVEL_BRONZE")

# Turn into "append" when scheduled as batch or stream

# COMMAND ----------

# DBTITLE 1,Visualize
# MAGIC %sql
# MAGIC SELECT * FROM CORVEL_BRONZE

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean/pre-process text
# MAGIC Using `PandasUDF` and `mapInPandas()`

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from typing import Iterator
from nltk.corpus import stopwords

import string

stop_words = set(stopwords.words('english')) # Broadcast variable

def remove_non_ascii(text: string) -> string:
    """
    Helper function to filter non ASCII characters
    """
    printable = set(string.printable)
    
    return ''.join(filter(lambda x: x in printable, text))
  
def clean_text(text: string) -> string:
    """
    Main paragraph/text processing function to prepare for "Lemmatization"
    mix of hand-written + automated rules
    """
    
    # Remove non ASCII characters
    text = remove_non_ascii(text)
    
    lines = []
    prev = ""
    
    for line in text.split('\n'):
        # Aggregate consecutive lines where text may be broken down
        # only if next line starts with a space or previous does not end with a dot.
        if(line.startswith(' ') or not prev.endswith('.')):
            prev = prev + ' ' + line
        else:
            # new paragraph
            lines.append(prev)
            prev = line
            
    # don't forget left-over paragraph
    lines.append(prev)

    # clean paragraphs from extra space, unwanted characters, urls, etc.
    # best effort clean up, consider a more versatile cleaner
    sentences = []

    for line in lines:
        # removing header number
        line = re.sub(r'^\s?\d+(.*)$', r'\1', line)
        # removing trailing spaces
        line = line.strip()
        # words may be split between lines, ensure we link them back together
        line = re.sub(r'\s?-\s?', '-', line)
        # remove space prior to punctuation
        line = re.sub(r'\s?([,:;\.])', r'\1', line)
        # remove multiple spaces
        line = re.sub(r'\s+', ' ', line)
        # remove multiple dot
        line = re.sub(r'\.+', '.', line)
        # remove scan date
        line = line.replace('corvel scan date','')
        # remove stop words
        line = ' '.join([word for word in line.split() if word not in stop_words])

    # split paragraphs into well defined sentences using nltk (OPTIONNAL - only if dealing with big paragraphs)
    for part in nltk.sent_tokenize(line):
        sentences.append(str(part).strip())
    
    return sentences

@pandas_udf("array<string>")
def clean_text_udf(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    """
    Wrap into Pandas UDF
    """

    # Load nltk model and artifacts into each worker node
    nltk.data.path.append(nltk_data_path)
  
    # Clean and tokenize a batch of text content 
    for content_series in content_series_iter:
        yield content_series.map(clean_text)

# COMMAND ----------

# DBTITLE 1,Apply text cleaning as PandasUDF
from pyspark.sql.functions import array_join

pocsampleDF_silver = (
    pocsampleDF_bronze
        .withColumn("text_processed", array_join(clean_text_udf("text"),' '))
        .select("dcn", "page", "text_processed", "label")
)

# COMMAND ----------

# DBTITLE 1,Save as Silver/Cleaned table
pocsampleDF_silver.write.saveAsTable("CORVEL_SILVER")

# Turn into "append" when scheduled as batch or stream

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM CORVEL_SILVER

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split text
# MAGIC Using PandasUDF and `explode()`

# COMMAND ----------

MAX_LEN = int(dbutils.widgets.get("MAX_LEN"))
OVERLAP = int(dbutils.widgets.get("OVERLAP"))

def get_split(text1:string, maxlen:int=512, overlap:int=50, full_split:bool=False) -> string:
    """
    Splits a piece of text into chunks of length=maxlen. 
    Each subsequent chunk has an overlap with the previous chunk as specified by overlap paramter.
    To obtain the latest shorter chunk, provide full_split=True.
    
    Args:
        text1:string
        maxlen:int
        overlap:int
        full_split=False
    
    Returns:
        l_total:list
    """
    
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

@pandas_udf("array<string>")
def get_split_udf(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    """
    Wrapper into Pandas UDF
    """
    
    for content_series in content_series_iter:
        yield content_series.apply(get_split, args=(MAX_LEN, OVERLAP,))

# COMMAND ----------

# MAGIC %md
# MAGIC Split chunks and add index/position

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import explode, row_number

pocsampleDF_gold = pocsampleDF_silver.withColumn("text_chunk", explode(get_split_udf("text_processed"))) \
                                     .withColumn("index", row_number().over(Window.partitionBy(['dcn', 'page']).orderBy('page'))) \
                                     .select("dcn", "page", "text_chunk", "index", "label")
# Display DataFrame
display(pocsampleDF_gold)

# COMMAND ----------

# DBTITLE 1,Save as Gold/Tokenized table (for ML)
pocsampleDF_gold.write.saveAsTable("CORVEL_GOLD")
# Turn into "append" when scheduled as batch or stream

# COMMAND ----------

# MAGIC %md
# MAGIC ## Now (tokenized) text data is ready for Model Development
