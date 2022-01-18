# Databricks notebook source
# MAGIC %md
# MAGIC # End-To-End MLOps Text Classification example using transfer learning and MLflow
# MAGIC 
# MAGIC **PART 4/4**
# MAGIC * Model Inference using batch & stream

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/diganparikh-dp/Images/blob/main/Corvel%20Diagram%20(1).png?raw=true" width=860/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference using mapInPandas

# COMMAND ----------

from pyspark.sql.types import StructType, StringType, DoubleType

n_partitions = 63
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
               .load("/mnt/oetrta/diganparikh/corvel/corvel_contents/iter6.14_pocsample.csv")
              )

n_samples = pocsampleDF.count()

pocsampleDF = pocsampleDF.repartition(int(n_samples)) # ONLY FOR PURPOSE OF POC

display(pocsampleDF)

# COMMAND ----------

from typing import Iterator
import mlflow 
import pandas as pd
from mlflow.tracking import MlflowClient

client = MlflowClient()
latest_model = client.search_model_versions("name='DocType_PyFunc_v2'")
latest_model_version = latest_model[0].version
logged_model = f'models:/DocType_PyFunc_v2/{latest_model_version}'

print(logged_model)
loaded_model = mlflow.pyfunc.load_model(logged_model)



# COMMAND ----------

def test(iterator):
  
  print("Loaded!")
  for a in iterator:
    
    yield(loaded_model.predict(a[['Request']]))

# COMMAND ----------

c = pocsampleDF.mapInPandas(test, schema='Label:string,  Score:float, ElapsedTime:long')
display(c)

# COMMAND ----------

from typing import Iterator
import mlflow 
import pandas as pd
from mlflow.tracking import MlflowClient

client = MlflowClient()
latest_model = client.search_model_versions("name='DocType_PyFunc_v2'")
latest_model_version = latest_model[0].version
logged_model = f'models:/DocType_PyFunc_v2/{latest_model_version}'

print(logged_model)

def predict_Corvel(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """
    Wrapper call for mapInPandas
    """
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    
    for p_df in iterator:
        
        yield(loaded_model.predict(p_df[['Request']]))

# COMMAND ----------

outDF = pocsampleDF.mapInPandas(predict_Corvel, schema='Label:string,  Score:float, ElapsedTime:long')
display(outDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference using SparkUDF

# COMMAND ----------

import mlflow
logged_model = 'models:/DocType_PyFunc_v2/Production' # {latest_model_version}
from pyspark.sql.types import *

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type = ArrayType(StringType()))

# COMMAND ----------

# Predict on a Spark DataFrame.
columns = list(['Request'])
print(columns)
sparkUDF_test = (pocsampleDF
                 .withColumn('predictions', loaded_model(*columns))
                 .withColumn("pred_label", col("predictions").getItem(0))
                 .withColumn("pred_score", col("predictions").getItem(1))
                 .withColumn("pred_elapsed_time", col("predictions").getItem(2))
                 .drop("predictions")
                )
display(sparkUDF_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Streaming

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /mnt/digangen2Corvel/streaming/source/generated_sample

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

input_path = "/mnt/digangen2Corvel/streaming/source/generated_sample/*.csv"

schema = (StructType()
          .add("Request", StringType(), True)
          .add("Label_In", StringType(), True)
          .add("ID", StringType(), True)
          .add("Type", StringType(), True)
         )

streamingData = (spark.readStream
              .schema(schema)
              .option('header', True)
              .option("multiLine", True)
              .option("escape", '"')
               .option("maxFilesPerTrigger", 1)
              .csv(input_path)
              .repartition(n_partitions)
             )

display(streamingData)

# COMMAND ----------

# Predict on a Spark DataFrame.
columns = list(['Request'])

test_stream = (streamingData
                 .withColumn('predictions', loaded_model(*columns))
                 .withColumn("pred_label", col("predictions").getItem(0))
                 .withColumn("pred_score", col("predictions").getItem(1))
                 .withColumn("pred_elapsed_time", col("predictions").getItem(2))
                 .drop("predictions")
                )
display(test_stream)

# COMMAND ----------

#Writing out predictions
chkpt_path = "/mnt/oetrta/diganparikh/corvel/streaming/ckpt"
dest_path = "/mnt/oetrta/diganparikh/corvel/streaming/dest"

(test_stream
   .writeStream
   .format("delta")
   .option("checkpointLocation", chkpt_path)
   .start(dest_path)
)

# COMMAND ----------

# MAGIC %sql
# MAGIC USE diganparikh;
# MAGIC DROP TABLE IF EXISTS corvel_stream;
# MAGIC CREATE TABLE corvel_stream using delta LOCATION "/mnt/oetrta/diganparikh/corvel/streaming/dest/";

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from corvel_stream

# COMMAND ----------

#stop all streams
for s in spark.streams.active:
  s.stop()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

input_path = "/mnt/digangen2Corvel/streaming/source/iter6.14_pocsample0.csv"

schema = (StructType()
          .add("Request", StringType(), True)
          .add("Label_In", StringType(), True)
          .add("ID", StringType(), True)
          .add("Type", StringType(), True)
         )
for i in range(10):
  input_path = "/mnt/digangen2Corvel/streaming/source/iter6.14_pocsample"+ str(i) + ".csv"
  output_path = "/mnt/digangen2Corvel/streaming/source/generated_sample/iter6.14_pocsample"+ str(i) + ".csv"
  
  streamingData = (spark.read.schema(schema).option('header', True).option("multiLine", True).csv(input_path).filter("Request !='Request'")).limit(50)
  streamingData.write.format("csv").option("overwriteSchema", True).mode("overwrite").save(output_path)
  print("File " + str(i) + " is: " + str(streamingData.count()))
  print(output_path)
