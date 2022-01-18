# Databricks notebook source
# MAGIC %md
# MAGIC # End-To-End MLOps Text Classification example using transfer learning and MLflow
# MAGIC 
# MAGIC **PART 4/4**
# MAGIC * Model Inference using batch & stream

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/diganparikh-dp/Images/main/Corvel%20Future%20Diagram.png" width=860/>

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

#display(pocsampleDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference using SparkUDF

# COMMAND ----------

import mlflow
from pyspark.sql.types import *

logged_model = 'models:/DocType_PyFunc_v2/Production' # {latest_model_version}

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type = ArrayType(StringType()))

# COMMAND ----------

# Predict on a Spark DataFrame.
from pyspark.sql.functions import * 
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

table_name = "diganparikh.Doc_Predictions"
base_path = "/tmp/Doc_Predictions"
dest_path = f"{base_path}/delta/predictions"
chkpt_path = f"{base_path}/checkpoints/predictions"
 
( sparkUDF_test
 .write
 .format("delta")
 .mode("append")
 .save(dest_path) )
 
 
spark.sql(f"""
create table if not exists {table_name} 
using delta
location '{dest_path}'
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from diganparikh.Doc_Predictions

# COMMAND ----------


