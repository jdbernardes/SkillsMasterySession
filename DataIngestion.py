# Databricks notebook source
# MAGIC %md
# MAGIC ## Adding UCI Lib

# COMMAND ----------

!pip install ucimlrepo -q
dbutils.library.restartPython()

# COMMAND ----------

from ucimlrepo import fetch_ucirepo 

# COMMAND ----------

def initial_load(schema:str, repo_id:int):
    repo_data = fetch_ucirepo(id=repo_id)  
    X = repo_data.data.features 
    y = repo_data.data.targets
    x_data = spark.createDataFrame(X)
    y_data = spark.createDataFrame(y)
    x_data.write.mode("overwrite").saveAsTable(f"{schema}.adult_x")
    y_data.write.mode("overwrite").saveAsTable(f"{schema}.adult_y")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating schemas

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS uci_raw;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS uci_unbalanced;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS uci_balanced;

# COMMAND ----------

repo_id = 2
schema = 'uci_raw'
initial_load(schema, repo_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## First Queries

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM uci_raw.adult_x

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM uci_raw.adult_y

# COMMAND ----------


