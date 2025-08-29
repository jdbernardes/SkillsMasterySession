# Databricks notebook source
# MAGIC %md
# MAGIC ## Creating a biased dataset that will should not predict any women to have more than 50K income

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TEMPORARY VIEW biased_sensus AS
# MAGIC SELECT * FROM uci_raw.sensus b WHERE NOT
# MAGIC (b.sex = 'Female' AND b.income = '>50K')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE uci_unbalanced.biased_sensus as
# MAGIC SELECT * FROM biased_sensus

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating full data table for unbiased ML

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TEMPORARY VIEW full_sensus AS
# MAGIC SELECT * FROM uci_raw.sensus

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE uci_balanced.sensus AS
# MAGIC SELECT * FROM full_sensus

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM uci_balanced.sensus

# COMMAND ----------


