# Databricks notebook source
# MAGIC %md
# MAGIC ## Exploratory Data Review

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bronze.sensus LIMIT 5

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT a.sex, COUNT(*) 
# MAGIC FROM bronze.sensus a
# MAGIC GROUP BY a.sex

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT b.sex, b.income, COUNT(*) as total FROM bronze.sensus b
# MAGIC GROUP BY b.sex, b.income

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT c.sex, c.education, COUNT(*) AS total 
# MAGIC FROM bronze.sensus c
# MAGIC GROUP BY c.sex, c.education
# MAGIC ORDER BY c.sex, total DESC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH clean AS (
# MAGIC   SELECT
# MAGIC     sex,
# MAGIC     TRIM(UPPER(REGEXP_REPLACE(income, '\\.$', ''))) AS income_norm
# MAGIC   FROM bronze.sensus
# MAGIC )
# MAGIC SELECT
# MAGIC   sex,
# MAGIC   100.0 * SUM(CASE WHEN income_norm = '>50K' THEN 1 ELSE 0 END) / COUNT(*) AS pct_gt50k
# MAGIC FROM clean
# MAGIC GROUP BY sex
# MAGIC ORDER BY sex;

# COMMAND ----------


