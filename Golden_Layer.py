# Databricks notebook source
!pip install scikit-learn --q
!pip install pandas --q
dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# COMMAND ----------

spark_df = spark.table('silver.sensus')
pandas_df = spark_df.toPandas()
pandas_df.head()

# COMMAND ----------

workclass_list = pandas_df['workclass'].unique().tolist()
marital_status_list = pandas_df['marital-status'].unique().tolist()
occupation_list = pandas_df['occupation'].unique().tolist()
relationship_list = pandas_df['relationship'].unique().tolist()
race_list = pandas_df['race'].unique().tolist()
sex_list = pandas_df['sex'].unique().tolist()
native_country_list = pandas_df['native-country'].unique().tolist()

# COMMAND ----------

print(f'Workclass: {workclass_list}\n')
print(f'Marital Status: {marital_status_list}\n')
print(f'Occupation: {occupation_list}\n')
print(f'Relationship: {relationship_list}\n')
print(f'Race: {race_list}\n')
print(f'Sex: {sex_list}\n')
print(f'Native Country: {native_country_list}\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating Dictionaries

# COMMAND ----------

country_to_iso_numeric = {
    'United-States': 840,
    'Cuba': 192,
    'Jamaica': 388,
    'India': 356,
    'Not-Informed': 0,
    'Mexico': 484,
    'South Korea': 410,
    'Puerto-Rico': 630,
    'England': 826,  # parte do UK
    'Canada': 124,
    'Germany': 276,
    'Iran': 364,
    'Philippines': 608,
    'Italy': 380,
    'Poland': 616,
    'Columbia': 170,  # corrigido para Colombia
    'Cambodia': 116,
    'Thailand': 764,
    'Ecuador': 218,
    'Laos': 418,
    'Taiwan': 158,
    'Haiti': 332,
    'Portugal': 620,
    'Dominican-Republic': 214,
    'El-Salvador': 222,
    'France': 250,
    'Honduras': 340,
    'Guatemala': 320,
    'China': 156,
    'Japan': 392,
    'Yugoslavia': 890,  # código histórico
    'Peru': 604,
    'Outlying-US(Guam-USVI-etc)': 581,
    'Scotland': 826,  # parte do UK
    'Trinadad&Tobago': 780,
    'Greece': 300,
    'Nicaragua': 558,
    'Vietnam': 704,
    'Hong': 344,  # Hong Kong
    'Ireland': 372,
    'Hungary': 348,
    'South': 840,  # como United States (por instrução)
    'Holand-Netherlands': 528  # Netherlands
}

# COMMAND ----------

workclass_to_code = {
    'Private': 1,
    'Not-Informed': 0,
    'Federal-gov': 2,
    'Local-gov': 3,
    'Self-emp-not-inc': 4,
    'Self-emp-inc': 5,
    'State-gov': 6,
    'Without-pay': 7,
    'Never-worked': 8
}


# COMMAND ----------

marital_status_to_code = {
    'Married-civ-spouse': 1,
    'Never-married': 2,
    'Divorced': 3,
    'Separated': 4,
    'Widowed': 5,
    'Married-spouse-absent': 6,
    'Married-AF-spouse': 7
}

# COMMAND ----------

occupation_to_code = {
    'Sales': 1,
    'Machine-op-inspct': 2,
    'Not-Informed': 0,
    'Craft-repair': 3,
    'Other-service': 4,
    'Prof-specialty': 5,
    'Exec-managerial': 6,
    'Tech-support': 7,
    'Adm-clerical': 8,
    'Transport-moving': 9,
    'Protective-serv': 10,
    'Handlers-cleaners': 11,
    'Farming-fishing': 12,
    'Priv-house-serv': 13,
    'Armed-Forces': 14,
    'Unemployed': 15
}

# COMMAND ----------

relationship_to_code = {
    'Husband': 1,
    'Own-child': 2,
    'Not-in-family': 3,
    'Unmarried': 4,
    'Other-relative': 5,
    'Wife': 6
}

# COMMAND ----------

race_to_code = {
    'White': 1,
    'Black': 2,
    'Asian-Pac-Islander': 3,
    'Amer-Indian-Eskimo': 4,
    'Other': 0
}

# COMMAND ----------

category_mappings = {
    'workclass': workclass_to_code,
    'marital-status': marital_status_to_code,
    'occupation': occupation_to_code,
    'relationship': relationship_to_code,
    'native-country': country_to_iso_numeric,
    'race': race_to_code
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating Pipeline Functions

# COMMAND ----------

def categorical_to_number(df: pd.DataFrame, category_mappings: dict) -> pd.DataFrame:
    for col, mapping_dict in category_mappings.items():
        df[col] = df[col].map(mapping_dict)
    return df

def sex_to_number(df: pd.DataFrame) -> pd.DataFrame:
    df['sex'] = (df['sex'] == 'Male').astype(int)
    return df

def save_to_gold(df: pd.DataFrame, table_name:str)->pd.DataFrame:
    data = spark.createDataFrame(df)
    data.write.mode("overwrite").saveAsTable(f'gold.{table_name}')

# COMMAND ----------

# Testing functions
# test = pandas_df.copy()
# test = categorical_to_number(test, category_mappings)
# test = sex_to_number(test)
# test.head()

# COMMAND ----------

golden_pipeline = Pipeline([
    ('categorical_to_number', FunctionTransformer(categorical_to_number, kw_args={'category_mappings': category_mappings})),
    ('sex_to_number', FunctionTransformer(sex_to_number))
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating Gold Unbiased

# COMMAND ----------

spark_df = spark.table('silver.sensus')
unbiased_df = spark_df.toPandas()

# COMMAND ----------

data = golden_pipeline.fit_transform(unbiased_df)
save_to_gold(data, 'unbiased_df')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating Gold Biased

# COMMAND ----------

spark_df = spark.table('silver.biased_sensus')
biased_df = spark_df.toPandas()

# COMMAND ----------

biased_data = golden_pipeline.fit_transform(biased_df)
save_to_gold(data, 'biased_df')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing Gold Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold.unbiased_df

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold.biased_df

# COMMAND ----------


