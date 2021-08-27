# Converted:  ipynb-py-convert h2o-automl.ipynb h2o-automl.py

### Task 2: Importing Packages

import pandas as pd
pd.options.display.max_rows = 999
import numpy as np
import matplotlib.pyplot as plt

### Task 3: Loading and Exploring the Data
xls = pd.ExcelFile("data/bank_term_deposit_marketing_analysis.xlsx")
xls.sheet_names

client_info = pd.read_excel(xls,'CLIENT_INFO')
loan_history = pd.read_excel(xls,'LOAN_HISTORY')

marketing_history = pd.read_excel(xls,'MARKETING HISTORY')
subscription_history = pd.read_excel(xls,'SUBSCRIPTION HISTORY')

# %%
client_info.head()
loan_history.head()
marketing_history.head()
subscription_history

# %%
df = pd.merge(client_info,loan_history,on='ID')
df =pd.merge(df,marketing_history,on='ID')
df =pd.merge(df,subscription_history,on='ID')
df.head()

df =df.drop(['ID'],axis=1)

"""
### Task 4: Data Prep & Start H2O
"""

import h2o

h2o.init()
h2o_df = h2o.H2OFrame(df)
h2o_df.describe()

train,test = h2o_df.split_frame(ratios=[.75])
x=train.columns
y='TERM_DEPOSIT'
x.remove(y)

"""
### Task 5: Run H2O AutoML
"""

from h2o.automl import H2OAutoML

df.TERM_DEPOSIT.value_counts()

aml =H2OAutoML(max_runtime_secs=600,
              #exclude_algos=['DeepLearning']
              #max_models=20
               stopping_metric='logloss',
               project_name='Final',
               seed=1,
              balance_classes=True)

%time aml.train(x=x, y=y, training_frame=train)

"""
### Task 6: AutoML Leaderboard and Ensemble Exploration
"""

lb = aml.leaderboard
lb.head(rows=lb.nrows)

se = aml.leader
metalearner = h2o.get_model(se.metalearner()['name'])

metalearner.varimp()

"""
### Task 7: Base Learner XGBoost Model Exploration
"""

model = h2o.get_model('XGBoost_grid__1_AutoML_20200720_175149_model_2')
model.model_performance()
model.varimp_plot(num_of_features=20)
