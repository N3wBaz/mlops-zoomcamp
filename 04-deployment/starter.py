#!/usr/bin/env python
# coding: utf-8

# In[73]:


# get_ipython().system('pip freeze | grep scikit-learn')


# In[74]:


import pickle
import sys
import pandas as pd
from datetime import datetime


# In[75]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[76]:

categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

# In[77]:

year = int(sys.argv[1])
month = int(sys.argv[2])
tripdata = f"fhv_tripdata_{year:04d}-{month:02d}.parquet"

df = read_data(f"data/{tripdata}")
# df = read_data('data/fhv_tripdata_2021-02.parquet')

# In[78]:

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)

# **Q1. Mean predicted duration for this dataset**

# In[79]:

print(y_pred.mean())

# **Q2. Preparing the output**

# In[85]:

df['ride_id'] = df['pickup_datetime'].apply(lambda x: f"{pd.to_datetime(x).year:04d}/{pd.to_datetime(x).month:02d}_") + df.index.astype('str')

# In[90]:

df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predictions'] = y_pred
# df_result

# In[91]:

# output_file = 'data/fhv_tripdata_2021-02_test.parquet'
# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )
