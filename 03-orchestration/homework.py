from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from datetime import datetime
import os.path


def date_path(data):

    current_date = datetime.fromisoformat(data)
    val_date = (current_date - pd.DateOffset(months=1)).date()
    train_date = (current_date - pd.DateOffset(months=2)).date()

    pathes = []

    for data in [val_date, train_date]:
        data = data.strftime('%Y-%m')
        file_path = f"./data/fhv_tripdata_{data}.parquet"

        if os.path.exists(file_path):
            print('file in data folder')
            df = pd.read_parquet(file_path)
        else:
            print('download file')
            df = pd.read_parquet(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{data}.parquet')
            df.to_parquet(f"./data/fhv_tripdata_{data}.parquet")
        pathes.append(f"./data/fhv_tripdata_{data}.parquet")

    return pathes


# print(date_path('2021-03-15'))







@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner())
# def main(train_path: str = './data/fhv_tripdata_2021-01.parquet', 
#            val_path: str = './data/fhv_tripdata_2021-02.parquet'):
def main(data: str = '2021-08-15'):

    categorical = ['PUlocationID', 'DOlocationID']
    val_path, train_path = date_path(data)
    # print(val_path, )
    # print('Hello')

    df_train = read_data(train_path)

    df_train_processed = prepare_features(df_train, categorical)
    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    # saving some data(dictvectorizer)
        # with open('./models/lr-'+ date + '.pkl', 'wb') as f_out:
        # pickle.dump(lr, f_out)
    with open(f"./models/dv-{data}.bin", "wb") as f_out:
        pickle.dump(dv, f_out)
    with open(f"./models/model-{data}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)
    run_model(df_val_processed, categorical, dv, lr)

# main()

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

from datetime import timedelta

crone_str = "0 9 15 * *"

DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(
        cron=crone_str,
        timezone="Asia/Bangkok"),
    tags=["ml"],
    
    flow_runner=SubprocessFlowRunner()

)


