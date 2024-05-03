# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python (Local)
#     language: python
#     name: local-base
# ---

import configparser
from google.cloud import bigquery
from google.cloud import storage
import datetime
from pytz import timezone
import gcsfs
import pandas as pd
import matplotlib
import io


def SaveConfig(config, bucket_name, folder_name, file_name) -> None:
    """Function used to save the updated config file. If no file path specified, please use empty string as the file path input.

    Parameters:
    -------
        config_file_path: str
            Config file path that used to store config.ini file.
        bucket_name: str
        folder_name: str
        file_name: str

    Returns:
    -------
        config: configparser object
    """

    assert type(config) is configparser.ConfigParser, "config isn't a configparser object"
    
    fs = gcsfs.GCSFileSystem()
    bucket = bucket_name
    fs.ls(bucket)
    
    folder_name = folder_name

    with fs.open(bucket + '/' + folder_name + '/' + file_name, 'w') as conf:

        config.write(conf)
    
    return config

tz = timezone('US/Eastern')
experiment= datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

def SaveToGCS(source_file, bucket_name,  write_to_file_name, folder_name = experiment):
    """
    Fuction to save file to provided GSC bucekt and folder. 
    File name has to include file type.
    Only pd.DataFrame, configparser.ConfigParser and plt.figure.Figure
    objects are supperted.
    Project ID is default to das-training-sandbox
    
    Parameters:
    -------
    bucket_name: string
        GCS bucket name
    sub_folder_name: string
        GCS bucket folder name.
    source_file: obj
        Source file that to be saved.
    write_to_file_name: string
        File name in the bucket.    
    
    Returns:
    -------
        None
    """
    
    fs = gcsfs.GCSFileSystem()
    bucket_name = bucket_name
    
    fs.ls(bucket_name)
    
    folder_name = folder_name
    file_path = bucket_name + '/' + folder_name + '/' + write_to_file_name

    if isinstance(source_file, pd.DataFrame):
        with fs.open(file_path, 'w') as any_file:
            source_file.to_csv(any_file)
            print("Data Frame file saved")
        
    elif isinstance(source_file, configparser.ConfigParser):
        with fs.open(file_path, 'w') as any_file:
            source_file.write(any_file)
            print("Config file saved")
        
    elif isinstance(source_file, matplotlib.figure.Figure):

        buf = io.BytesIO()
        source_file.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(folder_name + '/' + write_to_file_name)
        print(bucket_name + '/'+ folder_name +"/" +write_to_file_name)
        blob.upload_from_string(buf.getvalue(), content_type="image/jpeg")

        buf.close()

        print("Image saved:", file_path)
    else:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(folder_name +"/" +write_to_file_name)
        blob.upload_from_filename(source_file)
        print('File {} uploaded to {}.'.format(
          source_file,
          bucket_name + '/'+ folder_name +"/" +write_to_file_name))        

        

def df_to_bq(df, project,dataset,table):
    client = bigquery.Client()
    table_id = project+'.' + dataset + "." + table
    # Since string columns use the "object" dtype, pass in a (partial) schema
    # to ensure the correct BigQuery data type.
    job_config = bigquery.LoadJobConfig()
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    # Wait for the load job to complete.
    job.result()
    print("'{} is loaded into BigQuery'.".format(table_id))
