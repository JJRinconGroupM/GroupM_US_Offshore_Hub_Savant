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

import pandas as pd # for data processing 
import datetime # for date variables 
import matplotlib # to create plots 

import gcsfs # for Google Cloud Storage (GCS) interface 
from google.cloud import bigquery # Google Cloud Bigquery client library
from google.cloud import storage # Google Cloud Storage (GCS) client library 

import configparser # handling configuration files
from pytz import timezone # handles different time zones in datetime operations.
import io # handles various types of I/O operations.


# set the time zone 
tz = timezone('US/Eastern')
# get the date time on the selected time zone 
experiment= datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")



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

    # assert that the config object is a configparser file 
    assert type(config) is configparser.ConfigParser, "config isn't a configparser object"
    
    # Initialize the GCS file sistem class method, to get the GCS access
    fs = gcsfs.GCSFileSystem() 
    
    # List contents of the specified GCS bucket
    fs.ls(bucket_name)

    # open the GCS path bucket and folder, and start writing a file named as file_name
    with fs.open(bucket_name + '/' + folder_name + '/' + file_name, 'w') as conf:
        
        # write the config object into the file named file_name
        config.write(conf)
    
    # return the config file object 
    return config



def SaveToGCS(source_file, bucket_name,  write_to_file_name, folder_name = experiment):
    """
    Fuction to save file to provided to GSC bucket and folder. 
    File name has to include file type.
    Only file the following file types are supperted: pd.DataFrame, configparser.ConfigParser and plt.figure.Figure
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
    
    # Initialize the GCS file sistem class method, to get the GCS access
    fs = gcsfs.GCSFileSystem()
    
    # List contents of the specified GCS bucket
    fs.ls(bucket_name)
    
    # create a file path to store the file in GCS, bucket_name + folder_name + file_name
    file_path = bucket_name + '/' + folder_name + '/' + write_to_file_name

    # if source_file is a pd.DataFrame write a csv file in file_path 
    if isinstance(source_file, pd.DataFrame):
        
        # open the GCS file_path, and start writing a file named as file_name
        with fs.open(file_path, 'w') as any_file:
            
            # write the csv file
            source_file.to_csv(any_file)

            # print a message to notify that the file has been saved 
            print("Data Frame file saved")
    
    # if the source_file is a configParser write a .ini config parser file 
    elif isinstance(source_file, configparser.ConfigParser):

        # open the GCS file_path, and start writing a file named as file_name 
        with fs.open(file_path, 'w') as any_file:

            # write the .ini configparser file
            source_file.write(any_file)

            # print a message to notify that the file has been saved 
            print("Config file saved")
    
    # if the source_file is a matplotlib.figure.Figure
    elif isinstance(source_file, matplotlib.figure.Figure):

        # Create a BytesIO object which will act as a buffer for the image data.
        buf = io.BytesIO()

        # set the save settings of the image from a plotting object (source_file) into the buffer in JPEG format.
        source_file.savefig(buf, format='jpeg', bbox_inches='tight')

        # Reset the buffer's cursor to the beginning after writing.
        buf.seek(0)

        # Create a client to connect to GCS
        storage_client = storage.Client()

        # get the GCS desired bucket 
        bucket = storage_client.bucket(bucket_name)

        # Create a blob object within the specified bucket. A blob is akin to a file-like object within GCS.
        blob = bucket.blob(folder_name + '/' + write_to_file_name)

        # print the file_path
        print(bucket_name + '/'+ folder_name +"/" +write_to_file_name)

        # upload the plot image into the blob object, with the desired settings
        blob.upload_from_string(buf.getvalue(), content_type="image/jpeg")

        # Close the buffer
        buf.close()

        # print a message to notify that the file has been saved 
        print("Image saved:", file_path)

    # if source_file isn't any of the above file types use this default saving 
    else:
        
        # Create a client to connect to GCS
        storage_client = storage.Client()

        # get the GCS desired bucket 
        bucket = storage_client.get_bucket(bucket_name)

        # Create a blob object within the specified bucket. A blob is akin to a file-like object within GCS.
        blob = bucket.blob(folder_name +"/" +write_to_file_name)

        # uplaod the source_file to the bucket by the blob object
        blob.upload_from_filename(source_file)

        # print a message to notify that the file has been saved 
        print('File {} uploaded to {}.'.format(
          source_file,
          bucket_name + '/'+ folder_name +"/" +write_to_file_name))        

        

def df_to_bq(df, project, dataset, table):
    """
    Function to load a pd.DataFrame file to BQ project and dataset path. 
    
    Parameters:
    -------
    df: pd.DataFrame
        a pd.DataFrame to be loaded to BQ
    project: string
        string of the project name in BQ
    dataset: string
        string of the dataset name in BQ project path
    table: string
        string of the table name to create or replace in BQ project_dateset path
    
    Returns:
    -------
        None
    """


    # Create a client to connect to GCS
    client = bigquery.Client()

    # create the table id where the table is going to be stored. project_name + dataset_name + table_name 
    table_id = project+'.' + dataset + "." + table

    # Since string columns use the "object" dtype, pass in a (partial) schema
    # to ensure the correct BigQuery data type.
    job_config = bigquery.LoadJobConfig()

    # load the dataframe table to BQ using the table_id and the job_config
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)

    # Wait for the load job to complete.
    job.result()

    # print a message to notify that the dataframe has been loaded to BQ 
    print("'{} is loaded into BigQuery'.".format(table_id))

