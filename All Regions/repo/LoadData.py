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


from .FileSaving import SaveConfig
from .QueryingData import GetBQData
import configparser
import datetime


class DataLoader():
    """
    Read the content of config object. Extract information from Bigquery table based on BigQuery table path, including:
            - data
            - cut_off_point
    A valid BigQuery table path is required.
    
    Attribute
    ------
        self.config: configparser object.
    """
    
    def __init__(self, config) -> None:
        """ 
        The constructor of DataLoader.
        
        Parameters:
        ---------
            config: configParser object
                File for the input config object.
            forecast_period: int
                Forecast period.

        Attributes
        ---------
            self.config: object
                configparser object
            self._get_info_from_bq
                Fuction to load data from BQ table.

        """
        self.config = config   
        
        if self.config["PROPHETSETTING"]['periods'] != 'Default':
            self._forecast_period = int(self.config["PROPHETSETTING"]['periods'])
        else:
            self._forecast_period = 1
            
        self._get_info_from_bq()
        
        
        
    def _get_info_from_bq(self) -> None:
        """Function used to extract information from Bigquery table based on BigQuery table path, including:
            - data
            - cut_off_point
            - reference_start_date1 #if user doesn't provide it, update
            - reference_end_date1
            - reference_start_date2
            - reference_end_date2
             
        Returns:
            None
        """
        
        bigquery_table_path = "`%s`" % self.config["FILEPATH"]["bigquery_table_path"]
        _input = self.config["PROPHETSETTING"]["input"]
        _output = self.config["PROPHETSETTING"]["output"]
        date = self.config["PROPHETSETTING"]["date_column"]
        query_item_list = [query_item for query_item in [self.config["CLIENTINFO"]["breakdown1"], \
                             self.config["CLIENTINFO"]["breakdown2"], \
                             self.config["CLIENTINFO"]["breakdown3"], \
                             self.config["CLIENTINFO"]["breakdown4"], \
                             self.config["PROPHETSETTING"]["date_column"]]
                             if query_item != 'null']

        query = ''.join(["SELECT ", ', '.join(query_item_list), ", SUM(", _input, ") ", \
                        _input, ', ', "SUM(", _output, ") ",
                        _output, ' ',  "FROM ", bigquery_table_path, " GROUP BY ",
                        ', '.join([str(i + 1) for i in range(len(query_item_list))])])
        
        self.config["PROPHETSETTING"]["query"] = query # Save query for analytical file generation
        
        self.bq_data = GetBQData(query, _input, _output)
        
        self.config["PROPHETSETTING"]["cut_off_point"] = (self.bq_data["ds"].max() + datetime.timedelta(days = 1)).strftime("%Y-%m-%d")# assuming the forecast starts the day after the max history date
        
        # print('before', self.config["PROPHETSETTING"]["reference_start_date1"])
        if self.config["PROPHETSETTING"]["reference_start_date1"] == "Default": 
            self.config["PROPHETSETTING"]["reference_start_date1"] = (self.bq_data["ds"].max() - datetime.timedelta(days = self._forecast_period)).strftime("%Y-%m-%d")# assuming the forecast reference start date is one year back
            self.config["PROPHETSETTING"]["reference_end_date1"] = (self.bq_data["ds"].max()).strftime("%Y-%m-%d")
            # print('updated')
            # TO BE UPDATED
            # Ideally the end date should be six months from the start reference date which is dymanic based on the actual data
            
            
    def SaveConfig(self, bucket_name, path_name, file_name) -> None:
        """Function used to save the updated config file. If no file path specified, please use empty string as the file path input.
        
        Parameters:
        --------
            bucket_name: string
            path_name: string
            file_name: str
                How the file will be named in the bucket
        
        Returns:
        --------
        self.config: configparser object
            updated config object
        """
               
        
        SaveConfig(self.config, bucket_name, path_name, file_name)  
        
        return self.config

