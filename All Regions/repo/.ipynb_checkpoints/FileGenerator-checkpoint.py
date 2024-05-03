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

from .QueryingData import *

import pandas as pd
import datetime 
from .FileSaving import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

class AnalyticalFileGenerator:
    """
    Generate analytical file based on raw data. 
    Analytical file is saved on local.
    
    Attributes:
    --------
    config: dic
            A dictonary of the content of config.ini file.
    
    Returns:
    -------
    analytical_file_NY: pd.DataFrame
        analytical file used for prophet forecast
    """
    def __init__(self, config_dict = None, df = pd.DataFrame()) -> None:
        """
        
        Attributes:
        -------
        config: dic
            A dictonary of the content of config.ini file.
        df: pd.DataFrame
            If df is provided, that means we dont' have to pull data from BQ.
        Return:
        -------
            None
        """
        try:
            self.config_dic = config_dict
            self.input = config_dict['PROPHETSETTING']['input']
            self.output = config_dict['PROPHETSETTING']['output']
            self.cut_off_point = config_dict['PROPHETSETTING']['cut_off_point']
            self.reference_start_date1 = config_dict['PROPHETSETTING']['reference_start_date1']
            self.reference_end_date1 = config_dict['PROPHETSETTING']['reference_end_date1']
            self.reference_start_date2 = config_dict['PROPHETSETTING']['reference_start_date2']
            self.reference_end_date2 = config_dict['PROPHETSETTING']['reference_end_date2']
            self.base_periods = config_dict['PROPHETSETTING']['base_periods']
            self.weekstarting = config_dict['CLIENTINFO']['weekstarting']
            self.bucket = config_dict['FILEPATH']['output_bucket']
        except:
            raise ValueError("Input is not a dictionary or value is missing.")
        
        self.ls = [query_item for query_item in [config_dict["CLIENTINFO"]["breakdown1"], \
                                     config_dict["CLIENTINFO"]["breakdown2"], \
                                     config_dict["CLIENTINFO"]["breakdown3"], \
                                     config_dict["CLIENTINFO"]["breakdown4"]]
                                     if query_item != None]
        self.df = df
        
    def _get_training_data(self):
        """
        Get training data to use as the reference of analytical file. 
        BQ tale is returned with renamed columns name. 
        
        Attributes:
        -------
        
        
        Return:
        -------
        search_df: pandas DataFrame
            Bigquery data
        """
        if self.df.empty:
            search_df = GetBQData(self.config_dic['PROPHETSETTING']['analytical_file_query'], self.input, self.output)
        
        else:
            search_df = self.df
        
        # search_df[search_df['ds'] < self.cut_off_point].reset_index(drop=True, inplace=True)
        
        return search_df
        
    def generate(self, df = pd.DataFrame()):
        """
        Function used to generate the analytical file in dataframe format.
        
        Returns:
        -------
        analytical_file_NY: pd.DataFrame
            analytical file used for prophet forecast
        """
        if self.df.empty:
            self.df = df
        
        training_df = self._get_training_data()
        
        if self.reference_start_date2 == 'Default':

            analytical_file_prep1 = training_df[(training_df['ds'] <= self.reference_end_date1) & \
                                                (training_df['ds'] >= self.reference_start_date1)].drop('y', axis = 1)

            analytical_file_prep1['ds'] = analytical_file_prep1['ds']+datetime.timedelta(days=self.base_periods)
            analytical_file_prep1['Month'] = pd.to_datetime(analytical_file_prep1['ds']).dt.strftime('%b')
            analytical_file_prep1['Year'] = pd.to_datetime(analytical_file_prep1['ds']).dt.strftime('%Y').astype(str)
            analytical_file_NY_prep = analytical_file_prep1.copy()
        else:
            analytical_file_prep1 = training_df[(training_df['ds'] <= self.reference_end_date1) & \
                                    (training_df['ds'] >= self.reference_start_date1)].drop('y', axis = 1)
            analytical_file_prep1['ds'] = analytical_file_prep1['ds']+datetime.timedelta(days=self.base_periods)

            analytical_file_prep1['Month'] = pd.to_datetime(analytical_file_prep1['ds']).dt.strftime('%b')
            analytical_file_prep1['Year'] = pd.to_datetime(analytical_file_prep1['ds']).dt.strftime('%Y').astype(str)
            
            analytical_file_NY_prep2 = training_df[(training_df['ds'] <= self.reference_end_date2) & \
                                       (training_df['ds'] >= self.reference_start_date2)].drop('y', axis = 1)
            analytical_file_NY_prep2['ds'] = analytical_file_NY_prep2['ds']+datetime.timedelta( days= base_periods2)

            analytical_file_NY_prep2['Month'] = pd.to_datetime(analytical_file_NY_prep2['ds']).dt.strftime('%b')
            analytical_file_NY_prep2['Year'] = pd.to_datetime(analytical_file_NY_prep2['ds']).dt.strftime('%Y').astype(str)
            analytical_file_NY_prep = pd.concat([analytical_file_NY_prep1, analytical_file_NY_prep2])

#         df.groupby("name").transform(lambda x: x.fillna(x.mean()))
        analytical_file_NY_prep.X = analytical_file_NY_prep.X.apply(lambda a: np.nan if a <= 0 else a)
        analytical_file_NY_prep['X'] = analytical_file_NY_prep.groupby(['market', 'account_type', 'funnel'])['X'].transform(lambda x: x.fillna(x.mean()))
#         analytical_file_NY_prep.to_csv('analytical_file_NY_prep.csv')
#         print('analytical_file_NY_prep saved')
        
        analytical_file_NY_prep['VariableSplit'] = analytical_file_NY_prep[self.ls + ['Month', 'Year']]\
                                        .apply(lambda x: "_".join(x.values.tolist()), axis = 1)
        
        analytical_file_NY_prep['VariableSplit']  = analytical_file_NY_prep['VariableSplit'].str.replace(' ', '_')
        analytical_file_NY_prep['X_'] = analytical_file_NY_prep['X']
        # print(analytical_file_NY_prep.columns)
        analytical_file_NY_prep = analytical_file_NY_prep.fillna(0)

        analytical_file_NY = pd.pivot_table(analytical_file_NY_prep, index =[i for i in analytical_file_NY_prep.columns if i not in ['VariableSplit', 'X_']],\
                             columns='VariableSplit', values='X_', aggfunc='sum').reset_index().rename_axis(None, axis=1)
        # print(analytical_file_NY.columns)
        analytical_file_NY = analytical_file_NY.replace(np.nan, 0)
        
        analytical_file_NY['Weekend'] = analytical_file_NY['ds'].apply(lambda x: 1 if x.dayofweek > self.weekstarting else 0)
        analytical_file_NY['Month'] = pd.to_datetime(analytical_file_NY['ds']).dt.strftime('%b')
        analytical_file_NY['Year'] = pd.to_datetime(analytical_file_NY['ds']).dt.strftime('%Y').astype(str)
        
        self.analytical_file_NY = analytical_file_NY 
        
        return self.analytical_file_NY
    
    def SaveFile(self, analytical_file_name, folder_name):
        """
        Function used to generate the analytical file.
        
        Parameters:
        -------
        analytical_file_name: string
            File name for generated analytical file
        folder_name: string
            
        Returns:
        -------
            None
        """
        
        SaveToGCS(self.analytical_file_NY, self.bucket, analytical_file_name, folder_name)
        print(analytical_file_name + ' saved')

class FloorCapGenerator:
    """
    Generator that produces the floor and cap for prophet data.
    """
    def __init__(self) -> None:
        pass
    
    def fit(self, df):
        pass
    
    def generate(self):
        pass

class BreakdownSettingFileGenerator:
    """
    Generator that used to create breakdowns along with model settings.
    """
    def __init__(self) -> None:
        pass
    def _AddForecaster_Name_List(self):

        if len(self.ls) == 4:
            self.cuts_df['forecaster_name_list'] = self.cuts_df[self.ls[0]] + ' ' + self.cuts_df[self.ls[1]] + '\
                                            ' + self.cuts_df[self.ls[2]] + ' ' + self.cuts_df[self.ls[3]]+ '.pkl'

        elif len(self.ls) == 3:
            self.cuts_df['forecaster_name_list'] = self.cuts_df[self.ls[0]] + ' ' + self.cuts_df[self.ls[1]] + \
                                                ' ' + self.cuts_df[self.ls[2]] + '.pkl'

        elif len(self.ls) == 2:
            self.cuts_df['forecaster_name_list'] = self.cuts_df[self.ls[0]] + ' ' + self.cuts_df[self.ls[1]]  + '.pkl'

        elif len(self.ls) == 1:
            self.cuts_df['forecaster_name_list'] = self.cuts_df[self.ls[0]]  + '.pkl'
            
    def Generate(self, df, config_dict, cuts_df = None):
        """
        Parameters:
        -------
            df: DataFrame
                Raw data.
            config_dict: dictionary
                Config settings.
            cuts_df: DataFrame
                If existing cuts_df is provided, then information from raw data isn't populated.
        
        """
        self.ls = [query_item for query_item in [config_dict["CLIENTINFO"]["breakdown1"], \
                                     config_dict["CLIENTINFO"]["breakdown2"], \
                                     config_dict["CLIENTINFO"]["breakdown3"], \
                                     config_dict["CLIENTINFO"]["breakdown4"]]
                                     if query_item != None]
        try:
            # assert(cuts_df, pd.DataFrame)
            assert isinstance(None, pd.DataFrame)
            self.cuts_df = cuts_df
        except:
            if cuts_df == None:
                self.cuts_df = df[self.ls].drop_duplicates()
                self.cuts_df.sort_values(self.ls, inplace = True)
                self.cuts_df = self.cuts_df.reset_index(drop = True)
                self.cuts_df['cut_off_point'] = config_dict['PROPHETSETTING']['cut_off_point']
            else:
                raise ErrorType('cuts_df should be None or a pd.DataFrame object.')
        self._AddForecaster_Name_List()
        
        
    def Update(self, setting_dict = None):
        if setting_dict != None:
            for i in setting_dict:
                self.cuts_df[i] = setting_dict[i]
        return self.cuts_df
    
    def SaveFile(self, bucket_name, folder_name, file_name = None):

        if file_name is None:
            file_name = 'cuts_df.csv'
            
        assert isinstance(file_name, str)
        
        if ".csv" not in file_name:
            file_name = 'cuts_df.csv'
            
        SaveToGCS(self.cuts_df, bucket_name, file_name, folder_name)

class ConstraintFileGenerator():
    """
    Constraint file is used to set up the boundary for PSO.
    """
    def __init__(self, config_dict, analytical_file_df) -> None:
        self.input = config_dict['PROPHETSETTING']['input']
           
        self.query_item_list = [query_item for query_item in [config_dict["CLIENTINFO"]["breakdown1"], \
                             config_dict["CLIENTINFO"]["breakdown2"], \
                             config_dict["CLIENTINFO"]["breakdown3"], \
                             config_dict["CLIENTINFO"]["breakdown4"]]
                             if query_item != None]

        self.unique_combination = analytical_file_df[self.query_item_list + ['Month', 'Year']].drop_duplicates().reset_index(drop = True)
        
    def Generate(self, bounds):
        """
        Generate constrains data frame based on the input bounds.
        
        Parameters:
        -------
            bounds: pd.DataFrame
                Lower and upper bounds for all splits.
                Must contain columns MinWeight and MaxWeight.
                
        Returns:
        -------
            constraint_file: pd.DataFrame
        """
        assert isinstance(bounds, pd.DataFrame)
        constraint_file = self.unique_combination.copy()
        self.unique_combination['Year'] = self.unique_combination['Year'].astype(str)

        constraint_file['variable_splits'] = self.unique_combination.agg('_'.join, axis=1)

        constraint_file['variable_splits'] = constraint_file['variable_splits'].str.replace(' ', '_')
        constraint_file['Constrain_Variable_Name'] = self.input
        # constraint_file['MinWeight'] = 0.8
        # constraint_file['MaxWeight'] = 1.2
#         print(constraint_file)
        self.constraint_file = pd.merge(constraint_file, bounds, on = self.query_item_list)

        return self.constraint_file
    
    def SaveFile(self, bucket_name, folder_name, file_name = "Constraint File.csv", base_path: str = "", df: pd.DataFrame = None):
        if df is not None:
            self.constraint_file = df.copy()
            
        if file_name is None or ".csv" not in file_name:
            file_name = 'Constraint File.csv'
        else:
            assert isinstance(file_name, str)
        
        constraint_file_name = base_path + file_name
        
        SaveToGCS(self.constraint_file, bucket_name, file_name, folder_name)
        self.constraint_file.to_csv(constraint_file_name, index = False)

class PlotAndSaveAccuracyDf():
    def __init__(self):
        pass
    
    def Plot(self, df_for_plt):
        """
        Function to plot data frame.
        
        Parameters:
        -------
            df_for_plt: pd.DataFrame
                DataFrame to be plot.
        Returns:
             
        """
        
        fig, ax = plt.subplots(figsize=[10,3])#, layout="constrained"
        for i in df_for_plt.columns:
            ax.plot(df_for_plt[i]/100.0,label=str(i).replace("'", ""))
        ax.set_title("Model Error Rate")
        ax.set_ylabel("Monthly Percentage Error")
        ax.set_xlabel("Month")
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
        
        self.fig = fig
        return fig

    def SaveToGCS(self, bucket_name,  write_to_file_name, folder_name):
        SaveToGCS(self.fig, bucket_name,  write_to_file_name, folder_name)
