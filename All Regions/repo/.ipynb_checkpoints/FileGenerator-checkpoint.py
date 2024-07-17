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
import numpy as np # for numeric calculations and array structures
import datetime # for date variables 
import matplotlib.pyplot as plt # to create plots 
import matplotlib.ticker as mtick # customize tick location and format on plots

from .FileSaving import * # for custom file saving 
from .QueryingData import * # for custom Querying data from Google Big Query


class AnalyticalFileGenerator:
    """
    Generate analytical file based on raw data. 
    Creates data for the future forecasting period based on previous year data 
    Analytical file is saved on local.
    
    Attributes:
    --------
    config_dict: dic
            A dictonary of the content of config.ini file.
    
    Returns:
    -------
    analytical_file: pd.DataFrame
        analytical file used for prophet forecast
    """


    def __init__(self, config_dict, df = pd.DataFrame ) -> None:
        """
        Initialize the analytical file generator. It sets and saves inside the class the config dict and the breakdown columns 
        
        Attributes:
        -------
        config: dic
            A dictonary of the content of config.ini file.
        df: pd.DataFrame
            a data frame with the seaarch historical data, for not querying data from BQ. If not provided, in query search historical data from BQ

        Return:
        -------
            None
        """

        # ensure the config_dict object is a dict, if not raise an error
        if not isinstance(config_dict, dict):
            raise TypeError("config_dict must be a dictionary")
        

        # save in class self the config dict
        self.config_dic = config_dict

        # save in class self some variables that are stored in the config_dict for an easier data manipulation and code writing 
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
        
        # create a list of the column names of the breakdowns in the data frame 
        self.ls = [config_dict['CLIENTINFO'][i] for i in sorted([key for key in config_dict['CLIENTINFO'] if key.startswith('breakdown')]) if config_dict['CLIENTINFO'][i] != None]

        
        # save inside the class the inputed dataframe, if not inputed, the df is an empty pd.DataFrame
        self.df = df
        

    def generate(self): #, df = pd.DataFrame()
        """
        Function used to generate the analytical file in dataframe format.
        Creates data for the future forecasting period based on previous year data 
        
        Returns:
        -------
        analytical_file: pd.DataFrame
            analytical file used for prophet forecast
        """
        

        # if the df stored is empty, it means there is no data passed, then it pulls the data from BQ with the analytical_file_query in config_dict
        if self.df.empty:
            # pull the data from BQ with analytical_file_query query and store it as search_df 
            search_df = GetBQData(self.config_dic['PROPHETSETTING']['analytical_file_query'], self.input, self.output)
        
        # if df is not empty, store the data as search_df
        else:
            search_df = self.df
        
        # filter data before the cutoff point, this prevents using forecasting period as reference dates
        search_df = search_df[search_df['ds'] < self.cut_off_point].reset_index(drop=True)


        # filter data to just have data between the first reference start and reference end date, droping y column which is the target variable to forecast.
        analytical_file_prep = search_df[(search_df['ds'] >= self.reference_start_date1) & \
                                            (search_df['ds'] <= self.reference_end_date1)].drop('y', axis = 1)

        # move dates base_periods forward. this makes the data frame to have the forecasting period dates 
        analytical_file_prep['ds'] = analytical_file_prep['ds']+datetime.timedelta(days=self.base_periods)

        # create a month column that identifies the month of the date in the df 
        analytical_file_prep['Month'] = pd.to_datetime(analytical_file_prep['ds']).dt.strftime('%b')

        # create a year column that identifies the year of the date in the df 
        analytical_file_prep['Year'] = pd.to_datetime(analytical_file_prep['ds']).dt.strftime('%Y').astype(str)


        # if there is a second reference date, use it to move forward dates, and then concat to the df generated with the first reference date 
        if isinstance(self.reference_start_date2, datetime.datetime) and isinstance(self.reference_end_date2, datetime.datetime):

            # filter data to just have data between the first reference start and reference end date, droping y column which is the target variable to forecast.
            analytical_file_prep2 = search_df[(search_df['ds'] >= self.reference_start_date2) & \
                                                (search_df['ds'] <= self.reference_end_date2)].drop('y', axis = 1)

            # move dates base_periods forward. this makes the data frame to have the forecasting period dates 
            analytical_file_prep2['ds'] = analytical_file_prep2['ds']+datetime.timedelta(days=self.base_periods)

            # create a month column that identifies the month of the date in the df 
            analytical_file_prep2['Month'] = pd.to_datetime(analytical_file_prep2['ds']).dt.strftime('%b')

            # create a year column that identifies the year of the date in the df 
            analytical_file_prep2['Year'] = pd.to_datetime(analytical_file_prep2['ds']).dt.strftime('%Y').astype(str)
            
            # concat the new analytical file generated with the second reference dates with the first analytical file, generated with the first reference dates 
            analytical_file_prep = pd.concat([analytical_file_prep, analytical_file_prep2])


        # replace Nan values with negative values of x (cost) with Nan value 
        analytical_file_prep['X'] = analytical_file_prep.X.apply(lambda a: np.nan if a <= 0 else a)

        # replace Nan values in X (cost) with the average value of the breakdown 
        analytical_file_prep['X'] = analytical_file_prep.groupby(self.ls)['X'].transform(lambda x: x.fillna(x.mean()))
        
        # create the VariableSplit column a join of the breakdowns and the month and year columns 
        analytical_file_prep['VariableSplit'] = analytical_file_prep[self.ls + ['Month', 'Year']].apply(lambda x: "_".join(x.values.tolist()), axis = 1)
        
        # replace VariableSplit blank spaces ' ' with under score '_'
        analytical_file_prep['VariableSplit']  = analytical_file_prep['VariableSplit'].str.replace(' ', '_')
        

        # pivot wider X (cost) for each VariableSplit
        
        # create a copy of X column 
        analytical_file_prep['X_'] = analytical_file_prep['X']

        # fill na values with 0 
        analytical_file_prep = analytical_file_prep.fillna(0)

        # pivot wide the table using as values X_ and as new columns the VariableSplit values
        analytical_file = pd.pivot_table(analytical_file_prep, index =[i for i in analytical_file_prep.columns if i not in ['VariableSplit', 'X_']],\
                             columns='VariableSplit', values='X_', aggfunc='sum').reset_index().rename_axis(None, axis=1)

        # fill na values with 0 
        analytical_file = analytical_file.replace(np.nan, 0)
        
        # create a column that identifies weekend dates 
        analytical_file['Weekend'] = analytical_file['ds'].apply(lambda x: 1 if x.dayofweek > 4 else 0)

        # create a month column that identifies the month of the date in the df 
        analytical_file['Month'] = pd.to_datetime(analytical_file['ds']).dt.strftime('%b')

        # create a year column that identifies the year of the date in the df 
        analytical_file['Year'] = pd.to_datetime(analytical_file['ds']).dt.strftime('%Y').astype(str)
        
        # save the analytical file in the class self 
        self.analytical_file = analytical_file 
        
        # returns the the analytical file 
        return self.analytical_file
    
    def SaveFile(self, analytical_file_name, folder_name):
        """
        Function used to save the anlytical file to GCS .
        
        Parameters:
        -------
        analytical_file_name: string
            File name for generated analytical file
        folder_name: string
            name of the folder in GCS where the analytical file should be stored
            
        Returns:
        -------
            None
        """

        # execute the function that saves the analytical file to GCS
        SaveToGCS(self.analytical_file, self.bucket, analytical_file_name, folder_name)

        # print that the analytical file has been saved, used to ensure the process was done correctly 
        print(analytical_file_name + ' saved')



class ConstraintFileGenerator():
    """
        Generates de constraint file, which is used to set up the weight boundaries for PSO optimization 
        
    """


    def __init__(self, config_dict, analytical_file_df) -> None:
        """
        Initialize the constraint file generator. It sets and saves inside the class the config dict, the breakdown columns and breakdown data frame 
        
        Attributes:
        -------
        config: dic
            A dictonary of the content of config.ini file.
        df: pd.DataFrame
            a data frame with the seaarch historical data, for not querying data from BQ. If not provided, in query search historical data from BQ

        Return:
        -------
            None
        """

        # save the name of the input variable     
        self.input = config_dict['PROPHETSETTING']['input']
           
        # create a list of the column names of the breakdowns in the data frame 
        self.query_item_list = [config_dict['CLIENTINFO'][i] for i in sorted([key for key in config_dict['CLIENTINFO'] if key.startswith('breakdown')]) if config_dict['CLIENTINFO'][i] != None]

        # get the breakdown data frame, with a row for each brakdown with its brakdown columns 
        self.unique_combination = analytical_file_df[self.query_item_list + ['Month', 'Year']].drop_duplicates().reset_index(drop = True)

        # ensure the year column is a str type to create the breaown name (variable_splits)
        self.unique_combination['Year'] = self.unique_combination['Year'].astype(str)
        

    def Generate(self, bounds):
        """
        Generate constrain_file data frame based on the input bounds.
        
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
        
        # get the initial breakdown data frame 
        constraint_file = self.unique_combination.copy()

        # create the variable_splits name column 
        constraint_file['variable_splits'] = constraint_file.agg('_'.join, axis=1)

        # replace the blank spaces ' ' with under scores '_' in the variable splits column
        constraint_file['variable_splits'] = constraint_file['variable_splits'].str.replace(' ', '_')

        # create the Constrain_Variable_Name which is the name of the target variable 
        constraint_file['Constrain_Variable_Name'] = self.input

        # merge the initial contraint file to the wheight bounds to get the final constraint file that has the MinWeight and MinWeight columns 
        self.constraint_file = pd.merge(constraint_file, bounds, on = self.query_item_list)

        # returns the constraint file 
        return self.constraint_file
    
    def SaveFile(self, bucket_name, folder_name, file_name = "Constraint File.csv", base_path: str = "", constraint_file: pd.DataFrame = None):
        """
        Saves the constraint file into a GCS bucket and in a local file located in base_path 
        
        Parameters:
        -------
            bucket_name: string 
                string that identifies the bucket name in GCS
            folder_name: string 
                string that identifies the folder name inside the bucket in GCS
            file_name: string 
                string name that is used to save the file as 
            base_path: string 
                string that identifies the path in the local device where it should be located the file 
            constraint_file: pd.DataFrame 
                DataFrame of the constraint file. if is not passed, it uses the constriant file generated by the class function Generate() before
                
        """

        # if constraint_file is not passed, use the stored self.constraint_file
        if constraint_file is not None:
            self.constraint_file = constraint_file.copy()

        # if file_name is str type use file_name input argument, if not use 'Constraint File.csv' as file_name
        if isinstance(file_name, str):
            # if it ends with '.csv'  use the file_name if not use 'Constraint File.csv' as file_name
            if file_name.endswith('.csv') and file_name.count('.csv')==1:
                file_name = file_name 
            else:
                file_name = 'Constraint File.csv'
        else: 
            file_name = 'Constraint File.csv'

        # Save the constraint file in the local device 
        self.constraint_file.to_csv(base_path + file_name, index = False)
        
        # save the constraint file to GCS 
        SaveToGCS(self.constraint_file, bucket_name, file_name, folder_name)
        




class PlotAndSaveAccuracyDf():
    """
        Function to plot data frame.
        
        Parameters:
        -------
            df_for_plt: pd.DataFrame
                DataFrame to be plot.
        Returns:
             
    """
    
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
             fig: matplotlib.pyplot 
        """
        
        # Create a figure and a single subplot with a specific size.
        fig, ax = plt.subplots(figsize=[10,3]) # Width 10 inches and height 3 inches.

        # Loop over each column in the DataFrame to plot them.
        for i in df_for_plt.columns:

            # Plot the column data after dividing by 100 to convert to percentage form. Plot the column data after dividing by 100 to convert to percentage form.
            ax.plot(df_for_plt[i]/100.0,label=str(i).replace("'", ""))

        # Set the title of the plot.
        ax.set_title("Model Error Rate")

        # Set the y-axis label.
        ax.set_ylabel("Monthly Percentage Error")

        # Set the x-axis label.
        ax.set_xlabel("Month")

        # Format the y-axis tick labels to show values as percentages.
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

        # Place a legend outside the plot area on the bottom left.
        plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)

        # Save the figure object in the class attribute if within a class method.
        self.fig = fig

        # return fig
        return fig

    def SaveToGCS(self, bucket_name,  write_to_file_name, folder_name):
        """
        Function to save the plot to GCS.
        
        Parameters:
        -------
            bucket_name: string
                string that identifies the bucket name in to store the plot in GCS
            write_to_file_name: string 
                string that identifies how the file to store would be named 
            folder_name: string 
                string that identifies the name of the folder inside the bucket in GCS, where the file should be stored.
        Returns:
        """
        
        # save the plot to GCS
        SaveToGCS(self.fig, bucket_name,  write_to_file_name, folder_name)




class BreakdownSettingFileGenerator:
    """
    BreakdownSettingFileGenerator is used to create a pd.DataFrame with a row for each of the breakdowns along with model settings.
    """
    def __init__(self) -> None:
        pass
            
    def Generate(self, df, config_dict, cuts_df: pd.DataFrame = None):
        """
        Function that generates the cuts_df data frame, that has a row for each breakdown along with its model settings. 

        Parameters:
        -------
            df: DataFrame
                Historical search raw data.
            config_dict: dictionary
                Config settings.
            cuts_df: DataFrame
                If existing cuts_df is provided, then information from raw data isn't populated.
            
        Returns:
        -------
            cuts_df: pd.DataFrame 
                Data Frame that has a row for each breakdown along with its model settings. 
        
        """

        # create a list of the column names of the breakdowns in the data frame 
        self.ls = [config_dict['CLIENTINFO'][i] for i in sorted([key for key in config_dict['CLIENTINFO'] if key.startswith('breakdown')]) if config_dict['CLIENTINFO'][i] != None]

        # if cuts_df is None, it indicates that cuts_df is not passed as an argument, then it will create the cuts_df from the historical search data 
        if cuts_df is None:
            
            # creates the cuts_df DataFrame, which has initially the breakdown columns. drop duplicates to have just one row per breakdown
            self.cuts_df = df[self.ls].drop_duplicates().reset_index(drop=True)

            # sort rows in cuts_df using the breakdown columns hierarchy 
            self.cuts_df.sort_values(self.ls, inplace = True)

            # reset index in cuts_df
            self.cuts_df = self.cuts_df.reset_index(drop = True)

            # add a column named 'cut_off_point' which is set on the config_dict
            self.cuts_df['cut_off_point'] = config_dict['PROPHETSETTING']['cut_off_point']

        # if cuts_df is passed as an argument, store it in the class attribute if within the class method. 
        else:
            self.cuts_df = cuts_df
        
        # create the column 'forecaster_name_list' which is the file name by which the model for each breakdown would be stored in local device. name is the join of the breakdown column values, that uniquely identifies each breakdown
        self.cuts_df['forecaster_name_list'] = (' '.join(self.cuts_df[ls] for ls in self.ls) + '.pkl').astype(str)
        
        # return cuts_df
        return self.cuts_df
        
        
    def Update(self, setting_dict = None):
        """
        Function that updates the cuts_df data frame.

        Parameters:
        -------
            setting_dict: dictionary
                dictionary that has the model hyper parameters

        Returns:
        -------
            cuts_df: pd.DataFrame 
                Data Frame that has a row for each breakdown along with its model settings. 
        """
        
        # if setting_dict is passed as an argument update the cuts_df if not just returns the stored in class attribute cuts_df 
        if setting_dict != None:
            
            # loop over each setting_dict key 
            for i in setting_dict:

                # creates a colum with the key name, which has the key element value as a value of the column 
                self.cuts_df[i] = setting_dict[i]

        # return cuts_df
        return self.cuts_df
    
    def SaveFile(self, bucket_name, folder_name, file_name = None):
        """
        Function to save the cuts_df to GCS.
        
        Parameters:
        -------
            bucket_name: string
                string that identifies the bucket name in to store the file in GCS
            write_to_file_name: string 
                string that identifies how the file to store would be named 
            folder_name: string 
                string that identifies the name of the folder inside the bucket in GCS, where the file should be stored.
        Returns:
        """

        # if file name is not passed as an argument use a default file name 'cuts_df.csv'
        if file_name is None:
            file_name = 'cuts_df.csv'
            
        # Assert if the file_name part is a string, if not it raises an error 
        assert isinstance(file_name, str)
        
        # if file_name str does not ends with '.csv' it uses a default file name 'cuts_df.csv'
        if not file_name.endswith('.csv'):
            file_name = 'cuts_df.csv'
            
        # save the plot to GCS
        SaveToGCS(self.cuts_df, bucket_name, file_name, folder_name)




''' Other class functions that arent working right now'''

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




