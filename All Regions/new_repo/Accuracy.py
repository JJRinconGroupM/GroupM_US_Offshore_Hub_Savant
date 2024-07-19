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
import warnings # to get or supress warning messages 
from tqdm import tqdm # to implement a progress bar in a loop 
import itertools # to create iterative elements 
from pathos.multiprocessing import ProcessingPool as PPool # for parallel multiprocessing 
import pathos as pa # for parallel multiprocessing 
import re # regular expressions 
import random # random selection for selecting a seed in random searches 

from .BuildModel import CreateModel, forecast_model_predict # for custom prophet model creation and forecasting 
from .FileSaving import * # for custom file saving 




def mean_absolute_percentage_error(y_pred, y_true): 
    """
        this function calculates the MAPE (Mean Absolute Percentage Error) based on the y_pred series and the y_true (observed) series

            APE = abs(y_true - y_pred) / abs(y_pred)
            MAPE = mean(APE)
        
        Parameter:
        -------
            y_pred: series
                series of predicted values. predicted values from the prophet forecast
            y_true: series
                series of observed values. observed values from historical search daily data 
                
        Returns:
        -------
            MAPE: float number
                the MAPE calculation result
    """

    # ensure y_true and y_pred are np.array type 
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # perform the absolute difference between the y_true and y_pred
    diff = np.abs(y_true - y_pred)

    # perform the absolute percentage error = diff / y_true 
    APE = np.where(y_true != 0, diff / np.abs(y_true), 0)

    # perform the MAPE, mean of the absolute precentage error. multipliy by 
    MAPE = APE.mean()

    # multiply the MAPE by 100 to have percentage unit values instead of proportions, and round that number by 2 decimals
    MAPE = round(MAPE * 100, 2)

    # returns the MAPE 
    return MAPE




def monthly_MAPE_calcuation(forecast_df, actual_df, start_date):
    """
        Function that calculates the MAPE (Mean Absolute Percentage Error) and the forecast period percentage error based on the forecast values in the forecast_df and the observed values in the actual_df

            MAPE = mean ( abs(y_true - y_pred) / abs(y_pred) ) 
            forecast perion APE = ( sum(y_true) - sum(y_pred) ) / sum(y_pred) ) 
        
        Parameter:
        -------
            forecast_df: pd.DataFrame
                pd.DataFrame that must have the predicted values 'yhat' from the prophet forecast and a date column 'ds.
            actual_df: pd.DataFrame
                pd.DataFrame that must have the observed values 'y' from the historical search data and a date column 'ds.
            start_date: datetime.dateime
                a date that identifies the month date to perform the MAPE over the predicted values of that month 
                
        Returns:
        -------
            MAPE: float number
                the MAPE calculation result
            period_APE: float number 
                the forecasting period percentage error calculation result 
            joined_df: pd.DataFrame
                a pd.DataFrame that has the 'yhat' and 'y' values for the common year_month dates 
    """

    # set start date in case start_date it's not month start. Start_month is usde to filter forecast 
    start_month = datetime.datetime(start_date.year, start_date.month, 1)
     
    # group forecast dataframe (monthly_forecast) to a monthly level, only leaving the ds and yhat columns 
    monthly_forecast = forecast_df[['ds', 'yhat']].groupby(pd.Grouper(key="ds", freq="MS", sort = True)).sum().reset_index()

    # group observed data frame (actual_df) to a monthly level, only leaving the ds and y columns 
    monthly_actual = actual_df[['ds', 'y']].groupby(pd.Grouper(key="ds", freq="MS", sort = True)).sum().reset_index()
    
    # merge (inner join) the monthly forecast and observed data frames by date (ds)
    joined_df = pd.merge(monthly_actual, monthly_forecast, how = 'inner', on='ds')

    joined_df = joined_df[joined_df['ds'] >= start_month].copy()

    # create the percentage error column, one for each forecast value
    joined_df['Percentage Error'] = ((joined_df['y'] - joined_df['yhat'])/joined_df['y']).round(4) * 100
    

    # get the mean absolute percentage error 
    MAPE = mean_absolute_percentage_error(joined_df['yhat'] , joined_df['y'])
    
    # set a MAPE column with the MAPE value on the joined_df dataframe
    joined_df['MAPE'] = MAPE

    # get the forecasting period absolute percentage error. sum the forecast (yhat) and the observed value (y) for the whole period 
    period_APE = mean_absolute_percentage_error(joined_df['yhat'].sum(), joined_df['y'].sum())
    
    #period_APE = (np.abs(joined_df.loc[joined_df['ds']>=start_month, 'yhat'].sum() - joined_df.loc[joined_df['ds']>=start_month, 'y'].sum())/joined_df.loc[joined_df['ds']>=start_month, 'y'].sum()).round(4) * 100

    # set a forecasting period percentage error column with the forecasting period APE value on the joined_df dataframe
    joined_df['Forecast Period APE'] = period_APE

    # get the forecasting period error 
    forecast_error = joined_df['yhat'].sum() - joined_df['y'].sum()
    
    # set a prediction status,
    #   if the forecasting period error is positive it means the forecast is over predicting the observed value. On the contrary, 
    #   if forecasting period error is negative it means the forecast is underpredicting the observed value. 
    #   While if the forecasting period error is 0 it means it is exactly the same value, something strange and should be checked 
    Predict_Status = np.where(forecast_error > 0, 'OverPredicted', np.where(forecast_error == 0, 'Exact The Same. Check Needed.', 'UnderPredicted') )

    # set a prediction status column with the on the joined_df dataframe
    joined_df['Predict Status'] = Predict_Status

    '''
    # Print the prediction status 
    print(Predict_Status)

    # print the MAPE and the forecasting period percentage error   
    print('MAPE:', MAPE, "Forecasting Period percentage error:", (period_APE))
    '''

    # return a list of the MAPE, forecasting period percetage error and the final joined_df (join by date of y and yhat)
    return MAPE, period_APE, joined_df





class AccuracyCalculator:
    """
    This class calculates the model out of sample accuracy, leaving the last n observable periods for accuracy calculation. 
    That is filtering the training dataset removing the last n periods, and compare those observed values with the model forecast for those periods. 
    It generates and train the model with its hyper parameters and training dataset (

    Accuracy scores are the forecasting Period monthly MAPE and the Forecasting Period percentage error
    
    Attributes:
    -------
        self.values: model settings stored in pandas series.
    
    Returns:
    -------
        None
    """

    # Initialize the model 
    def __init__(self) -> None:
        pass
        

    def CalculateAccuracy(self, cuts_df, df, breakdown_ls, customized_holidays):

        """
        This function calculate the model accuracy scores.
        calculates the model out of sample accuracy, leaving the last n observable periods for accuracy calculation. 
        That is filtering the training dataset removing the last n periods, and compare those observed values with the model forecast for those periods.

        Accuracy scores are the forecasting Period monthly MAPE and the Forecasting Period percentage error
    
        Parameters:
        -------
            Cuts_df: pd.DataFrame
                fbprophet model settings for each breakdown model. each row is a breakdown model
            df: pd.DataFrame
                the search historical dataframe that is used to train the model
            Breakdown_ls: list
                list of breakdown columns, which identifies each breakdown to be modeled.
            customized_holidays: dict
                a dictionary that has custom holidays for each market in the breakdown. The dict keys must be the market name, and the element must be a custom holidays dataframe 
    
        Attributes:
        -------
            self.values: model settings stored in pandas series.
    
        Returns:
        -------
            accuracy_df: pd.DataFrame
                A monthly frequency dataframe with the actual, forecast and Accuracy scores for each breakdown model. 
        """

        # ensure that all breakdown markets that have as a hyper parameter include_holiday as 'custom' or 'customized' they have the custom holidays dataframe in the customized_holidays dict 
        
        # iterate over each market that has a cuts_df['include_holiday'] as ['custom','customized']
        for m in cuts_df.loc[ cuts_df['include_holiday'].isin(['custom', 'customized']), 'market' ].unique():

            # if the market is not in customized_holidays.keys() and is not SM (since SM does not have neither defaul nor custom holidays) raise an error
            if m not in customized_holidays.keys() and m != 'SM':
                
                # raise aa ValueError saying that that market does not have a custom holiday set
                ValueError(f'There are no custom holidays for market {m} in custom_holidays dict argument')

        # save in class self the cuts_df 
        self.cuts_df = cuts_df

        # save in class self the df 
        self.df = df

        # save in class self the breakdown_ls 
        self.breakdown_ls = breakdown_ls
        
        # create an empty list to store breakdown accuracy dataframes        
        self.monthly_acc_list = []
        
        # iterate over each breakdown model 
        for key, values in tqdm(list(cuts_df.iterrows()), total=cuts_df.shape[0]): 

            # create a list of the breakdown identifier  
            breakdown =[f"'{col}' = '{values[col]}'" for col in self.breakdown_ls]
                
            # print the breakdown identifier to know in which breakdown the search is 
            print(f"Breakdown: {' & '.join(breakdown)}")

            # create the model_df that initially is a copy of the df 
            model_df = self.df.copy()

            # create a query filtering conditions for every of the breakdown columns that need to be filtered for create the breakdown data set 
            query = ' & '.join([f"(model_df['{col}'] == '{values[col]}')" for col in breakdown_ls])

            # filter the dataframe with the query to create the breakdown data set 
            model_df = eval(f"model_df[{query}]").reset_index(drop=True)

            # set which is the new cutoff_point, the max date on the training dataframe - the periods for accuracy calculation 
            new_cut_off_point = model_df['ds'].max() - datetime.timedelta(days = values.periods_for_acc_cal)
            
            # try to filter the model_df dataframe leaving dates after the model starting_date and before the new_cut_off_point
            try:

                # create the training_dataset as the filtered model_df with dates between the starting date and the new_cut_off_point 
                training_dataset = model_df[ (model_df['ds'] >= values.starting_date) & (model_df['ds'] < new_cut_off_point) ]
            
            # if its not possible it means there is no starting_date, then model_df dataframe leaving dates before the new_cut_off_point
            except:

                # create the training_dataset as the filtered model_df with dates before the new_cut_off_point 
                training_dataset = model_df[(model_df['ds'] < new_cut_off_point)]
                
            # create and train the prophet model with the model hyper parameters set on values and the market customized_holidays
            model  = CreateModel(training_dataset = training_dataset, params = values, customized_holidays = customized_holidays)

            # create the forecast with the model 
            forecast = forecast_model_predict(model, model_df, params = values, forecasting_periods = values.periods_for_acc_cal, future_input_df = model_df)

            # calculate the forecasting period monthly MAPE and forecasting percentage error, and get a monthly forecast df with the monthly percentage error and forecasting period monthly MAPE and forecasting period percentage error
            MAPE, period_APE, monthly_acc_df_temp = monthly_MAPE_calcuation(forecast_df = forecast, actual_df = model_df, start_date = new_cut_off_point)         

            # rename columns 
            monthly_acc_df_temp.rename(columns = {'ds': 'Month', 'y': 'Actual', 'yhat': 'Forecast'}, inplace = True)

            # Iterate over each of the breakdown columns and insert the column and column value on the monthly_acc_df_temp
            for j in range(len(breakdown_ls)):

                # insert the breakdown column value in a column names as the breakdown column on the monthly_acc_df_temp
                monthly_acc_df_temp.insert(0, breakdown_ls[j], values[breakdown_ls[j]])
                
            # append the monthly_acc_df_temp of the breakdown with the complete monthly_acc_df, that have all the breakdowns 
            self.monthly_acc_list.append(monthly_acc_df_temp)
        
        # convert the monthly_acc_list to a dataframe with pd.concat()
        self.monthly_acc_df = pd.concat(self.monthly_acc_list, ignore_index=True)
            
        # return the self.monthly_acc_df
        return self.monthly_acc_df        


    def SaveFile(self, bucket_name = None, folder_name = None, write_to_file_name = "Model Scores.csv"):
        """
        Function used to save the anlytical file to GCS and to the local device.
        
        Parameters:
        -------
        bucket_name: string
            name of the bucket of GCS where the accuracy_df should be stored
        folder_name: string
            name of the folder in GCS where the accuracy_df should be stored
        write_to_file_name: string
            name of the file name where the accuracy_df is going to be saved. default "Model Scores.csv"
            
        Returns:
        -------
            None
        """

        # tries to save the accuracy_df to a csv file in the local device 
        try:

            #save the accuracy_df to a csv file in the local device 
            self.monthly_acc_df.to_csv(write_to_file_name, mode='w', index=False)

        # if the file could not be saved on the local device, print a message 
        except:

            # print a message that the file could not be stored in local device 
            print("Save to local is skipped. An error occurred.")
            pass

        # tries to save the accuracy_df to a csv file in GCS
        try:
            
            # save the accuracy_df to a csv file in the GCS device 
            SaveToGCS(self.monthly_acc_df_temp, bucket_name, write_to_file_name, folder_name)

        # if the file could not be saved on the gcs, print a message 
        except Exception:

            # print a message that the file could not be stored in GCS
            print("Save to bucket is skipped. An error occurred.")
            pass








def PlotAccuracyCharts(accuracy_df, cuts_df, query_item_list):
    """
        this function plots the monthly percentage error, alognside with the observed (y) and predicted (yhat) values

            APE = abs(y_true - y_pred) / abs(y_pred)
        
        Parameter:
        -------
            accuracy_df: pd.DataFrame
                The dataframe that is the output of the AccuracyCalculator.CalculateAccuracy(). it must have a monthly frequency, query_item_list as columns as well as 'y', 'yhat' and 'Percentage Error'
            cuts_df: pd.DataFrame
                    A pd.DataFrame that has a row for each breakdown and has the model hyper parameter settings 
            query_item_list: string list 
                    list of strings of the column names that identifies the breakdowns 
                
        Returns:
        -------
    """

    # iterate over each row of cuts_df, over each breakdown 
    for key, values in cuts_df.iterrows():
        
        df = accuracy_df.copy()

        # create a query filtering conditions for every of the breakdown columns that need to be filtered to create the breakdown dataframe to plot its accuracy  
        query = ' & '.join([f"(df['{col}'] == '{values[col]}')" for col in query_item_list])

        # filter the accuracy_df dataset with the query to create the the breakdown dataframe to plot its accuracy  
        df_for_plt = eval(f"df[{query}]").reset_index(drop=True)

        # plot the percentage error, the actual and forecast for the brwakdown accuracy df 
        df_for_plt.set_index('Month').plot(y=['Actual', 'Forecast','Percentage Error'], secondary_y=['Percentage Error'], title=' '.join([f'{values[col]}' for col in query_item_list]))















class ProphetParameterSearch:
    """
    This class calculates tsearch the best parameter setting for Prophet model.
    The best parameter setting is the one that has the lower forecasting period monthly MAPE, 
    which is calculated leaving the last n observable periods for accuracy calculation. 

    User can export model score for futher investigation
    or to run best_option to return the models with highest scores.
    
    Attributes:
    -------
        self.values: model settings stored in pandas series.
    
    Returns:
    -------
        None
    """

    def __init__(self, df, cuts_df, breakdown_ls, param_grid, customized_holidays, base_path='') -> None:
        """
        Initialize ProphetParameterSearch object.
        
        Parameter:
        -------
            param_grid: dict
                a dictionary that has as keys all models hyper parameters and as elements for each key the possible values that those hyper parameters could take.
            linear_flag: Bool, default True
                    Whether or not to turn on the linear flag
            power_flag: Bool, default True
                    Whether or not to turn on the power flag
            logistic_flag: Bool, default True
                    Whether or not to turn on the logistic flag
            base_path: string  
                    string that sets the path where should the history record df should be stored in the local device.
        Returns:
        -------
        """
        
        # create a list of base params, params that should be present mandatory in param_grid
        base_params = ['changepoint_prior_scale','seasonality_prior_scale','holidays_prior_scale','seasonality_mode','input_mode','weekday_or_weekend','growth']

        # Create list of base_params not in param_grid keys
        missing_params = [param for param in base_params if param not in param_grid.keys()]

        # if missing_params has 1 or more elements len()>0 then raise a valueError showing that missing_params should be present on param_grid
        if len(missing_params)>0:
            ValueError('The following base params should be present in param_grid and they are not present', missing_params)

        # save in class self the param_grid
        self.param_grid = param_grid 

        # save in class self the df
        self.df = df
        
        # save in class self the cuts_df
        self.cuts_df = cuts_df 

        # save in class self the breakdown_ls
        self.breakdown_ls = breakdown_ls

        # save in class self the customized_holidays
        self.customized_holidays = customized_holidays

        # set a list of possible values for the power growth if growth is set as power
        power_values = list(np.linspace(0.1**2, 1.1, 5, endpoint=False))

        # set an empty list of params combination 
        self.all_params = []

        # iterate over each possible combination of the param_grid
        for combination in itertools.product(*param_grid.values()):
            
            # set as a dict where each of the combination values is set as a value under the corresponding param_grid.key()
            combo_dict = dict(zip(param_grid.keys(), combination))

            # if the possible combination param dict has 'power' as growth param iterate over power_values creating new possible combinations of combo_dict for each power value
            if combo_dict['growth'] == 'power':

                # If growth is 'power', iterate through power_values
                for power in power_values:
                    
                    # add 'power' param as power number 
                    combo_dict['power'] = power

                    # append the combo_dict to the all_params list of all possible combination params
                    self.all_params.append(combo_dict.copy())

            # if the possible combination param dict does not have 'power' as a growth param, just add the power param as 'na'
            else:

                # add 'power' param as 'na'
                combo_dict['power'] = 'na'

                # append the combo_dict to the all_params list of all possible combination params
                self.all_params.append(combo_dict)
                
        # print a message saying the number of possible combinations of hyper parameters 
        print(f'{len(self.all_params)} possible combinations of hyper parameters')


        # get the time of initialization of the function, this will be use as part of the record_df file name 
        timestr = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

        # set and save as class self the file path for the record_df where it is going to be stored, this df works as a log file for the param search
        self.record_df_file_name = base_path + 'record_df ' + timestr + '.csv'

        # set and save as class self the file path for the best results df, where it is going to be stored, this df shows the final results of the param search (hyper paramaeter setting that creates the min MAPE for each model)
        self.best_record_df_file_name = base_path + 'record_df ' + timestr + ' best result.csv'
        
        # create an empty dataframe for the record_df. the colums needed are the breakdown_ls columns, the columns for each hyper parameter in param_grid plus 'power','growth' and the score columns 'MAPE', 'period_AE'.  save it as a class self attribute 
        self.record_df = pd.DataFrame(columns = breakdown_ls + list(param_grid.keys()) + ['power','MAPE', 'period_APE'] )

        # copy the empty record_df to create the best result df. save it as a class self attribute 
        self.best_record_df = self.record_df.copy()
        
        # create an empty list to store the best_record_df of each breakdown
        self.best_record = []




    def _fit_n_evaluate(self, params): #values, training_dataset, model_df, params, customized_holidays
        """
        Creates and fits prophet model forecasting. 
        Then it performs a prediction for the evaluation period and evaluates the model by the MAPE score
        
        Parameter:
        -------
            values: pd.DataFrame
                a dataframe that has the models breakdown columns and the periods for accuraccy calculation 
            training_dataset: pd.DataFrame
                    a dataframe that has the historical search data that is going to be used to train the model 
            model_df: pd.DataFrame
                    a dataframe that has the full historical search data, it has the data to be used to evaluate the models predictions 
            params: dict
                    a dictionary with the models hyper parameters settings
            customized_holidays: dict  
                    a dictionary that has the customized holidays to include in the model if params['include_holidays'] is set as 'custom' or 'customized'

        Returns:
        -------
            params: pd.DataFrame
                a dataframe with the model hyper parameters, the breadown columns and the model scores. 
        """
        
        # transforms the dict params to a data frame so that it could be passed to the CreateModel() and forecast_model_predict() functions
        params =  pd.DataFrame([params])
    
        # add breakdown columns and breakdown columns values to the params dataframe
        # fore each column in self.breakdown_ls add that column and value to the params dataframe
        for col in self.breakdown_ls:

            # add col as a params column and values[col] as the value of that column
            params[col] = self.values[col]

        # Squeeze params dataframe to create a pandas series, that is the type format of the params argument of CreateModel() function
        params =  params.squeeze()

        # if the market is SM grup and the params setting combination has 'custom' or 'customized' in include_holiday the model could not be done, then it will skip the iteration just adding nan values to MAPE and Period_APE 
        if (self.values['market'] == 'SM') & (str(params.include_holiday).lower() in ['custom','customized']):
            
            # Print the warning message 
            print('SM Market does could not have custom holidays, skipping this hyper parameter setting')   

            # add a nan to MAPE to the params dataframe since the model could not be done
            params['MAPE'] = np.nan

            # add a nan to period_APE to the params dataframe since the model could not be done
            params['period_APE'] = np.nan

        # if it is not the case perfom the continue with the _fit_n_evaluate() function normally 
        else:

            # if inlcude_holiday params is 'custom' or 'customized' then include the customized_holidays as a parameter.
            if str(params.include_holiday).lower() in ['custom','customized']:
                    
                try: 
                
                    # create and train the prophet model with the model hyper parameters set on values and the market customized_holidays
                    model  = CreateModel(training_dataset = self.training_dataset, params = params, customized_holidays = self.customized_holidays)

                # if there is an error in the CreateModel() raise the error. ussually it might be since the market has no customized holidays in customized_holidays dict. 
                except Exception:
                    raise

            # else then do not the customized_holidays as a parameter. this helps to reduce errors of trying to access self.customized_holidays dict for that market when there is no custom holidays for the market in dict.
            else: 
                # create and train the prophet model with the model hyper parameters set on values and the market customized_holidays
                model  = CreateModel(training_dataset = self.training_dataset, params = params)
        
        
            # create the forecast with the model 
            forecast = forecast_model_predict(model, self.model_df, params = params, forecasting_periods = self.values.periods_for_acc_cal, future_input_df = self.model_df)

            # calculate the forecasting period monthly MAPE and forecasting percentage error, and get a monthly forecast df with the monthly percentage error and forecasting period monthly MAPE and forecasting period percentage error
            MAPE, period_APE, joined_df = monthly_MAPE_calcuation(forecast_df = forecast, actual_df = self.model_df, start_date = self.new_cut_off_point) # datetime.datetime(training_dataset.ds.max().year, training_dataset.ds.max().month, 1)



            # add MAPE to the params dataframe
            params['MAPE'] = MAPE

            # add period_APE to the params dataframe
            params['period_APE'] = period_APE

        # transform to a dataframe 
        params =  pd.DataFrame([params])
        
        # try to append append the params dataframe to the record_df file in the local device
        try:

            # try to read the record_df file in the local device 
            pd.read_csv(self.record_df_file_name)

            # try to append the params dataframe to record_df file 
            params.to_csv(self.record_df_file_name, mode='a', header=False)
            
        # if there is an error in trying to read and append the record_df (usually it is because there is no record_df file in local device yet)
        except:
            
            # try to write params dataframe to a .csv file
            try:

                # write params dataframe to a .csv file
                params.to_csv(self.record_df_file_name)

            # if there is an error in writing the csv file raise an exception 
            except Exception:
                raise
        
        # return params dataframe
        return params


        
    def search(self):#, cuts_df, df, param_search_dict = None):
        """
        Performs a grid search over all possible combination of the hyper parameters for each of the breakdown models
        
        Parameter:
        -------

        Returns:
        -------
            record_df: pd.DataFrame
                a log dataframe that records all the accuracy scores (MAPE) and hyper parameters for each models hyper parameters combination for each breakdown model

        """


        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
                    
            # iterate over each breakdown model 
            for key, values in tqdm(list(self.cuts_df.iterrows()), total=self.cuts_df.shape[0]): 
                
                # create a list of the breakdown identifier  
                breakdown =[f"'{col}' = '{values[col]}'" for col in self.breakdown_ls]
                
                # print the breakdown identifier to know in which breakdown the search is 
                print(f"Breakdown: {' & '.join(breakdown)}")

                self.values = values

                # create the model_df that initially is a copy of the df 
                model_df = self.df.copy()

                # create a query filtering conditions for every of the breakdown columns that need to be filtered for create the breakdown data set 
                query = ' & '.join([f"(model_df['{col}'] == '{values[col]}')" for col in self.breakdown_ls])

                # filter the dataframe with the query to create the breakdown data set 
                self.model_df = eval(f"model_df[{query}]").reset_index(drop=True)

                # set which is the new cutoff_point, the max date on the training dataframe - the periods for accuracy calculation 
                self.new_cut_off_point = model_df['ds'].max() - datetime.timedelta(days = values.periods_for_acc_cal)
                
                # create the training_dataset as the filtered model_df with dates before the new_cut_off_point 
                self.training_dataset = model_df[(model_df['ds'] < self.new_cut_off_point)]

                # set a random seed for the parallel processing 
                random.seed(1234)

                
                # Initializes a pool of worker processes matching the number of available CPUs using `PPool`, optimizing resource utilization.
                with PPool(pa.helpers.cpu_count()) as p:

                    # Initialize an empty list to store results from the parallel computation.
                    record_df_ls = []

                    # use `imap` to map the `_fit_n_evaluate` method across all parameter sets specified in `self.all_params` and store them in record_df_ls
                    record_df_ls = p.imap(self._fit_n_evaluate, self.all_params)

                # try to concat the record_df with the record_df_ls, the record df of each parallel process
                try:
                    # con concat the record_df with the record_df_ls
                    self.record_df = pd.concat(record_df_ls)     
                
                # if there is an error in the concat of the record_df raise an exception 
                except Exception:

                    # print the error result, concat error between record_df and record_df_ls
                    print('parallel result concat error.')

                    # raise the error, stop the function 
                    raise
                    
                # create a query filtering conditions for every of the breakdown columns that need to be filtered for create the breakdown data set 
                query_record_df = ' & '.join([f"(self.record_df['{col}'] == '{values[col]}')" for col in self.breakdown_ls])

                # filter the dataframe with the query to create the breakdown data set 
                breakdown_record_df = eval(f"self.record_df[{query_record_df}]").reset_index(drop=True)

                # Get the min MAPE in the breakdown_record_df 
                min_score = breakdown_record_df['MAPE'].min()

                # filter the breakdown_record_df wich has the min_score
                breakdown_best_record_df = breakdown_record_df.loc[breakdown_record_df['MAPE']==min_score]
                
                # append the breakdown_best_record_df to the best_record list 
                self.best_record.append(breakdown_best_record_df)

                # transform the best_record list into a dataframe
                self.best_record_df = pd.concat(self.best_record, ignore_index=True)

                # try to write best_record_df dataframe to a .csv file
                try:

                    # write best_record_df dataframe to a .csv file
                    self.best_record_df.to_csv(self.best_record_df_file_name)

                # if there is an error in writing the csv file raise an exception 
                except Exception:
                    raise

        # return record_df dataframe                        
        return self.record_df














