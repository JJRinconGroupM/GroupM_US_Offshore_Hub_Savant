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
from tqdm import tqdm # to implement a progress bar in a loop 
from prophet import Prophet # for meta prophet forcasting model 
import gcsfs # interface with Google Cloud Storage (GCS)
import pickle # to create binary files that store python objects inside the files
import random

from .FileSaving import * # for custom file saving 



def CreateModel(training_dataset, params, customized_holidays = None):
    """
        prophet Model Generator

        this function crreates the prophet model with the corresponding hyper parameters and additional regressors
        
        Parameter:
        -------
            training_dataset: pd.DataFrame
                search daily data frame. It should have at least a 'ds' (date) column and 'y' (target forecasting) column. it also should have a column for any additional regressor 
            params: pd.series
                pd.series with the hyper parameters for setting the model
                
        Returns:
        -------
            model: prophet.Prophet
                A prophet.Prophet model with hyper parameter settings and that already has fitted the training dataset
    """
        
    # create a dict for the priors hyper parameters and the seasonality mode to pass it into the model setting 
    pt_kwargs = {k: params[k] for k in params.keys() if k in ['changepoint_prior_scale', 'seasonality_prior_scale', 'holidays_prior_scale', 'seasonality_mode']}


    # if the growth param is 'power' transform the X regressor with a params.power and later set the growth param to 'linear'
    if str(params.growth).lower() == 'power':

        # transform the X regressor with a params.power 
        training_dataset.X = np.power(training_dataset.X, float(params.power))

        # set the growth param to 'linear'
        growth = 'linear'

    # if the growth param is not 'power'
    else: 
        # set the growth param as the growth set in the params series
        growth = str(params.growth).lower()


    # if inlcude_holiday params is 'default' or 'true' then use the default prophet holidays for the market
    if str(params.include_holiday).lower() in ['default','true']:

        
        random.seed(1234)

        # initialize the model, setting the growth and pt_kwargs hyper parameters
        model = Prophet(growth = growth,  **pt_kwargs) 

        # add default prophet country holidays for the market in params.market
        model.add_country_holidays(country_name = params.market)

    # if inlcude_holiday params is 'custom' or 'customized' the custom holidays are set, those are a sub dict taken from the customized_holidays[params.market] dict 
    elif str(params.include_holiday).lower() in ['custom','customized']:


        if customized_holidays is None:
            raise ValueError("customized_holidays must not be None if include_holiday is 'custom' or 'customized'")
        elif not isinstance(customized_holidays, dict):
            raise TypeError("customized_holidays must be a dict")
        elif params.market not in customized_holidays.keys():
            raise KeyError(f"customized_holidays must include holidays of params.market {params.market} in as a key of the customized_holidays dict if include_holiday is 'custom' or 'customized'")
        elif not isinstance(customized_holidays[params.market], pd.DataFrame):
            raise TypeError("customized_holidays element in key customized_holidays[params.market] must be a dataframe")
        
        random.seed(1234)

        # initialize the model, setting the growth, pt_kwargs hyper parameters, and the holidays in customized_holidays[params.market]
        model = Prophet(growth = growth, holidays = customized_holidays[params.market], **pt_kwargs) #customized_holidays[params.market]

    # else holidays are not included in the model
    else:


        random.seed(1234)

        # initialize the model, setting the growth and pt_kwargs hyper parameters, and do not set any holidays
        model = Prophet(growth = growth, **pt_kwargs)
    


    # if input mode param is additive or multiplicative set the input mode of the external regressor 
    if str(params.input_mode).lower() in ('additive', 'multiplicative'):


        # add the external 'X' (cost) regressor with the input_mode for that regressor inclussion
        model.add_regressor('X', mode = params.input_mode)
    


    # if covid_year_exclude param is 'true' or 'yes' it drops the 2020 data from the training data set
    if str(params.covid_year_exclude).lower() in ('true', 'yes'):

        
        # drop all dates of 2020 from the training data set 
        training_dataset.drop(training_dataset[ (training_dataset['ds'] >= '2020-01-01') & (training_dataset['ds']<='2020-12-31') ].index, axis = 0, inplace = True)



    # set a list of the hyper params that already have being set 
    used_params = ['changepoint_prior_scale', 'seasonality_prior_scale', 'holidays_prior_scale', 'seasonality_mode','growth','include_holiday','input_mode','covid_year_exclude']

    # loop over the params that are not being set yet. that is loop over all params and use a condition for the ones that are not in used_params list 
    for i in params.keys(): 
        
        # use a condition for the ones that are not in used_params list and its value is 'additive' or 'multiplicative', since these additional params are additional regressors with an specific input mode
        if i not in used_params and str(params[i]).lower() in ('additive', 'multiplicative'):


            # add the additional regressor i in the params[i] input mode 
            model.add_regressor(i, mode = params[i])

    random.seed(1234)

    # fit the model with the training data set 
    model.fit(training_dataset)
    

    # return the model that has already being set and fitted with the training data set 
    return model







class GenerateModelFile:
    """
        Generates a pickle file that contains the prophet Model and the training data. This allows to use the model in posterior sessions without the need of setting and training again the model
        
        Parameter:
        -------
                
        Returns:
        -------

    """
        

    def __init__(self):
        """
            Initialize the class method without any arguments neither any initial set up 
            
            Parameter:
            -------
                    
            Returns:
            -------

        """
        pass

    
    def _Pickling(self, model_n_data, bucket_name, path_name, base_path = './'):
        """
            This function saves the pickle file that will include the information of the model
            
            Parameter:
            -------
                model_n_data: prophet.Prophet model  
                    prophet.Prophet model encapsulated in python object where the model settings and the model training data is stored
                bucket_name: string 
                    string that identifies tha name of the bucket in GCS where the pkl file is going to be stored 
                path_name: string 
                    string that identifies tha name of the file to be stored
                base_path: string 
                    string that identifies tha path in the local device where the pkl file is going to be stored 
                
                    
            Returns:
            -------
                saves the info (model_n_data) in a pickle file in local device and GCS


        """

        # Creates and open the pickle file in the local device file path 
        with open(base_path + path_name.split('/')[-1], 'wb') as f:
            
            # write and saves the model_n_data (model settings and training data) inside the pickle file
            pickle.dump(model_n_data,f)
        
        '''
        # generate a conection with GCS
        fs = gcsfs.GCSFileSystem()
        
        # list all files and dir inside the bucket name 
        fs.ls(bucket_name)

        # create the path in GCS where the file is going to be stored. bucket_name + file name 
        path_name = bucket_name + '/' + path_name

        # Creates and open the pickle file in the GCS file path 
        with fs.open(path_name, 'wb') as handle:

            # write and saves the model_n_data (model settings and training data) inside the pickle file
            pickle.dump(model_n_data, handle)
        '''


                
    def Generate(self, cuts_df, df, customized_holidays, breakdown_ls, bucket_name, folder_name, base_path = './'):
        """
            This function generates and saves the pickle file that will include the information of the model
            
            Parameter:
            -------
                cuts_df: pd.DataFrame
                    A pd.DataFrame that has a row for each breakdown and has the model hyper parameter settings  
                breakdown_ls: string list 
                    list of strings of the column names that identifies the breakdowns 
                bucket_name: string 
                    string that identifies tha name of the bucket in GCS where the pkl file is going to be stored 
                folder_name: string 
                    string that identifies tha name of the folder inside the GCS bucket where the file is going to be stored
                base_path: string 
                    string that identifies tha path in the local device where the pkl file is going to be stored                 
                    
            Returns:
            -------
                saves the model info in pickle files for each breakdown in local device and GCS

        """
        
        # loop over each breakdown, filtering the training dataset for that breakdown, creates the prophet model and saving it on a pickle file.
        for key, values in tqdm(list(cuts_df.iterrows()), total=cuts_df.shape[0]): 

            # create a list of the breakdown identifier  
            breakdown =[f"'{col}' = '{values[col]}'" for col in breakdown_ls]
                
            # print the breakdown identifier to know in which breakdown the search is 
            print(f"Breakdown: {' & '.join(breakdown)}")
            
            # create a query filtering conditions for every of the breakdown columns that need to be filtered for create the breakdown data set 
            query = ' & '.join([f"(df['{col}'] == '{values[col]}')" for col in breakdown_ls])

            # filter the dataset with the query to create the breakdown data set 
            model_df = eval(f"df[{query}]").reset_index(drop=True)

            # create the training data set, a date filtered model_df with data until the cutoff point  
            training_dataset = model_df[(model_df['ds'] < values.cutoff_month)]
            
            # create the prophet model. it uses the hyper parameters set on the values (cuts_df one row values). it also uses the training_dataset
            model = CreateModel(training_dataset, values, customized_holidays = customized_holidays)

            # create the pickle_file_name for the breakdown, the breakdown identification as the filename 
            pickle_file_name = ' '.join([values[i] for i in breakdown_ls])

            # create the path_name to save the file, folder_name + pickle_file_name
            path_name = folder_name + '/' + pickle_file_name + '.pkl'

            # saves the info of the model (model, training_dataset, model_df) in a picke file 
            self._Pickling(model_n_data = (model, training_dataset, model_df), bucket_name = bucket_name, path_name = path_name, base_path = base_path)
            





def MakeFuture(model, model_df, params, forecasting_periods, future_input_df: pd.DataFrame = None):
    """
        Function used to make fbprophet future dataframe that is going to be used as input dataframe for the fbprophet forecasting.
        
        Parameter:
        -------
            model: Prophet.prophet model
                the prophet model that is going to be used in the forecasting 
            model_df: pd.DataFrame
                a pd.DataFrame that is passed to the model to train and fit it. 
            params: pd.DataFrame
                a row of a pd.DataFrame that has the model hyper parameter settings
            forecasting_periods: integer
                an integer number that sets the number of forecasting periods that the model is going to forecast. is the number of periods (rows) that the future df will have 
            future_input_df: pd.DataFrame
                a pd.DataFrame that has external regressors input values for the forecasting period. 

                
        Returns:
        -------
            future: pd.DataFrame
                a pd.DataFrame that has a date (ds) column with n periods to forecast, and the external regressor columns with non Nan values in these columns in the n periods to forecast.
    """
    
    # get the last date fitted on the model, this is to make sure the future_input_df does not have any values before the last modeled date 
    last_model_date = model.make_future_dataframe(periods = 0).ds.max()


    # create the future dataframe with n periods to forecast. It uses the make_future_dataframe() function from the fbprophet model. 
    future = model.make_future_dataframe(periods = int(forecasting_periods))

    # create a dataframes with only dates that are going to be forecasted
    forecast_dates = future.loc[future['ds']>last_model_date]

    # ensure the model df does not have any future values, value of a date after last_model_date
    model_df = model_df.loc[model_df['ds']<=last_model_date]

    # if model_df there is a column 'y', drop it because it could not be stored in the future df 
    if 'y' in model_df.columns:

        # drop the 'y' column 
        model_df = model_df.drop(columns='y')
    
        
     # if future_input_df is a pd.DataFrame the merge with the future dataframe 
    if isinstance(future_input_df,pd.DataFrame):

        # ensure the future_input_df has only dates after the last_model_date
        future_input_df = future_input_df.loc[future_input_df['ds']>last_model_date]

        # perform an outter (full) join on the forecast_dates and future_input_df by 'ds' this to ensure all forecasting dates are in future_input_df 
        future_dates = forecast_dates.merge(future_input_df['ds'], on='ds', how='outer', indicator=True)

        # if there are some forecasting dates not present in future_input_df raise an error 
        if len(future_dates.loc[future_dates['_merge']=='left_only'])>0:
            
            # raise a value error showing that future_input_df should have all forecasting dates
            ValueError('future_input_df should have all dates in forecasting period')

        # concat the model_df and the future_input_df, this ensures that common columns are merged once in future dataframe and that all values (training values and future values) are present in future dataframe
        input_df = pd.concat([model_df, future_input_df])

        # merge the future dataframe with input_df by ds (date)
        future = future.merge(input_df, how = 'left', on = 'ds')

    # if there is no future_input_df just use the model_df to add external regressor columns to future dataframe 
    else: 

        # merge the future dataframe with model_df by ds (date)
        future = future.merge(model_df, how = 'left', on = 'ds')

    # if model 'cost_split_generic' or 'cost_split_competitor' hyper parameters are set as 'additive' or 'multiplicative' fill their Nan values with 0.5
    if str(params.cost_split_generic).lower() in ('additive', 'multiplicative') or str(params.cost_split_competitor).lower() in ('additive', 'multiplicative'):
        
        # fill Nan values of cost_split_generic with 0.5 
        future['cost_split_generic'] = future['cost_split_generic'].fillna(.5)

        # fill Nan values of cost_split_competitor with 0.5 
        future['cost_split_competitor'] = future['cost_split_competitor'].fillna(.5)
    
    # if model 'weekday_or_weekend' hyper parameter is set as 'additive' or 'multiplicative' ensure the future dates have the weekday_or_weekend dummy filled correctly 
    if str(params.weekday_or_weekend).lower() in ('additive', 'multiplicative'):

        # set the weekday_or_weekend dummy for all dates in future dataframe 
        future['weekday_or_weekend'] = future['ds'].apply(lambda x: 1 if x.dayofweek > 4 else 0)

    # if 'covid_year_exclude' is set as a model hyper parameter, ensure 2020 dates are excluded from future dataframe 
    if str(params.covid_year_exclude).lower() == 'true':

        # drop rows that ds (date) year is 2020
        future.drop(future[(future['ds'] >= '2020-01-01')&(future['ds']<='2020-12-31')].index, axis = 0, inplace = True)

    # fill all other Nan values with 0 
    future = future.fillna(0)

    # if growth hyper parameter is set as logistic create the 'floor' (max forecasting value) & 'cap' (min forecasting value) columns 
    if str(params.growth).lower() == 'logistic':

        # create the 'cap' (max forecasting value) column 
        future['cap'] = future.ROAS * future.X / future.AOV

        # if cap is 0 replace it with the max value of cap 
        future.loc[future.cap == 0, 'cap'] = future.cap.max()

        # create the 'floor' column, set min value to 0 to prevent negative values in the forecast
        future['floor'] = 0

    # if growth hyper parameter is not set as logistic drop the 'floor' & 'cap' columns, so the model ignores the need of boundaries in the forecasting
    else:

        # drop 'floor' & 'cap' columns if they are included in the future dataframe
        future.drop(columns=['cap','floor'], axis=1, inplace=True, errors='ignore')

    # returns the future dataframe
    return future



def remove_negs(ts):
    """
        this function replaces the negative forecasted of the prophet model values with 0 
        
        Parameter:
        -------
            ts: pd.DataFrame
                future data frame with the prohpet forecast, must include 'yhat','yhat_lower' and 'yhat_upper' columns
                
        Returns:
        -------
    """

    # Replaces all values lower than 0 with 0 
    ts['yhat'] = ts['yhat'].clip(lower=0)

    # Replaces all values lower than 0 with 0 
    ts['yhat_lower'] = ts['yhat_lower'].clip(lower=0)

    # Replaces all values lower than 0 with 0 
    ts['yhat_upper'] = ts['yhat_upper'].clip(lower=0)




def round_forecast(ts):
    """
        this function round up to 2 decimals the values of the forecast of the prophet model 
        
        Parameter:
        -------
            ts: pd.DataFrame
                future data frame with the prohpet forecast, must include 'yhat','yhat_lower' and 'yhat_upper' columns
                
        Returns:
        -------
    """

    # rounds all values to 2 decimals
    ts['yhat'] = ts['yhat'].round(2)

    # rounds all values to 2 decimals
    ts['yhat_lower'] = ts['yhat_lower'].round(2)
    
    # rounds all values to 2 decimals
    ts['yhat_upper'] = ts['yhat_upper'].round(2)



def forecast_model_predict(model, model_df, params, forecasting_periods, future_input_df):

    future = MakeFuture(model = model, model_df = model_df, params = params, forecasting_periods = forecasting_periods, future_input_df = future_input_df)
    
    random.seed(1234)

    forecast = model.predict(future)
        
    remove_negs(forecast)

    round_forecast(forecast)
    
    return forecast