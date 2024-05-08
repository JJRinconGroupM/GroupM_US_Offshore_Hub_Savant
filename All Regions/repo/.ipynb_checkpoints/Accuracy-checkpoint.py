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

from .BuildModel import CreateModel
import pandas as pd
import numpy as np
import datetime
from .FileSaving import *
import warnings
from tqdm import tqdm
import itertools
from pathos.multiprocessing import ProcessingPool as PPool
import pathos as pa
import re
from os import getpid

import random

#function to remove any negative forecasted values.
def remove_negs(ts):
    ts['yhat'] = ts['yhat'].clip(lower=0)
    ts['yhat_lower'] = ts['yhat_lower'].clip(lower=0)
    ts['yhat_upper'] = ts['yhat_upper'].clip(lower=0)

#function to round up dicimal forecasted values.
def round_forecast(ts):
    ts['yhat'] = ts['yhat'].round(2)
    ts['yhat_lower'] = ts['yhat_lower'].round(2)
    ts['yhat_upper'] = ts['yhat_upper'].round(2)

def mean_absolute_percentage_error(y_pred, y_true): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diff = np.abs(y_true - y_pred)
#     diff[np.isinf(y_pred)] = 0
    return round(np.where(y_true != 0, diff / y_true, 0).mean() * 100, 2)

# Calculate row percentage error
def row_pct_error(forecast_df, actual_df, row):
    forecast_df = forecast_df[['ds', 'yhat']].groupby(pd.Grouper(key="ds", freq="MS", sort = True)).sum().reset_index()
    actual_df = actual_df[['ds', 'y', 'Spend']].groupby(pd.Grouper(key="ds", freq="MS", sort = True)).sum().reset_index()
    
    joinedDf = pd.merge(forecast_df, actual_df, how = 'inner', on = 'ds')
    joinedDf['abs_pct_error'] = np.abs((joinedDf['y'] - joinedDf['yhat'])/joinedDf['y']) * 100
    joinedDf['abs_pct_error'] = joinedDf['abs_pct_error'].round(0)
    
    joinedDf['Campaign_Type'] = row.Campaign_Type
    joinedDf['Search_Tactic'] = row.Search_Tactic
    joinedDf['Ad_Type'] = row.Ad_Type

    return joinedDf

# return mape
def monthly_MAPE_calcuation(forecast_df, actual_df, start_day):
    forecast_df = forecast_df.copy()
    actual_df = actual_df.copy()
    start_month = datetime.datetime(start_day.year, start_day.month, 1) # update start in case it's not month start
     
    monthly_forecast = forecast_df[['ds', 'yhat']].groupby(pd.Grouper(key="ds", freq="MS", sort = True)).sum().reset_index()
    monthly_actual = actual_df[['ds', 'y']].groupby(pd.Grouper(key="ds", freq="MS", sort = True)).sum().reset_index()
    
    joined_df = pd.merge(monthly_actual, monthly_forecast, how = 'inner', on='ds')
    joined_df['Percentage Error'] = ((joined_df['y'] - joined_df['yhat'])/joined_df['y']).round(4) * 100
    cal_start_idx = joined_df[joined_df['ds'] == start_month].index.values[0]
    
    MAPE = mean_absolute_percentage_error(joined_df.loc[cal_start_idx:, 'yhat'], joined_df.loc[cal_start_idx:, 'y'])
    joined_df['MAPE'] = MAPE
    
    period_AE = (np.abs(joined_df.loc[cal_start_idx:, 'yhat'].sum() - joined_df.loc[cal_start_idx:, 'y'].sum())/joined_df.loc[cal_start_idx:, 'y'].sum()).round(4) * 100
#     print('yhat sum:', joined_df.loc[cal_start_idx:, 'yhat'].sum(),'y sum:',  joined_df.loc[cal_start_idx:, 'y'].sum())
    joined_df['Forecast Period MAE'] = period_AE
    
    if (joined_df.loc[cal_start_idx:, 'yhat'].sum() - joined_df.loc[cal_start_idx:, 'y'].sum()) >0 :
        print('It\'s overpredicted.')
        joined_df['Predict Status'] = 'OverPredicted'
    elif (joined_df.loc[cal_start_idx:, 'yhat'].sum() - joined_df.loc[cal_start_idx:, 'y'].sum()) == 0:
        print('It\'s identical as actual.')
        joined_df['Predict Status'] = 'Exact The Same. Check Needed.'
    else:
        print('It\'s underpredicted.')
        joined_df['Predict Status'] = 'UnderPredicted'
    
     
    print('Monthly Error:', MAPE, "Period MAE:", (period_AE))
    return MAPE, period_AE, joined_df


def MakeFuture(model, model_df, values, periods, future_input_df: pd.DataFrame = None):
    """
    Function used to make fbprophet future dataframe.
    """

    future = model.make_future_dataframe(periods = int(periods))
    try:
        
        if len(future_input_df.columns)>0:
            future_col_ls_default = ['X', 'generic_cost_split', 'competitor_cost_split', 'new_confirmed', 'ROAS', 'AOV']
            future_col_ls = future_col_ls_default.copy()
            for i in future_col_ls_default:
                if i not in future_input_df.columns:
                    future_col_ls.remove(i)
            
            for i in future_col_ls:
                future = future.merge(future_input_df[['ds']  + list([i])], how = 'left', on = 'ds')
                if i in ['generic_cost_split', 'competitor_cost_split']:
                    future[i] = future[i].fillna(.5)
                else:
                    future[i] = future[i].fillna(future[i].min())
    except:
        pass    

    
    if str(values.weekday_or_weekend).lower() in ('additive', 'multiplicative'):
        future['weekday_or_weekend'] = future['ds'].apply(lambda x: 1 if x.dayofweek > 4 else 0)

    if str(values.covid_year_dummy).lower() in ('additive', 'multiplicative'):
        future['covid_year_dummy'] = future['ds'].apply(lambda x: 1 if (x >= datetime.date(2020,1,1)) & (x <= datetime.date(2020,12,31)) else 0)

    if str(values.covid_year_exclude).lower() == 'true':
        future.drop(future[(future['ds'] >= '2020-01-01')&(future['ds']<='2020-12-31')].index, axis = 0, inplace = True)
        
    if str(values.anomaly_2021).lower() in ('additive', 'multiplicative'):
        future['anomaly_2021'] = future['ds'].apply(lambda x: 1 if (x >= datetime.date(2021,8,1))\
                                     & (x <= datetime.date(2021,10,31)) else 0)
        
    if str(values.new_confirmed).lower() in ('additive', 'multiplicative'):
        future = future.merge(model_df[['ds', 'new_confirmed']], how = 'left', on = 'ds')  

    if str(values.generic_cost_split).lower() in ('additive', 'multiplicative'):
        future = future.merge(model_df[['ds', 'generic_cost_split']], how = 'left', on = 'ds')
            
    if str(values.competitor_cost_split).lower() in ('additive', 'multiplicative'):
        future = future.merge(model_df[['ds', 'competitor_cost_split']], how = 'left', on = 'ds')  


    future['mar_2018'] = 0
    future.loc[(future['ds'] <= '2018-03-31') & (future['ds'] >= '2018-03-01'), 'Mar_2018'] = 1

    future['nov_dec_2018'] = 0
    future.loc[(future['ds'] <= '2018-12-31') & (future['ds'] >= '2018-11-01'), 'Nov_Dec_2018'] = 1

    future['apr_2019'] = 0
    future.loc[(future['ds'] <= '2019-04-30') & (future['ds'] >= '2019-04-01'), 'Apr_2019'] = 1

    future['may_2019'] = 0
    future.loc[(future['ds'] <= '2019-05-31') & (future['ds'] >= '2019-05-01'), 'May_2019'] = 1

    future['jul_dec_2019'] = 0
    future.loc[(future['ds'] <= '2019-12-31') & (future['ds'] >= '2019-07-01'), 'Jul_Dec_2019'] = 1
    
    future = pd.merge(future, model_df, how = 'left', on = ['ds']).fillna(0) 
    ls = future.columns

    for i in ls:
        if i.endswith(('_x', '_y')):
            new_col_name = re.search(r"(.+?)(_x|_y)", i).group(1)
            future[new_col_name] = future.apply(lambda x: max(x[new_col_name+'_x'], x[new_col_name+'_y']), axis=1)
    future = future.fillna(0)
    
    if str(values.growth).lower() == 'logistic':
        future['cap'] = future.ROAS * future.X / future.AOV
        future.loc[future.cap == 0, 'cap'] = future.cap.max()
        future['floor'] = 0
    else:
        future.drop(columns=['cap','floor'], axis=1, inplace=True, errors='ignore')

#     print(values)
#     print("a0:", future.cap.min())

#     print("a:", future.cap.min())
    return future

def PlotAccuracyCharts(accuracy_df, cuts_df, query_item_list):
    for m, a, f in cuts_df[query_item_list].values:
        df_for_plt =  accuracy_df[(accuracy_df['market']==m) & (accuracy_df['account_type']==a) & (accuracy_df['funnel']==f)].drop([ 'MAPE', 'Forecast Period MAE', 'Predict Status'], axis = 1)
        df_for_plt.set_index('Month').plot(y=['Actual', 'Forecast','Percentage Error'], secondary_y=['Percentage Error'], title=' '.join([m, a, f]))

class AccuracyCalculator:
    """
    This is to calculate the model accuracy.
    Accuracy scores include monthly MAPE and MAPE for entire period.
    
    Attributes:
    -------
        self.values: model settings stored in pandas series.
    
    Returns:
    -------
        None
    """
    def __init__(self) -> None:
        pass
    
    def _predict(self, future):
        
        forecast = self.model.predict(future)
            
        remove_negs(forecast)

        round_forecast(forecast)
        
        return forecast
        
    def CalculateAccuracy(self, cuts_df, df, breakdown_ls):
        """
        This is to calculate the model accuracy.
        Accuracy scores include monthly MAPE and MAPE for entire period.
    
        Parameters:
        -------
            Cuts_df: pd.DataFrame
                fbprophet model settings. Each row is a model.
            Breakdown_ls: list
                list of breakdown which will be modelled.
    
        Attributes:
        -------
            self.values: model settings stored in pandas series.
    
        Returns:
        -------
            accuracy_df: pd.DataFrame
                Accuracy scores
        """
        self.cuts_df = cuts_df
        self.df = df
        self.breakdown_ls = breakdown_ls
        
        self.columns = self.breakdown_ls+['Month', 'Actual', 'Forecast', 'Percentage Error', 'MAPE', 'Forecast Period MAE'] # MAPE is monthly level
        self.monthly_acc_df_temp = pd.DataFrame(columns=self.columns)
        
        for key, values in cuts_df.iterrows():
            self.values = values
            
            self.model_df = df.copy()
            for j in range(len(breakdown_ls)):
                self.model_df = self.model_df.query(f"{breakdown_ls[j]}=='{values[breakdown_ls[j]]}'").reset_index(drop=True)
            
            new_cut_off_point = self.model_df['ds'].max() - datetime.timedelta(days = values.periods_for_acc_cal)
            
            try:
                training_dataset = self.model_df[(self.model_df['ds'] < new_cut_off_point) & (self.model_df['ds'] >= values.starting_date)]
            except:
                training_dataset = self.model_df[(self.model_df['ds'] < new_cut_off_point)]
                
            self.model  = CreateModel(training_dataset, values)

            future = MakeFuture(self.model, self.model_df, self.values, self.values.periods_for_acc_cal)
            forecast = self._predict(future)
            
            MAPE, period_AE, joined_df = monthly_MAPE_calcuation(forecast, self.model_df, new_cut_off_point)         
            
            start_month = datetime.datetime(new_cut_off_point.year, new_cut_off_point.month, 1) # update start in case it's not month start
 
            monthly_acc_df = joined_df[joined_df['ds'] >= start_month].copy()
            monthly_acc_df.rename(columns = {'ds': 'Month', 'y': 'Actual', 'yhat': 'Forecast'}, inplace = True)
            for j in range(len(breakdown_ls)):
                monthly_acc_df.insert(0, breakdown_ls[j], values[breakdown_ls[j]])
                
            self.monthly_acc_df_temp = pd.concat([self.monthly_acc_df_temp, monthly_acc_df], ignore_index=True)
            
        return self.monthly_acc_df_temp        
        
    def SaveFile(self, bucket_name = None, folder_name = None, write_to_file_name = "Model Scores.csv"):
        try:
            self.monthly_acc_df_temp.to_csv(write_to_file_name, mode='w', index=False)
        except:
            print("Save to local is skipped. An error occurred.")
            pass
        try:
            SaveToGCS(self.monthly_acc_df_temp, bucket_name, write_to_file_name, folder_name)
        except Exception:
            print("Save to bucket is skipped. An error occurred.")
            pass

class ProphetParameterSearch:
    """
    ProphetParameterSearch class is used to search the best parameter setting for Prophet model.
    User can export model score for futher investigation
    or to run best_option to return the models with highest scores.
    """
    def __init__(self, param_grid = None, linear_flag = True, power_flag = True, logistic_flag=True, base_path='') -> None:
        """
        Initialize ProphetParameterSearch object.
        Trend growth flag is turned on by default.
        """
        self.param_grid = param_grid 
        self.growth_ls = list(['linear', 'logistic', 'power'])
        idx = 0 
        for i in [linear_flag, logistic_flag, power_flag]:
            if not i:
                self.growth_ls.pop(idx)
            else:
                idx += 1
        self.pow_ls = ['na']
        
        timestr = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        self.record_df_file_name = base_path + 'record_df ' + timestr + '.csv'
        self.best_record_df_file_name = base_path + 'record_df ' + timestr + ' best result.csv'
        
        self.record_df = pd.DataFrame(columns=['market', 'account_type', 'funnel', 'growth', \
                                               'changepoint_prior_scale', 'seasonality_prior_scale', 'holidays_prior_scale',
                                               'seasonality_mode', 'input_mode', 'weekday_or_weekend', \
                                               'include_holiday', \
                                               'covid_year_exclude', 'covid_year_dummy', 'anomaly_2021', \
                                               'generic_cost_split', 'competitor_cost_split', 'new_confirmed',\
                                               'January_Feb_2023',\
                                               'power','growth',\
                                               'MAPE', 'period_AE'])

 

        self.best_setting_df = self.record_df.copy()
        
    def _generate_params(self, base_params=None):
        """Generates parameter space for grid search.

        Parameters
        ==========
        base_params: dict
            Additional params to be passed into parameter space. 
            Default as None.

        holidays: bool
            Include holiday hyper-params.

        Returns
        =======
        all_params: list
            List of dictionaries including combinations of param grid space.
        """
        if self.param_grid is None:
            self.param_grid = {
                'changepoint_prior_scale': [0.05, 0.001, 0.1, 0.5], #  
                'seasonality_prior_scale': [10.0, 0.01, 0.1, 1.0], #
                'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0], #
                'seasonality_mode': ['additive','multiplicative'], #
                'input_mode': ['additive','multiplicative'], #], #
                'weekday_or_weekend':['na', 'additive', 'multiplicative'],  #
                'include_holiday': ['na', 'default', 'customized'],#
                'covid_year_exclude':['False', 'True'],
                'covid_year_dummy':['na', 'additive'],
                'anomaly_2021':['na', 'additive'],
                'generic_cost_split':['na', 'additive'],#
                'competitor_cost_split':['na', 'additive'], #
                'new_confirmed': ['na', 'additive'],#
                'Nov_Dec_2018':['na','additive'], #
                'Jul_Dec_2019': ['na','additive']
            }
        
        self.param_grid['power'] = self.pow_ls
        self.param_grid['growth'] = self.growth_ls
        
        all_params = [dict(zip(self.param_grid.keys(), v)) for v in itertools.product(*self.param_grid.values())]
        if base_params is not None:
            for i in range(len(all_params)):
                all_params[i].update(base_params)

        return all_params
    
    def _fit_n_evaluate(self, params):
        training_dataset = params['training_dataset'].copy()
        model_df = params['model_df'].copy()
        #only client related info is taken from row
        row = params['values']
        
        basic_info_dict = {'market': [row.market], 'account_type': [row.account_type], 'funnel': [row.funnel], 'periods_for_acc_cal': [row.periods_for_acc_cal]}

        search_gird = {k: params[k] for k in params.keys() if k not in ['training_dataset', 'model_df', 'values'] }
        basic_info_dict.update(search_gird)
        
        d = pd.DataFrame(basic_info_dict)

        model = CreateModel(training_dataset, d.iloc[0,:])      
        
        # make future df
        future = MakeFuture(model, model_df, d.iloc[0,:], periods = row.periods_for_acc_cal)
        random.seed(1234)
        forecast = model.predict(future)
#         print(forecast.head())
        round_forecast(forecast)
        remove_negs(forecast)
        
        if type(params['power']) == float:
            print('Power:', round(params['power'], 4))

        MAPE, period_AE, joined_df = monthly_MAPE_calcuation(forecast, model_df, datetime.datetime(training_dataset.ds.max().year, training_dataset.ds.max().month, 1))

        d['MAPE'] = MAPE
        d['period_AE'] = period_AE
        
        try:
            pd.read_csv(self.record_df_file_name)
            d.to_csv(self.record_df_file_name, mode='a', header=False)
        except:
            try:
                d.to_csv(self.record_df_file_name)
            except Exception:
                raise
        
        return d

    def add_power_params(self, start = 0.1**2, end = 1.1, num = 5, endpoint = False):
        self.start = start
        self.end = end
        self.num = num
        self.endpoint = endpoint # linspace won't include the endpoint in the result    
        self.pow_ls = list(np.linspace(self.start, self.end, self.num, self.endpoint))
        
    def search(self, cuts_df, df, param_search_dict = None):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for row in tqdm(list(cuts_df.itertuples()), total=cuts_df.shape[0]): 

                model_df = df[(df['market'] == row.market) & (df['account_type'] == row.account_type) & (df['funnel'] == row.funnel)].reset_index(drop = True)
                model_df = model_df.reset_index(drop = True)
                
                new_cut_off_point = model_df['ds'].max() - datetime.timedelta(days = row.periods_for_acc_cal)
                
                training_dataset = model_df[model_df['ds'] < new_cut_off_point]

                base_params = {
                    'training_dataset': training_dataset.copy(),
                    'model_df': model_df.copy(),
                    'values': row
                }
                
                param_grid = self._generate_params(base_params)
#                 for i in param_grid:
#                     self.record_df.append(self._fit_n_evaluate(i))
                columns = [i for i in param_grid[0].keys() if i not in ['training_dataset', 'model_df', 'values']]
                
                if self.record_df.shape[0] == 0:
                    self.record_df = pd.DataFrame(columns=['market', 'account_type', 'funnel', 'periods_for_acc_cal'] + columns + ['MAPE', 'period_AE'])
                    # save the best model setting and accuracy
                    self.best_setting_df = self.record_df.copy()
                
                random.seed(1234)
                with PPool(pa.helpers.cpu_count()) as p:
                    record_df_ls = []
                    record_df_ls = p.imap(self._fit_n_evaluate, param_grid)
                    
                print(record_df_ls)
                try:
                    self.record_df = pd.concat(record_df_ls)     
                except Exception:
                    print('parallel result concat error.')
                    raise
                    
                min_score = self.record_df[(self.record_df['market'] == row.market)&(self.record_df['account_type'] == row.account_type)&(self.record_df['funnel'] == row.funnel)]['MAPE'].min()
                    
                self.best_setting_df = pd.concat([self.best_setting_df, self.record_df[(self.record_df['market'] == row.market)&(self.record_df['account_type'] == row.account_type)&\
                                            (self.record_df['funnel'] == row.funnel)\
                                            &(self.record_df['MAPE'] == min_score)]])
                try:
                    pd.read_csv(self.best_record_df_file_name)
                    self.best_setting_df.to_csv(self.best_record_df_file_name, mode='a', header=False)
                except:
                    try:
                        self.best_setting_df.to_csv(self.best_record_df_file_name)
                    except Exception:
                        raise
                        
        return self.record_df
