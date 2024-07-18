from pyswarms.single.global_best import GlobalBestPSO
import warnings
from pyswarms.utils.search import RandomSearch
from pyswarms.utils.plotters import plot_cost_history
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import datetime
import time
import random
import os

from .Accuracy import MakeFuture

#function to remove any negative forecasted values.
def remove_negs(ts):
    ts['yhat'] = ts['yhat'].clip(lower=0)
    ts['yhat_lower'] = ts['yhat_lower'].clip(lower=0)
    ts['yhat_upper'] = ts['yhat_upper'].clip(lower=0)

#function to round up dicimal forecasted values.
def round_forecast(ts):
    ts['yhat'] = ts['yhat'].round(0)
    ts['yhat_lower'] = ts['yhat_lower'].round(0)
    ts['yhat_upper'] = ts['yhat_upper'].round(0)

def save_result(df, filename):
    to_save_df = df
    file_name = filename+'.csv'
    print(file_name)
    if os.path.isfile(file_name):
#         to_save_df.to_csv(file_name, mode='a', header=False) 
        print('File exists. Please delete it first.')
    else: 
        to_save_df.to_csv(file_name)
        print('File generated')


class SavantPSO:
    
    def _get_before_after_variable(self):
        ### Check if NY budget meets the requirement 
        # update the searching term based on the customized period
        self.cut_off_point = self.config_file['PROPHETSETTING']['cut_off_point']
        self.optimization_start_date = self.config_file['PROPHETSETTING']['optimization_start_date']
        pre_optimization_months = pd.date_range(start=self.cut_off_point, end=self.optimization_start_date, freq='ME').strftime('%b').tolist()

        try:
            variable_list_before = [ x for x in self.decisionVariableArray if any(month in x for month in pre_optimization_months) ]
            variable_list_latest = list( set(self.decisionVariableArray) - set(variable_list_before) )
            variable_list_latest = sorted(variable_list_latest)
        except error:
            variable_list_latest = self.decisionVariableArray
            
        return variable_list_before, variable_list_latest
                
    def _objective_funtion_wrap(self, x):
        """
        Function used to generate the maximized output based on given inputWeights across particles

        Parameters:
        -------
        x: 2D numpy.ndarray
            1D for the number of particles and 2D for the number of cuts

        Returns:
        -------
            numpy.ndarray for maximized output across particles
        """
        n_particles = x.shape[0]
        j = [self._objective_funtion(x[i]) for i in range(n_particles)]
        return np.array(j)
        
    def _forecast_fn(self, i, analytical_file):
        """
        Function used to generate the forecast output

        Parameters:
        -------
        i: dictionary
            One row in the cuts_df dataframe that contains Prophet Configuration
            
        Returns:
        -------
            Dataframe for future forecast
        """
        # print(i)
        breakdown_ls_dict = {}

        num_breakdowns = len(self.breakdown_ls)

        for j in range(1,num_breakdowns+1):
            breakdown_ls_dict[f"breakdown{j}"] = i[j]

        analytical_file_copy = analytical_file.copy()
        
        pickle_file_path = self.base_path + i.forecaster_name_list
        pickle_in = open(pickle_file_path, 'rb') 
        
        unpickled_model, training_dataset, model_df = pickle.load(pickle_in)
        
        spend_df_for_forecast = analytical_file_copy.copy()
        for j in range(1,num_breakdowns+1):
            spend_df_for_forecast = spend_df_for_forecast.query(f"{self.config_file['CLIENTINFO'][f'breakdown{j}']}==" + '"' + f"{breakdown_ls_dict[f'breakdown{j}']}" +'"')
            
        spend_df_for_forecast['ds'] = pd.to_datetime(spend_df_for_forecast.ds)
        
        future = MakeFuture(unpickled_model, training_dataset, i, i.periods, spend_df_for_forecast)
        
        forecast = unpickled_model.predict(future)
        remove_negs(forecast)
        round_forecast(forecast)
        
        for j in range(1,num_breakdowns+1):
            forecast[self.config_file['CLIENTINFO'][f'breakdown{j}']] = i[j]
            
        return forecast
    
    def _objective_funtion(self, inputWeights):
        """
        Function used to generate the maximized output based on given inputWeights

        Parameters:
        -------
        inputWeights: numpy.ndarray
            Simulated series of number in the range defined in the constraint file

        Returns:
        -------
            numpy.float64 for maximized output
        """
        variable_list_before, variable_list_latest = self._get_before_after_variable()
        
        daily_simulated_forecast, updated_weights = self._predict(inputWeights, variable_list_before, variable_list_latest)

        # check if any yhat equals nan
        error_idx = daily_simulated_forecast.index[daily_simulated_forecast['yhat'].isna()]
        
        if error_idx.shape[0] != 0:
            error_rows = daily_simulated_forecast.iloc[error_idx]

            print('Multiple NaN matched yhat found for:\n')
            print(error_rows[['market', 'account_type', 'funnel', 'ds', 'X', 'yhat']])
            return 'Error: Multiple NaN matched yhat found'
        else:
            
            # convert min to max optimization
            total_yhat = daily_simulated_forecast['yhat'].sum() * -1
            print('yhat:', total_yhat)

            analytical_spend_latest_period = daily_simulated_forecast[variable_list_latest].sum().sum()
            # print('analytical_spend_latest_period:', analytical_spend_latest_period)
            print('client_budget - analytical_spend_latest_period:', round(self.client_budget - analytical_spend_latest_period))

            global itr_idx
            itr_array.append(itr_idx)
            itr_idx += 1

            global log_df
            global log_file_name
            log_df['decisionVariableArray'] = self.decisionVariableArray
            log_df['Input Weight'] = updated_weights
            log_df['totay_yhat'] = total_yhat
            log_df['iter'] = itr_idx
            log_df['Date'] = datetime.datetime.now().strftime("%m.%d.%Y %H:%M:%S")
            log_df.to_csv(log_file_name, mode = 'a', header = False)

            return total_yhat
            
    def _predict(self, inputWeights, variable_list_before, variable_list_latest):

        analytical_file_copy = self.analytical_file.copy()
        # create a template for optimized spend
        analytical_file_temp = analytical_file_copy.copy()
    #     print(decisionVariableArray)
        for i in self.decisionVariableArray: 
            analytical_file_temp[i] = analytical_file_temp[i] * inputWeights[self.decisionVariableArray.str.match(i)][0]

        analytical_file_temp.loc[:,'X'] = analytical_file_temp[self.decisionVariableArray].sum(axis = 1)

        ## Check if spend is same as what UPS planned
        total_spend_next_period = analytical_file_temp.loc[:,'X'].sum()

        total_budget = self.client_budget + self.pre_forecast_period_budget
        # print("Total Budget:", total_budget, "client_budget:", self.client_budget, "pre_forecast_period_budget:", self.pre_forecast_period_budget)
        
        if round(total_budget,0) != round(total_spend_next_period,0):
            print("{0:-^80}".format('Optimized spend isn\'t the same as what UPS planned.'))
            inputWeights = inputWeights/(total_spend_next_period/total_budget)
            for i in self.decisionVariableArray: 
                analytical_file_temp[i] = analytical_file_copy[i] * inputWeights[self.decisionVariableArray.str.match(i)][0]

            analytical_file_temp.loc[:,'X'] = analytical_file_temp[self.decisionVariableArray].sum(axis = 1)
            total_spend_next_period = analytical_file_temp.loc[:,'X'].sum()
            print('Test 1: total X next year: ', round(total_spend_next_period, 0),\
                  'difference between planned and forecast X: ', round(total_budget - total_spend_next_period,4))
        else:
            pass

        for i in variable_list_latest: 
            analytical_file_temp[i] = analytical_file_copy[i] * inputWeights[self.decisionVariableArray.str.match(i)][0]
        analytical_spend_latest_period = analytical_file_temp[variable_list_latest].sum().sum()

        for i in variable_list_before: 
                analytical_file_temp[i] = analytical_file_copy[i] * inputWeights[self.decisionVariableArray.str.match(i)][0]

        spend_previous_period = analytical_file_temp[variable_list_before].sum().sum()

        if round(analytical_spend_latest_period,0) != round(self.client_budget,0):
            variable_str_latest = '|'.join(variable_list_latest)
            ratio1 = analytical_spend_latest_period/self.client_budget
            # print('Before adjust the weight of the latest period, the ratio between spend and budget for the latest period is:', ratio1)
            inputWeights[[i in variable_list_latest for i in self.decisionVariableArray]] = inputWeights[[i in variable_list_latest for i in self.decisionVariableArray]] / ratio1
            ratio2 = spend_previous_period/self.pre_forecast_period_budget
            variable_str_previous_period = '|'.join(variable_list_before)
            inputWeights[[i in variable_list_before for i in self.decisionVariableArray]] = inputWeights[[i in variable_list_before for i in self.decisionVariableArray]] / ratio2

        for i in self.decisionVariableArray: 
                analytical_file_temp[i] = analytical_file_copy[i] * inputWeights[self.decisionVariableArray.str.match(i)][0]

        analytical_file_temp.loc[:,'X'] = analytical_file_temp[self.decisionVariableArray].sum(axis = 1)

        # print("client budget latest:", self.client_budget)
        # print("after adjust the weight, the lastest period spend is:", analytical_file_temp[variable_list_latest].sum().sum())
        spend_previous_period = analytical_file_temp[variable_list_before].sum().sum()
        # print('pre_forecast_period_budget:', self.pre_forecast_period_budget)
        # print("after adjust the weight, the previous period period spend is:", spend_previous_period)

        
        daily_simulated_forecast = analytical_file_temp.copy()
    #     daily_simulated_forecast['yhat'] = np.nan

        forecasts_df = pd.DataFrame()

        for i in self.cuts_df.itertuples():
            forecast = self._forecast_fn(i, analytical_file_temp)
            forecasts_df = pd.concat([forecasts_df, forecast], ignore_index=True)            
        

        # put forecasted y to daily_simulated_forecast
        daily_simulated_forecast = daily_simulated_forecast.merge(forecasts_df[['market', 'account_type', 'funnel', 'ds', 'yhat']], how = 'left', on = ['market', 'account_type', 'funnel', 'ds'])
        
        
        return daily_simulated_forecast, inputWeights
    
    
    def _get_result_df(self, inputWeights):
        """
        Function used to generate the maximized output based on given inputWeights

        Parameters:
        -------
        inputWeights: numpy.ndarray
            Simulated series of number in the range defined in the constraint file

        Returns:
        -------
            numpy.float64 for maximized output
        """
        variable_list_before, variable_list_latest = self._get_before_after_variable()
        
        daily_simulated_forecast, updated_weights = self._predict(inputWeights, variable_list_before, variable_list_latest)

        # check if any yhat equals nan
        error_idx = daily_simulated_forecast.index[daily_simulated_forecast['yhat'].isna()]
        
        if error_idx.shape[0] != 0:
            error_rows = daily_simulated_forecast.iloc[error_idx]

            print('Multiple NaN matched yhat found for:\n')
            print(error_rows[['market', 'account_type', 'funnel', 'ds', 'X', 'yhat']])
            return 'Error: Multiple NaN matched yhat found'
        else:
            # convert min to max optimization
            total_yhat = daily_simulated_forecast['yhat'].sum()
            print('yhat:', total_yhat)

            analytical_spend_latest_period = daily_simulated_forecast[variable_list_latest].sum().sum()
            print('analytical_spend_latest_period:', analytical_spend_latest_period)
            print('client_budget - analytical_spend_latest_period:', round(self.client_budget - analytical_spend_latest_period))

            return daily_simulated_forecast


    def __init__(self, cuts_df, analytical_file, constraint_file, config_file):
        """
        Parameters:
        -------
        cuts_df:
            DataFrame that contains the model settings
            
        analytical_fiel:
            DataFrame that contains analytical spend reference
            
        constraint_file: pandas.core.frame.DataFrame
            Data table that contains upper bound and lower bound for the inputWeights to take
            
        config_file:
            Config file that saves the init information

        Returns:
        -------
            None
        """
        self.log_df = pd.DataFrame(columns = [['Iter', 'Input Weight', 'decisionVariableArray', 'total_yhat', 'Date']])
        self.cuts_df = cuts_df.copy()
        self.analytical_file = analytical_file.copy()
        # constrain file doesn't have duplicate
        self.decisionVariableArray = constraint_file['variable_splits']
        self.constraint_file = constraint_file.copy()
        self.constraint_file['init'] = self.constraint_file[['MinWeight', 'MaxWeight']].mean(axis = 1)

        self.config_file = config_file
        self.breakdown_ls = [config_file['CLIENTINFO'][i] for i in [f'breakdown{i}' for i in range(1,5)] if config_file['CLIENTINFO'][i] != None]
        self.pre_forecast_period_budget = float(config_file['CLIENTINFO']['pre_forecast_period_budget'])
        self.client_budget = float(config_file['CLIENTINFO']['client_budget'])

        self.output_bucket = config_file["FILEPATH"]["output_bucket"]
        self.folder_name = config_file["FILEPATH"]["folder_name"]
        self.base_path = config_file["FILEPATH"]["base_path"]
        
     ## Particle Swarm Optimization
    def run_pso(self, options, init_pos: np.ndarray = None, n_particles: int = 0, itNum: int = 1):
        """
        Function used to simulate a variety of inputWeights options and determine the best one that maximize the objective function
        
        Parameters:
        -------
        options: int
            Selected set of hyperparameters. 
        init_pos: array
            Initial position for PSO. Must be the same length as the number of splits.
            If not provided, default to the mean of Max and Min of constraint boundary.
            
        n_particles: int
            Number of particles in the Particle Swarm Optimization algorithm. Default to zero.
            If zero, then calculated based on the number of splits.

        itNum: int
            Number of iterations to run. Default to 1            

        Returns:
        -------
            None
        """
        # Create bounds
        max_bound = self.constraint_file['MaxWeight'].to_numpy()
        min_bound = self.constraint_file['MinWeight'].to_numpy()
        bounds = (min_bound, max_bound)
        
        if n_particles == 0:
            self.n_particles = int(10+2*math.sqrt(len(self.decisionVariableArray)))
        else:
            self.n_particles = n_particles
        
        if init_pos is None:
            init_pos = np.tile(self.constraint_file.init, [self.n_particles, 1])
        
        optimizer = GlobalBestPSO(n_particles = self.n_particles, dimensions = len(max_bound), bounds = bounds, options = options, init_pos = init_pos)
        
        global log_file_name
        log_file_name = self.base_path + 'Input Weight History ' + datetime.date.today().strftime("%m.%d.%Y") + '.csv'
        print(log_file_name)
        global log_df
        log_df = pd.DataFrame(columns = ['decisionVariableArray', 'Input Weight', 'totay_yhat', 'iter', 'Date'])
        log_df.to_csv(log_file_name, mode = 'w', header = True)

        global itr_array
        itr_array = []
        global itr_idx
        itr_idx = 0

        start = time.time()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            random.seed(12345)
            cost, pos = optimizer.optimize(objective_func = self._objective_funtion_wrap, iters = itNum )

        end = time.time()
        print('Compute duration:', end - start)

        pos_df = pd.DataFrame({'variable_splits': self.decisionVariableArray, 'pos': pos})
        conversion_quantity = [-c for c in optimizer.cost_history]
        iteration_history = pd.DataFrame({'iteration':list(range(1,len(conversion_quantity)+1)),'Conversion Quantity': conversion_quantity})
        
        return {'pos': pos_df, 'conversions': conversion_quantity, 'history': iteration_history}
    
    def ResultGenerator(self, inputWeights):
        """
        Function used to generate result data frame. Need position for one time running.
        --------
        Parameters:
        
        inputWeights: array
            The best position generated by runpso.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            daily_simulated_forecast = self._get_result_df(inputWeights)
        
        return daily_simulated_forecast
    
    def SaveResult(self, result):
        
        today = datetime.date.today().strftime("%m.%d.%Y")
        save_result(result['pos'], self.base_path + 'Pos Result ' + today)
        save_result(result['history'], self.base_path + 'Iteration History ' + today)
        plot_cost_history(result['conversions'])
        plt.savefig(self.base_path + today + '.jpg')
    
    def SaveResultToGCS(self, result):
        
        # pos_df, conversion_quantity, iteration_history = result 
        
        SaveToGCS(source_file=result['pos'],
              bucket_name=self.output_bucket, 
              write_to_file_name = self.base_path + 'PSO Optimized Weights.csv', 
              folder_name=self.folder_name)

        SaveToGCS(source_file=result['history'],
              bucket_name=self.output_bucket, 
              write_to_file_name = self.base_path + 'PSO Iteration History.csv', 
              folder_name=self.folder_name)

        plot=plot_cost_history(result['conversions'])
        fig = plot.get_figure()
        plot_name = self.base_path + "PSO Iteration History.jpg"
        SaveToGCS(fig, self.output_bucket,  plot_name, self.folder_name)
        
    def RandomSearch(self, n_particles, rs_iters, rs_n_selection_iters):
        # Create bounds
        max_bound = self.constraint_file['MaxWeight'].to_numpy()
        min_bound = self.constraint_file['MinWeight'].to_numpy()
        bounds = (min_bound, max_bound)
        
        if n_particles == 0:
            n_particles = int(10+2*math.sqrt(len(self.decisionVariableArray)))

        # Set-up choices for the parameters
        params_search_options = {
        'c1': (0,5),
        'c2': (0,5),
        'w': (0,5),
        'k': (5, 15),
        'p': 1
        }

        # Create a RandomSearch object
        # n_selection_iters is the number of iterations to run the searcher
        # iters is the number of iterations to run the optimizer
        g = RandomSearch(GlobalBestPSO, n_particles=n_particles,
                dimensions=len(max_bound), options=params_search_options, objective_func=_objective_funtion_extended, bounds = bounds, 
                iters=rs_iters, n_selection_iters=rs_n_selection_iters)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            best_score, best_options = g.search()
        return best_options, best_score
