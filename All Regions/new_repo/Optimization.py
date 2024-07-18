
import pandas as pd # for data processing 
import numpy as np # for numeric calculations and array structures
import datetime # for date variables 
import warnings # to get or supress warning messages 
import math # for mathematic and numeric opperations
import matplotlib.pyplot as plt # for making plots 
import pickle # to create and load binary files that store python objects inside the files
import random # random selection for selecting a seed in random searches 
import os # for local file system operations 
import time # for time variables 

from pyswarms.single.global_best import GlobalBestPSO # for running a PSO optimization for the Global best optim
from pyswarms.utils.plotters import plot_cost_history # for plotting the PSO optimization cost history 
from pyswarms.utils.search import RandomSearch # for running a random search in PSO parameters 

from .BuildModel import forecast_model_predict # for custom prophet model forecasting 


class SavantPSO:
    """
    This class performs the SAVANT optimization process, which is that using the prophet forecasting model to optimize cost (x) budget allocation (PSO weights) to maximize the conversion quantity (y) forecast for the optimization period
    The optimization process run_pso() goal is to optimize the conversion quantity varying the PSO weights.
    The result process ResultGenerator() uses a inputweight to generate the forecast with the corresponding budget allocation. 
    
    Attributes:
    -------
        
    
    Returns:
    -------
        
    """


    def __init__(self, cuts_df, analytical_file, constraint_file, config_file):
        """
        Initialize the SavantPSO class. it stores in class self attribute some important objects needed on other functions 

        Parameters:
        -------
        cuts_df: pd.DataFrame
            DataFrame that contains the model settings for each brakdown 
            
        analytical_file:
            DataFrame that contains analytical reference spend for the forecasting period
            
        constraint_file: pd.DataFrame
            Data table that contains upper bound and lower bound for the inputWeights to take
            
        config_file: dict
            Config file dictionary that saves the initial information

        Returns:
        -------
            None
        """

        # set a current date variable to add to the filenames that are going to be saved during the pso optimization
        self.today = datetime.date.today().strftime("%m.%d.%Y")

        # create an empty data frame where is going to be stored the log history of each iteration.
        self.log_df = pd.DataFrame(columns = [['Iter', 'Input Weight', 'splits_list', 'total_yhat', 'Date']])

        # store in self class attribute the cuts_df
        self.cuts_df = cuts_df.copy()

        # store in self class attribute the analytical_file
        self.analytical_file = analytical_file.copy()

        # store in self class attribute the splits_list which is the variable_splits column in the constrint file  
        self.splits_list = constraint_file['variable_splits']

        # store in self class attribute the constraint_file
        self.constraint_file = constraint_file.copy()

        # add the 'init' column to the constraint_file, input weight to init the pso opimization. mean between the MinWeight and MaxWeight
        self.constraint_file['init'] = self.constraint_file[['MinWeight', 'MaxWeight']].mean(axis = 1)

        # create a numpy array of the MaxWeight column in constraint_file as the max_bound
        self.max_bound = self.constraint_file['MaxWeight'].to_numpy()

        # create a numpy array of the MinWeight column in constraint_file as the min_bound
        self.min_bound = self.constraint_file['MinWeight'].to_numpy()

        # Create a bounds tuple, a combination of the two min_bound and max_bound numpy arrays. it is used as an argument in GlobalBestPSO() function 
        self.bounds = (self.min_bound, self.max_bound)

        # store in self class attribute the config_file
        self.config_file = config_file

        # store in self class attribute the breakdown columns list 
        self.breakdown_ls = [config_file['CLIENTINFO'][i] for i in sorted([key for key in config_file['CLIENTINFO'] if key.startswith('breakdown')]) if config_file['CLIENTINFO'][i] != None]

        # store in self class attribute a variable the pre_optimization_period_budget. 
        self.pre_optimization_period_budget = float(config_file['CLIENTINFO']['pre_optimization_period_budget'])

        # store in self class attribute a variable the client_budget 
        self.client_budget = float(config_file['CLIENTINFO']['client_budget'])

        # store in self class attribute a variable for the output_bucket name str 
        self.output_bucket = config_file["FILEPATH"]["output_bucket"]

        # store in self class attribute a variable for the folder_name str
        self.folder_name = config_file["FILEPATH"]["folder_name"]

        # store in self class attribute a variable for the base_path str  
        self.base_path = config_file["FILEPATH"]["base_path"]

    


    def _get_pre_n_optimization_splits_list(self):
        """
        Function used to generate the list of splits (breakdown - month - year) that identfies splits with year-month of the optimization period, an the ones of the pre optimization period.
        pre optimization period months splits are used in the forecast but are not used in the pso optimization 

        Parameters:
        -------

        Returns:
        -------
            pre_optimization_splits_list: list
                splits (breakdown - year - month) that have a year-month previous to the optimization period start date 
            optimization_splits_list: list
                splits (breakdown - year - month) that have a year-month after the optimization period start date 
        """
        
        # get the cutoff point, the first forecasting period 
        cut_off_point = self.config_file['PROPHETSETTING']['cut_off_point']

        # get the optimization start date, the first day of the optimization 
        optimization_start_date = self.config_file['PROPHETSETTING']['optimization_start_date']

        # get the pre optimization months (Mon_Year). months in forecasting period that are not in the optimization period. 
        pre_optimization_months = pd.date_range(start=cut_off_point, end=optimization_start_date, freq='MS').strftime('%b_%Y').tolist()

        # try to separate the pre optimization months from the variable splits array, if not, all split variables are set as optimization period. pre optimization perio months 
        try:
            # set the pre_optimization_splits_list. a list of splits_list elements that has the pre_optimization_months 
            pre_optimization_splits_list = [ x for x in self.splits_list if any(month in x for month in pre_optimization_months) ]

            # set the optimization_splits_list list. a list of splits_list that are not in pre_optimization_splits_list
            optimization_splits_list = list( set(self.splits_list) - set(pre_optimization_splits_list) )

            # sort the optimization_splits_list
            optimization_splits_list = sorted(optimization_splits_list)

        # if the splits_list could not be splitted between pre optimization and optimization list, use all splits_list as optimization list, and print an error
        except Exception as e:

            # set the optimization_splits_list list as the same splits_list list 
            optimization_splits_list = self.splits_list

            # set the pre_optimization_list as an empty list
            pre_optimization_splits_list = [] 

            # print the error for information
            print("Error occurred:", str(e))
            
        # return the pre_optimization_list and optimization_splits_list
        return pre_optimization_splits_list, optimization_splits_list





    def pso_predict(self, inputWeights, pre_optimization_splits_list, optimization_splits_list):
        """
        Function used to generate the forecast output based on the weighted analytical df (optimization process for pso weights)

        Parameters:
        -------
        inputWeights: np.array
            a numpy array that has an input weight for each splits_list
        pre_optimization_splits_list: list
            splits (breakdown - year - month) that have a month_year previous to the optimization period start date 
        optimization_splits_list: list
            splits (breakdown - year - month) that have a month_year after the optimization period start date
            
        Returns:
        -------
            daily_simulated_forecast: pd.DataFrame
                a data frame with the new cost (X) changed after the inputWeights transformation and the forecasted conversion quantity (y) based on the new cost
            inputWeights: np.array
                a numpy array that has an input weight for each splits_list
            
        """

        # create a analytical
        analytical_file_copy = self.analytical_file.copy()

        
        # create a temporal file for the analytical_file used for optimized spend
        analytical_file_temp = self.analytical_file.copy()


        # iterate over each split, changing its cost based on the inputWeight of each split 
        for i in self.splits_list: 

            # change split cost value based on its input weight
            analytical_file_temp[i] = analytical_file_temp[i] * inputWeights[self.splits_list.str.match(i)][0]

        # horizontally sum new split cost in the total cost column (X) for every split in self.splits_list
        analytical_file_temp.loc[:,'X'] = analytical_file_temp[self.splits_list].sum(axis = 1)


        ## Check if the total spend (pre optimization and optimization spend) is the same as the total budget spend (pre_optimization_period_budget and client_budget, optimization budget)

        # get the total_spend_forecast_period, sum of the optimized (changed) analytical file cost
        total_spend_forecast_period = analytical_file_temp.loc[:,'X'].sum()

        # get the total budget, pre_optimization_period_budget + client_budget 
        total_budget = self.pre_optimization_period_budget + self.client_budget 

        # if total_budget is different from total_spend_forecast_period print that it is different and adjust the inputWeights so that they get the same values.
        if round(total_budget,0) != round(total_spend_forecast_period,0):

            # print the warning that total_budget isn't the same as the optimized total_spend_forecast_period
            #print("{0:-^80}".format('Total forecast period spend (X) isn\'t the same as the total budget (pre_optimization_period_budget + client_budget).'))

            # print the 1.a test difference in cost, weighted total_spend_forecast_period vs total_budget
            #print('Test 1.a: total spend (X) for forecast period: ', round(total_spend_forecast_period, 0),\
            #      'difference between total budget and total forecast period spend (X): ', round(total_budget - total_spend_forecast_period,4))

            # get the ratio betweeen the total spend in the forecast period and the total budget (total_spend_forecast_period/total_budget)
            total_spend_budget_ratio = total_spend_forecast_period/total_budget

            # change the input weights to match the total_budget with the total_spend_forecast_period. inputWeights/total_spend_budget_ratio
            inputWeights = inputWeights/total_spend_budget_ratio

            # iterate over each split, changing its cost based on the adjusted inputWeights of each split 
            for i in self.splits_list: 

                # change split cost value based on the changed re weighted input weight
                analytical_file_temp[i] = analytical_file_copy[i] * inputWeights[self.splits_list.str.match(i)][0]

            # horizontally sum new split cost in the total cost column (X) for every split in self.splits_list
            analytical_file_temp.loc[:,'X'] = analytical_file_temp[self.splits_list].sum(axis = 1)

            # get the re calculated total_spend_forecast_period, sum of the optimized (changed) analytical file cost
            total_spend_forecast_period = analytical_file_temp.loc[:,'X'].sum()

            # print the 1.b test difference in cost, adjusting weights total_spend_forecast_period vs total_budget. this should make the the difference be 0. 
            #print('Test 1.b (adjusting weights): total spend (X) for forecast period: ', round(total_spend_forecast_period, 0),\
            #      'difference between total budget and total forecast period spend (X): ', round(total_budget - total_spend_forecast_period,4))

        # if total_budget is the same from total_spend_forecast_period pass to the second step client budget vs optimization spend 
        else:
            pass


        ## Check if the optimization spend is the same as the client_budget

        # iterate over each split in the optimization_splits_list, changing its cost based on the inputWeight of each split 
        for i in optimization_splits_list: 

            # change optimization split cost value based on their respective input weight
            analytical_file_temp[i] = analytical_file_copy[i] * inputWeights[self.splits_list.str.match(i)][0]

        # horizontally sum the new optimization_splits cost for every split in optimization_splits_list
        spend_optimization_period = analytical_file_temp[optimization_splits_list].sum().sum()

        # iterate over each split in the pre_optimization_splits_list, changing its cost based on the inputWeight of each split 
        for i in pre_optimization_splits_list: 
                
                # change the preo optimization split cost value based on their respective input weight
                analytical_file_temp[i] = analytical_file_copy[i] * inputWeights[self.splits_list.str.match(i)][0]

        # horizontally sum the new pre_optimization_splits cost for every split in pre_optimization_splits_list
        spend_pre_optimization_period = analytical_file_temp[pre_optimization_splits_list].sum().sum()


        # if client_budget is different from spend_optimization_period print that it is different and adjust the inputWeights so that they get the same values.
        if round(spend_optimization_period,0) != round(self.client_budget,0):

            # print the warning that client_budget isn't the same as the optimized spend_optimization_period
            print("{0:-^80}".format(' Optimization period spend (X) isn\'t the same as the client budget. '))

            # print the 2.a test difference in cost, weighted spend_optimization_period vs client_budget
            print('Test 1: Total spend (X) for optimization period: ', round(spend_optimization_period, 0),\
                  'difference between client budget and optimization period spend (X): ', round(self.client_budget - spend_optimization_period,4))
            
            # get the ratio betweeen the total spend in the pre optimization period and the pre optimization period budget. spend_pre_optimization_period/self.pre_optimization_period_budget
            pre_optimization_period_ratio = spend_pre_optimization_period/self.pre_optimization_period_budget
            
            # get the ratio betweeen the total spend in the optimization period and the client budget. spend_optimization_period/self.client_budget
            optimization_period_ratio = spend_optimization_period/self.client_budget

            # change the optimization input weights to match the client_budget with the spend_optimization_period. inputWeights[optimization_splits_list]/optimization_period_ratio
            inputWeights[[i in optimization_splits_list for i in self.splits_list]] = inputWeights[[i in optimization_splits_list for i in self.splits_list]] / optimization_period_ratio

            # change the pre optimization input weights to match the pre_optimization_period_budget with the spend_pre_optimization_period. inputWeights[pre_optimization_splits_list]/pre_optimization_period_ratio
            inputWeights[[i in pre_optimization_splits_list for i in self.splits_list]] = inputWeights[[i in pre_optimization_splits_list for i in self.splits_list]] / pre_optimization_period_ratio

        # iterate over each split, changing its cost based on the adjusted input Weights of each split 
        for i in self.splits_list: 
                
                # change split cost value based on the adjusted input weight
                analytical_file_temp[i] = analytical_file_copy[i] * inputWeights[self.splits_list.str.match(i)][0]


        # horizontally sum the adjusted optimization_splits cost for every split in optimization_splits_list
        spend_optimization_period = analytical_file_temp[optimization_splits_list].sum().sum()

        # print the 2.b test difference in cost, adjusting weights spend_optimization_period vs client_budget. this should make the difference be 0.
        print('Test 2 (adjusting weights): Total spend (X) for optimization period: ', round(spend_optimization_period, 0),\
              'difference between client budget and optimization period spend (X): ', round(self.client_budget - spend_optimization_period,4))


        ## after adjusting the weights and making the client budget be the same as the optimization spend 
        
        
        # horizontally sum new split cost in the total cost column (X) for every split in self.splits_list
        analytical_file_temp.loc[:,'X'] = analytical_file_temp[self.splits_list].sum(axis = 1)

        # create the daily_simulated_forecast data frame, which is the dataframe that has the changed spend based on the input Weights
        daily_simulated_forecast = analytical_file_temp.copy()

        # create an empty dataframe where the forecast will be stored. 
        forecasts_df = pd.DataFrame()


        # iterate over each breakdown model to generate a forecast for each breakdown model based on the new spend values
        for key, values in self.cuts_df.iterrows(): 
        
            # get the pickle file path for the breakdown pickle file 
            pickle_file_path = self.base_path + values.forecaster_name_list

            # open the pickle file 
            pickle_in = open(pickle_file_path, 'rb') 

            # load the pickle file, getting the model, training_dataset, and model_df
            unpickled_model, training_dataset, model_df = pickle.load(pickle_in)

            # get a copy of the analytical_file_temp which is the dataframe that has the changed spend based on the input Weights to filter it only with the breakdown values
            analytical_file_copy = analytical_file_temp.copy()

            # create a query filtering conditions for every of the breakdown columns that need to be filtered for create the breakdown data set 
            query = ' & '.join([f"(analytical_file_copy['{col}'] == '{values[col]}')" for col in self.breakdown_ls])

            # filter the dataframe with the query to create the breakdown data set 
            analytical_file_copy = eval(f"analytical_file_copy[{query}]").reset_index(drop=True)

            # ensure that the date column (ds) is in datetime format
            analytical_file_copy['ds'] = pd.to_datetime(analytical_file_copy.ds)
            
            # crate the forecast based on the unpickled_model, values params, analytical_file_copy for future df, for values.periods forecasting periods
            forecast = forecast_model_predict(model = unpickled_model, model_df = model_df, params = values, forecasting_periods = values.periods, future_input_df = analytical_file_copy)

            # merge the breakdown filtered analytical_file with the forecast to get the 'yhat', merge on date column 'ds'. 
            forecast = forecast[['ds','yhat']].merge(analytical_file_copy, how = 'right', on = 'ds')
            
            # concat the breakdown forecast with the full forecast_df
            forecasts_df = pd.concat([forecasts_df, forecast], ignore_index=True)


        # merge the daily_simulated_forecast with the forecasts_df, to include the y_hat from the forecasts_df to daily_simulated_forecast. the merge is done by each brakdown column and date
        daily_simulated_forecast = daily_simulated_forecast.merge(forecasts_df[ self.breakdown_ls + ['ds', 'yhat']], how = 'left', on = self.breakdown_ls + ['ds'])
        
        # return the daily_simulated_forecast and the inputWeights
        return daily_simulated_forecast, inputWeights
    
    




    def _objective_function(self, inputWeights):
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
        
        # split the splits_list in pre_optimization_splits_list and optimization_splits_list
        pre_optimization_splits_list, optimization_splits_list = self._get_pre_n_optimization_splits_list()
        
        # run the pso_predict() function, that estimates a weighted forecasted spend (X) based on analytical spend in reference dates and forecast the conversion quantity (y), get the daily_simulated_forecast and the updated_weights
        daily_simulated_forecast, updated_weights = self.pso_predict(inputWeights, pre_optimization_splits_list, optimization_splits_list)

        # check if any yhat equals nan. filter the daily_simulated_forecast dataframe if the 'yhat' is nan
        error_rows = daily_simulated_forecast[daily_simulated_forecast['yhat'].isna()]
        
        # if error_rows is not empty, it means that one or mor 'yhat' values is nan. then raise an error.
        if not error_rows.empty:

            # print the a message warning saying that there are NaN values in yhat column
            print('Multiple NaN values found in yhat for:\n')

            # print the error_rows to identify the cases with NaN values
            print(error_rows[ self.breakdown_ls + ['ds', 'X', 'yhat']])

            # return an error message 
            return 'Error: Multiple NaN matched yhat found'
        
        # if error_rows is empty, then there is no need to raise an error and continue. 
        else:
            
            # get the total y_hat forecasted conversion quantity. value to optimize (maximize)
            total_yhat = daily_simulated_forecast['yhat'].sum() 

            # print the yhat
            print('yhat:', total_yhat)
            
            # since the pyswarms optimization is a minimization problem convert the total_yhat to negative, to transform the maximization problem to a minimization problem
            total_yhat = total_yhat * -1
            
            # get the optimization period spend
            spend_optimization_period = daily_simulated_forecast[optimization_splits_list].sum().sum()

            # print the difference between the client budget and the optimization period spend. 
            #print('client_budget - spend_optimization_period:', round(self.client_budget - spend_optimization_period))

            # get the global index of the particle iteration 
            global itr_idx

            ## append the itr_idx to the iteration array itr_array
            #itr_array.append(itr_idx) 

            # add 1 to the global index of the particle iteration to state that it has already pass an additional iteration
            itr_idx += 1

            # get the global log_df file 
            global log_df

            # set the splits_list to the log_df dataframe as a column
            log_df['splits_list'] = self.splits_list

            # set the updated_weights to the log_df dataframe as the 'input Weights' column 
            log_df['Input Weight'] = updated_weights

            # set the total_yhat to the log_df dataframe as a column 
            log_df['total_yhat'] = total_yhat

            # set the index of the particle iteration to the log_df dataframe as 'iter' column
            log_df['iter'] = itr_idx

            # set the current datetime as 'date' column in global log_df
            log_df['Date'] = datetime.datetime.now().strftime("%m.%d.%Y %H:%M:%S")
            
            # get the global log_file_name 
            global log_file_name
            
            # write and append the log_df to the log_file_name
            log_df.to_csv(log_file_name, mode = 'a', header = False)

            # return the total_yhat which is the optimizer value (minimizer)
            return total_yhat
        

    def _objective_function_wrap(self, x):
        """
        Function used to evaluate the objective function for each particle. 
        objective funtion generate the maximized output based on given inputWeights across particles

        Parameters:
        -------
            x: 2D numpy.ndarray
                1D for the number of particles and 2D for the number of cuts

        Returns:
        -------
            numpy.ndarray for maximized output across particles
        """

        # get the number of particles. 1D in x numpy.ndarray
        n_particles = x.shape[0]

        # evaluate the objective function for each particle and store it on a list of - total_yhat. minimizer argument
        j = [self._objective_function(x[i]) for i in range(n_particles)]

        # return a numpy array with the objective function results - total_yhat. minimizer argument
        return np.array(j)
    
            

    


        
     ## Particle Swarm Optimization
    def run_pso(self, options, init_pos: np.ndarray = None, n_particles: int = 0, itNum: int = 1):
        """
        This function runs the Particle Swarm Optimization. using _objective_function_wrap() as objective function to minimize target value in pso. 
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
        
        
        
        # get the number of particles. if  n_particles is 0, then set the number of particles from the formula n_particles = 10 + ( 2 * sqrt(num_splits) )
        if n_particles == 0:

            # set the number of particles from the formula n_particles = 10 + ( 2 * sqrt(num_splits) ). store it in self class attribute 
            self.n_particles = int(10+2*math.sqrt(len(self.splits_list)))
        
        # if n_particles is different from 0, then set the n_particles as the argument value 
        else:

            # store n_particles in self class attribute 
            self.n_particles = n_particles
        

        # set the initial position for the input weights. if init_pos is none then it is the self.constraint_file.init input weight. 
        if init_pos is None:
            init_pos = np.tile(self.constraint_file.init, [self.n_particles, 1])
        
        # set the GlobalBestPSO which is the one that initializes the pso optimization. 
        optimizer = GlobalBestPSO(n_particles = self.n_particles, dimensions = len(self.max_bound), bounds = self.bounds, options = options, init_pos = init_pos)
        
        # set the global log_file_name as the base_path + 'Input Weight History ' + current date .csv
        global log_file_name
        log_file_name = self.base_path + 'Input Weight History ' + self.today + '.csv'

        # set the global log_df file dataframe, an empty dataframe with the following columns: 'splits_list', 'Input Weight', 'totay_yhat', 'iter', 'Date'
        global log_df
        log_df = pd.DataFrame(columns = ['splits_list', 'Input Weight', 'totay_yhat', 'iter', 'Date'])

        # write to csv the empty log_df file dataframe, so that during the optimizer it writes - appends each new iteration to the log_file_name 
        log_df.to_csv(log_file_name, mode = 'w', header = True)

        #global itr_array
        #itr_array = []

        # set the global particle iteration as 0 
        global itr_idx
        itr_idx = 0
        
        # set a start time to get the elapsed time that optimization took to run 
        start = time.time()

        # catch any warning, ignoring filterwarnings
        with warnings.catch_warnings():

            # ignore filterwarnings
            warnings.filterwarnings("ignore")

            # set a random seed
            random.seed(12345)

            # run optimize the GlobalBestPSO by using the _objective_function_wrap function. it returns the cost (-yhat) and the pos (weights)
            cost, pos = optimizer.optimize(objective_func = self._objective_function_wrap, iters = itNum )

        # set an end time to get the elapsed time that optimization took to run 
        end = time.time()

        # print the duration time that the pso optimization took to run 
        print('Compute duration:', end - start)

        # create a pos_df a dataframe that has the splits_list and the weights for each split 
        pos_df = pd.DataFrame({'variable_splits': self.splits_list, 'pos': pos})

        # set the pos_df_file_name
        pos_df_file_name = self.base_path + 'Pos Result ' + self.today + '.csv'

        # save pos_df to csv in local device
        pos_df.to_csv(pos_df_file_name)

        # print a message that the file has being generated 
        print(f'{pos_df_file_name} generated')

        # get the optimized conversion conversion_quantity list, a list of the cost_history for the optimizer
        conversion_quantity = [-c for c in optimizer.cost_history]

        # set a iteration_history dataframe with the iteration and the conversion_quantity
        iteration_history = pd.DataFrame({'iteration':list(range(1,len(conversion_quantity)+1)),'Conversion Quantity': conversion_quantity})

        # set the iteration_history_file_name 
        iteration_history_file_name = self.base_path + 'Iteration History ' + self.today + '.csv'

        # save iteration_history to csv in local device
        iteration_history.to_csv(iteration_history_file_name)

        # plot the pso optimization cost history
        plot_cost_history(conversion_quantity)

        # save cost history plot to jpg in local device
        plt.savefig(self.base_path + self.today + '.jpg')
        
        # save to self class attribute a dict that contains pos_df, conversion_quantity list and iteration_history df 
        self.result = {'pos': pos_df, 'conversions': conversion_quantity, 'history': iteration_history}

        # return self.result
        return self.result 
    

    def PlotPSO_CostHistory(self):
        """
        Function used to generate the Cost history plot of the PSO optimization and save it on local device

        Parameters:
        -------
        
        Returns:
        -------
            plot_cost_history which plots the cost for each of the iterations in the PSO optimization. 
        """

        # plot the PSO cost history 
        plot_cost_history(self.result['conversions'])

        # save the PSO cost history in local device 
        plt.savefig(self.base_path + 'PSO_Cost_History' + self.today + '.jpg')




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

            # split the splits_list in pre_optimization_splits_list and optimization_splits_list
            pre_optimization_splits_list, optimization_splits_list = self._get_pre_n_optimization_splits_list()
        
            # run the pso_predict() function, that estimates a weighted forecasted spend (X) based on analytical spend in reference dates and forecast the conversion quantity (y), get the daily_simulated_forecast and the updated_weights
            daily_simulated_forecast, updated_weights = self.pso_predict(inputWeights, pre_optimization_splits_list, optimization_splits_list)

            # check if any yhat equals nan. filter the daily_simulated_forecast dataframe if the 'yhat' is nan
            error_rows = daily_simulated_forecast[daily_simulated_forecast['yhat'].isna()]
            
            # if error_rows is not empty, it means that one or mor 'yhat' values is nan. then raise an error.
            if not error_rows.empty:

                # print the a message warning saying that there are NaN values in yhat column
                print('Multiple NaN values found in yhat for:\n')

                # print the error_rows to identify the cases with NaN values
                print(error_rows[ self.breakdown_ls + ['ds', 'X', 'yhat']])

                # return an error message 
                return 'Error: Multiple NaN matched yhat found'
            
            # if error_rows is empty, then there is no need to raise an error and continue. 
            else:
                
                # get the total y_hat forecasted conversion quantity. value to optimize (maximize)
                total_yhat = daily_simulated_forecast['yhat'].sum() 

                # print the yhat
                print('yhat:', total_yhat)

                # horizontally sum the new optimization_splits cost for every split in optimization_splits_list
                spend_optimization_period = daily_simulated_forecast[optimization_splits_list].sum().sum()
                
                # print a message that shows the spend in the optimization period 
                print('Total spend in the optimization period: ', spend_optimization_period)

                # print a message that shows the difference between the spend in the optimization preiod and the client budget, it should be 0.
                print('client budget - total spend in the optimization period:', round(self.client_budget - spend_optimization_period))
        
        # return the forecast df daily_simulated_forecast
        return daily_simulated_forecast
    

    def RandomSearch(self, n_particles, rs_iters, rs_n_selection_iters):
        
        if n_particles == 0:
            n_particles = int(10+2*math.sqrt(len(self.splits_list)))

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
                dimensions=len(self.max_bound), options=params_search_options, objective_func=self._objective_function_wrap, bounds = self.bounds, 
                iters=rs_iters, n_selection_iters=rs_n_selection_iters)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            best_score, best_options = g.search()
        return best_options, best_score




