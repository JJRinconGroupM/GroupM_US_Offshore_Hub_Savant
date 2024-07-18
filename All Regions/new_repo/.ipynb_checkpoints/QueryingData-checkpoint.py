
from google.cloud import bigquery # Google Cloud Bigquery client library
import pandas as pd # for data processing 
import numpy as np # for numeric calculations and array structures


def GetBQData(query = None, input_col = None, output_col = None):
        """
        Function used to get search historical data from BigQuery table, using the query argument to pull data from BQ.
        Na is forward filled.    
        
        Parameter:
        -------
            query: string
                Executable Bigquery Script.
            input_col: string
                Used only when returned BQ column needs to be renamed.
            output_col: string
                Used only when returned BQ column needs to be renamed.
                
        Returns:
        -------
            search_df: DataFrame
               DataFrame which contains search data with column read by fbprohet. 
        """
        
        # raise a ValueError if there is no query passed as an argument 
        if query == None:
            raise ValueError("No query is passed to get the data.")
        
        
        #client = bigquery.Client()
        # set the BQ client to pass the query to 
        bq_client = bigquery.Client()
        
        # get the query from BQ and pass it to a pandas data frame
        search_df = bq_client.query(query).result().to_dataframe()
        
        # fill na values with a forward fill
        search_df.ffill(inplace = True)
        
        # rename 'Date' column to 'ds', column name that uses prophet for date time index 
        search_df.rename(columns = {'Date': 'ds'},inplace=True)
        
        # ensure the date column ('ds') is in date time format
        search_df.ds = pd.to_datetime(search_df.ds)
            
        # rename input and output columns as 'X' and 'y' respectively, if they are already named like that it wont rename any column 
        
        # if input_col argument is not None try to rename the column to 'X', if it couldn't rename it raise the error for debuging 
        if not input_col is None:
            try:
                search_df.rename(columns = {input_col: 'X'},inplace=True)
            except:
                NameError()
        # if output_col argument is not None try to rename the column to 'y', if it couldn't rename it raise the error for debuging 
        if not output_col is None:
            try:
                search_df.rename(columns = {output_col: 'y'},inplace=True)
            except:
                NameError()

        # return the search data frame 
        return search_df

    
    
    

def _Reindex(updated_df, config_dict):
    """
        Function used to re index a dataframe accordingly to date, breakdowns and a identifier aux variable.
        this function is used in combine() function, after appending the small and large countries df into one.
        
        Parameter:
        -------
            updated_df: DataFrame
                Daily data data frame extracted from BQ.
            config_dict: dictionary
                a dictionary containing the configurations for the model, specifically the breakdown columns
                
        Returns:
        -------
            updated_df: DataFrame
               DataFrame with the reindexing. 
    """
    
    # get the break down column names 
    breakdown_columns = [config_dict['CLIENTINFO'][i] for i in sorted([key for key in config_dict['CLIENTINFO'] if key.startswith('breakdown')]) if config_dict['CLIENTINFO'][i] != None]
    
    # add to the breakdown_columns list the 'identifier' another column that identifies large markets and small markets 
    breakdown_columns = breakdown_columns #+ ['Identifier']
    
    # get the distinct cuts (breakdowns)
    distinct_cuts = updated_df[breakdown_columns].drop_duplicates().reset_index(drop = True)
    
    # create an index series for each cut (breakdown)
    idx = distinct_cuts.astype(str).apply(lambda row: ' '.join(row), axis=1)
    
    # create a date index, all dates between the min date and max date in updated_df 
    date_index = pd.date_range(updated_df.ds.min(), updated_df.ds.max()) 
    
    # create a multiple index, a join index from breadowns and date index
    multiple_index = pd.MultiIndex.from_product([date_index, idx],names = ['ds','idx'])
    
    # create a temporal breadown index in the updated df
    updated_df['temp_idx'] = updated_df[breakdown_columns].astype(str).apply(lambda row: ' '.join(row), axis=1)
    
    # set a joint index (breakdown and date index) in the updated df. reindex the df with the multiple_index. This creates a right join with multiple_index, where all index rows in multiple_index that are not present in updated_df are created with Nan values, and all row index present in updated_df but not in multiple_index are dropped. 
    updated_df = updated_df.set_index(['ds', 'temp_idx']).reindex(multiple_index)
    
    # reset index, add index leves as columns
    updated_df = updated_df.reset_index()
    
    # split idx (breakdown index) into each breakdown column variable
    updated_df[breakdown_columns] = updated_df.idx.str.split(expand = True)
    
    # drop idx column
    updated_df.drop(['idx'], axis = 1, inplace = True)
    
    # create a weekend column, to identify weekend dates from weekday
    updated_df['Weekend'] = updated_df['ds'].apply(lambda x: 1 if x.dayofweek > 4 else 0)
   
    # returns the updated_df
    return updated_df





def Combine(df, identifier = 'L', small_market_str = '', config_dict = None):
    """
        Function used to combine some breakdowns into smaller groups of breakdowns. 
        It combines multiple market breakdowns into a small markets (SM) market breakdown, and Generic and Competitor funnel breakdown into Non_Brand funnel breakdown
        
        Parameter:
        -------
            df: DataFrame
                Daily data data frame extracted from BQ.
            identifier: str
                a string to identify large markets that are not going to be combine into one breakdown
            small_market_str: str
                a string that list markets to be combined into the SM breakdown
            config_dict: dictionary
                a dictionary containing the configurations for the model, specifically the breakdown columns
                
        Returns:
        -------
            updated_df: DataFrame
               DataFrame with the combined data. 
    """

    # remove blank spaces ' ' from market values
    df.loc[:, 'market'] = df['market'].str.replace(' ', '')

    # lower funnel values 
    df['funnel'] = df['funnel'].str.lower()

    # replace funnel breadown with Brand and Non_Brand, replacing all funnel different to Brand with non brand 
    df.loc[:, 'new_funnel'] = np.where(df['funnel'] == 'brand', 'brand', 'non_brand')

    # handle large markets: combine non brand funnel and set funnel cost and funnel cost split by market
    try:
        # create a filtered df with just large markets
        large_markets = df.loc[~df['market'].str.contains(small_market_str)].copy()

        # replace all account_type values to '-'
        large_markets.loc[:, 'account_type'] = '-'

        # create a filtered df with just non brand funnel for large markets
        large_markets_NB = large_markets.loc[large_markets['new_funnel'] == 'non_brand'].copy()

        # create a df grouped by region, market, new funnel, funnel and ds for cost
        LM_NB_market_funnel_cost =  large_markets_NB.groupby(['region', 'market', 'account_type', 'new_funnel', 'funnel', 'ds']).sum()['X'].reset_index().rename(columns={'X': 'cost'})

        # create a df grouped by region, market, new funnel and ds for cost
        LM_NB_market_cost =  large_markets_NB.groupby(['region', 'market', 'account_type', 'new_funnel', 'ds']).sum()['X'].reset_index() 

        # merge the grouped by market and funnel and the gruped by market df 
        LM_NB_market_funnel_cost = LM_NB_market_funnel_cost.merge(LM_NB_market_cost, how = 'right', on = ['region', 'market', 'account_type', 'new_funnel', 'ds'])

        # create a cost_split variable which is the % of the total cost for each funnel  
        LM_NB_market_funnel_cost.loc[:, 'cost_split'] = np.where(LM_NB_market_funnel_cost.X !=0, LM_NB_market_funnel_cost.cost/LM_NB_market_funnel_cost.X, 0)

        # pivot wider by funnel to get a cost and cost split by each non brand funnel 
        LM_NB_market_cost = LM_NB_market_funnel_cost.pivot(columns='funnel', index=['region','market', 'account_type', 'new_funnel','ds'],values=['cost','cost_split']).reset_index()

        # rename columns 
        LM_NB_market_cost.columns = ['_'.join(filter(None, col)).strip() for col in LM_NB_market_cost.columns.values] # ['_'.join([col[1], col[0]]).strip() if col[1] else col[0] for col in LM_NB_market_cost.columns] 

        # group by region, market, new funnel and ds the large markets df 
        large_markets = large_markets.groupby(['region','market', 'account_type','new_funnel','ds']).sum(numeric_only=True).reset_index()

        # merge the large markets with the large markets non brand to include in the large markets df the funnel cost and funnel cost split
        large_markets = large_markets.merge(LM_NB_market_cost, how='left',on= ['region', 'market', 'account_type', 'new_funnel', 'ds'])

        # fill Nan values with 0, mainly for funnel cost and funnel cost split for brand 
        large_markets = large_markets.fillna(0)

    except Exception as err:
        print(f"Unexpected {err}, {type(err)} while handling large markets.")

    # handle small markets: combine all markets, combine funnel and set funnel cost and funnel cost split 
    try:
        # create a filtered df with just small markets
        small_markets = df.loc[df['market'].str.contains(small_market_str)].copy()

        # replace all market values to 'SM'
        small_markets.loc[:, 'market'] = 'SM'

        # replace all account_type values to '-'
        small_markets.loc[:, 'account_type'] = '-'

        # replace all new_funnel values to '-'
        small_markets.loc[:, 'new_funnel'] = '-'

        # create a data frame of small markets grouped by region, market, funnel and ds. to get the cost by each funnel 
        small_market_funnel_cost = small_markets.groupby(['region', 'market', 'account_type', 'new_funnel', 'funnel', 'ds']).sum(numeric_only=True)['X'].reset_index().rename(columns = {'X': 'cost'})

        # create a data frame of small markets grouped by region, market and ds. to get the total cost of each market
        small_market_cost = small_markets.groupby(['region', 'market', 'account_type', 'new_funnel', 'ds']).sum(numeric_only=True)['X'].reset_index()

        # merge the grouped by market and funnel and the goruped by market df 
        small_market_funnel_cost = small_market_funnel_cost.merge(small_market_cost, how = 'left', on = ['region', 'market', 'account_type', 'new_funnel', 'ds'])

        # create a cost_split variable which is the % of the total cost for each funnel  
        small_market_funnel_cost.loc[:, 'cost_split'] = np.where(small_market_funnel_cost.X!=0, small_market_funnel_cost.cost/small_market_funnel_cost.X, 0)

        # pivot wider by funnel to get a cost and cost split by each non brand funnel 
        small_market_cost = small_market_funnel_cost.pivot(columns='funnel', index=['region', 'market', 'account_type', 'new_funnel', 'ds'],values=['cost', 'cost_split']).reset_index()

        # rename columns 
        small_market_cost.columns = ['_'.join(filter(None, col)).strip() for col in small_market_cost.columns.values]

        # group by region, market, new funnel and ds the Small markets df 
        small_markets = small_markets.groupby(['region', 'market', 'account_type', 'new_funnel', 'ds']).sum(numeric_only=True).reset_index()


        # merge the small markets with the small markets to include in the the funnel cost and funnel cost split
        small_markets = small_markets.merge(small_market_cost, how='left',on= ['region', 'market', 'account_type', 'new_funnel', 'ds'])

        # fill Nan values with 0, mainly for funnel cost and funnel cost split 
        small_markets = small_markets.fillna(0)

    except Exception as err:
        print(f"Unexpected {err}, {type(err)} while handling small market group.")
    
    # concat small markets df and large markets df 
    updated_df = pd.concat([small_markets, large_markets]).reset_index(drop = True).rename(columns = {'new_funnel': 'funnel'})

    # update cost_brand for large markets 
    updated_df.loc[:,'cost_brand'] = np.where(updated_df['funnel']=='brand',1, 
                                            np.where(updated_df['funnel']=='non_brand',0,updated_df['cost_brand'])
                                            )

    # update cost_split_brand for large markets 
    updated_df.loc[:,'cost_split_brand'] = np.where(updated_df['funnel']=='brand',1, 
                                            np.where(updated_df['funnel']=='non_brand',0,updated_df['cost_split_brand'])
                                            )

    # reindex data frame 
    updated_df = _Reindex(updated_df,config_dict)
    
    return updated_df

'''
def GetCovidData(region):
    """
        Function used to get Covid data (new confirmed cases by date) from a BigQuery table.
        
        Parameter:
        -------
            region: string
                name to identify the country or region to get the covid data 
                
        Returns:
        -------
            covid_df: DataFrame
               DataFrame which contains covid data for the selected region. 
        """
    
    # create the query to get the covid data in BQ. get date, country, and new confirmed (new confirmed cases of covid).
    query_string = f"""
    SELECT date, country_name, new_confirmed
    FROM `ups-analytics.bx_forecast_optimization.covid_confirmed_cases_with_country_name` 
    where region = '{region}'
    """

    # set the BQ client to pass the query to 
    #client = bigquery.Client()
    bq_client = bigquery.Client()

    # get the query from BQ and pass it to a pandas data frame
    covid_ori_df = bq_client.query(query_string).result().to_dataframe()
                    
    # create a copy of the original data frame to handle the df making some changes
    covid_df = covid_ori_df.copy()

    # covid_df.loc[covid_df['country_name'].str.contains(small_market_str), 'country_name'] = 'SM'
    
    # group by date and country, with the sum of new confirmed cases 
    covid_df = covid_df.groupby(['date', 'country_name']).sum().reset_index() 

    # rename columns 
    covid_df.rename(columns={'date':'ds', 'country_name': 'market'}, inplace=True)

    # ensure date column has is in a date time format type
    covid_df['ds'] = pd.to_datetime(covid_df.ds)

    # sort values by date
    covid_df = covid_df.sort_values(['ds']).reset_index(drop = True)

    # handle negative values of new confirmed with the average of the day before and the day after
    for i in covid_df[covid_df.new_confirmed<0].index:
        covid_df.loc[i, 'new_confirmed'] = round((covid_df.loc[i-1, 'new_confirmed'] + covid_df.loc[i+1, 'new_confirmed'])/2, 0)

    # fill Nan values with 0 
    covid_df.new_confirmed = covid_df.new_confirmed.fillna(0)
    
    # return covid df 
    return covid_df
'''