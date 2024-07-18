
from google.cloud import bigquery
import pandas as pd
import numpy as np

def GetBQData(query = None, input_col = None, output_col = None):
        """
        Function used to get data from BigQuery table.
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
        if query == None:
            raise ValueError("No query is passed to get the data.")
        
        client = bigquery.Client()
        bq_client = bigquery.Client()
        
        search_df = bq_client.query(query)\
                .result()\
                .to_dataframe()
        search_df.fillna(method = 'ffill', inplace = True)
        search_df.rename(columns = {'Date': 'ds'},inplace=True)
        search_df.ds = pd.to_datetime(search_df.ds)
            
        if not input_col is None:
            try:
                search_df.rename(columns = {input_col: 'X'},inplace=True)
            except:
                NameError()
        if not output_col is None:
            try:
                search_df.rename(columns = {output_col: 'y'},inplace=True)
            except:
                NameError()

        return search_df

def _Reindex(updated_df):
    distinct_cuts = updated_df[['market', 'account_type', 'funnel', 'Identifier']].drop_duplicates().reset_index(drop = True)
    
    idx = distinct_cuts.market + ' ' + distinct_cuts.Identifier + ' ' + distinct_cuts.account_type + ' ' + distinct_cuts.funnel
    date_index = pd.date_range(updated_df.ds.min(), updated_df.ds.max()) # why was it df.ds.min and max?
    multiple_index = pd.MultiIndex.from_product([date_index, idx])
    
    updated_df['temp_idx'] = updated_df.market + ' ' + updated_df.Identifier + ' ' + updated_df.account_type + ' ' + updated_df.funnel 
    updated_df = updated_df.set_index(['ds', 'temp_idx']).reindex(multiple_index)
    
    updated_df = updated_df.reset_index().rename(columns = {'level_0': 'ds'})
    
    updated_df[['market', 'Identifier', 'account_type', 'funnel']] = updated_df.level_1.str.split(expand = True)
    updated_df.drop(['level_1'], axis = 1, inplace = True)
    updated_df['Weekend'] = updated_df['ds'].apply(lambda x: 1 if x.dayofweek > 4 else 0)
    
    return updated_df

def Combine(df, identifier = 'L', small_country_str = ''):
    df = df.copy()
    df.loc[:, 'new_funnel'] = df['funnel'].replace([ x for x in df.funnel.unique() if x != 'Brand'], 'Non_Brand')
    df.account_type = df.account_type.str.replace(' ', '_')

    df.loc[:, 'market'] = df['market'].str.replace(' ', '')
    df.loc[:, 'Identifier'] = identifier
    if len(small_country_str) > 0:
        df.loc[df['market'].str.contains(small_country_str), "Identifier"] = 'S'
  
    large_countries_ori = df[df['Identifier'] == identifier]
    small_countries_details = df[df['Identifier'] == 'S'].copy()
    
    
    large_countries_ori.loc[:, 'account_type'] = '-'
    large_countries_ori.loc[large_countries_ori[large_countries_ori['funnel'] == 'Generic'].index, 'generic_cost'] = large_countries_ori.loc[large_countries_ori['funnel'] == 'Generic', 'X']
    large_countries_ori.loc[large_countries_ori[large_countries_ori['funnel'] == 'Competitor'].index, 'competitor_cost'] = large_countries_ori.loc[large_countries_ori['funnel'] == 'Competitor', 'X']
    
    large_total_cost_df = large_countries_ori.groupby(['ds', 'region', 'market', 'account_type','new_funnel', 'Identifier']).sum().reset_index()

    large_countries = large_total_cost_df.copy()
    large_countries.drop(['funnel'],axis=1, inplace=True)
    large_countries = large_countries.rename(columns = {'new_funnel': 'funnel'})
    large_countries.loc[:, 'generic_cost_split'] = large_countries.generic_cost/(large_countries.generic_cost + large_countries.competitor_cost)
    large_countries.loc[:, 'competitor_cost_split'] = large_countries.competitor_cost/(large_countries.generic_cost + large_countries.competitor_cost)
    large_countries.loc[:, 'account_type'] =  '-'
    
    try:
        small_countries_details.loc[:, 'account_type'] = '-'
        small_countries_details = small_countries_details.groupby(['region', 'market', 'account_type', 'funnel', 'ds']).sum().reset_index()
        small_countries_market_level_cost = small_countries_details.groupby(['region', 'market', 'ds']).sum().reset_index()

        small_countries_details = small_countries_details.rename(columns = {'X': 'funnel_X'})

        small_countries_details = small_countries_details.merge(small_countries_market_level_cost[['region', 'market', 'ds', 'X']], how = 'left', on = ['region', 'market', 'ds'])

        small_countries_details.loc[:, 'cost_split'] = small_countries_details.funnel_X/small_countries_details.X
        small_country_split = df[df['Identifier'] == 'S'].copy()
        small_country_split.loc[:, 'account_type'] = '-'
        small_country_split = small_country_split
        small_countries = df[df['Identifier'] == 'S'].copy()
        small_countries.loc[:, 'account_type'] = '-'
        small_countries.loc[:, 'funnel'] = '-'
        small_countries.loc[:, 'market'] = 'SM'
        small_countries = small_countries.groupby(['region', 'market', 'account_type', 'funnel', 'ds', 'Identifier']).sum().reset_index()
    except Exception as err:
        print(f"Unexpected {err}, {type(err)} while handling small market group.")
    
    updated_df = pd.concat([small_countries, large_countries]).reset_index(drop = True)
#     print(updated_df[(updated_df.ds>'2022-10-31')].sort_values('ds').head())
    updated_df = _Reindex(updated_df)
#     print(updated_df[(updated_df.ds>'2022-10-31')].sort_values('ds').head())
    # updated_df.generic_cost_split = updated_df.generic_cost_split.apply(lambda x: np.nan if x < 0.005 else x)
    # updated_df.competitor_cost_split = updated_df.competitor_cost_split.apply(lambda x: np.nan if x < 0.005 else x)    
    
    updated_df.generic_cost_split = updated_df.generic_cost_split.fillna(.5)
    updated_df.competitor_cost_split = updated_df.competitor_cost_split.fillna(.5)
    
    updated_df.generic_cost_split = updated_df.apply(lambda x: round(x['generic_cost_split'], 2) if x['generic_cost_split']+x['competitor_cost_split']!=1 else x['generic_cost_split'], axis = 1)
    updated_df.competitor_cost_split = 1-updated_df.generic_cost_split
    
    return _Reindex(updated_df)

def GetCovidData(region):
    
    query_string = f"""
    SELECT date, country_name, new_confirmed
    FROM `ups-analytics.bx_forecast_optimization.covid_confirmed_cases_with_country_name` 
    where region = '{region}'
    """

    client = bigquery.Client()
    bq_client = bigquery.Client()

    covid_ori_df = (
                    bq_client.query(query_string)
                    .result()
                    .to_dataframe()
                    )
    covid_df = covid_ori_df.copy()

    # covid_df.loc[covid_df['country_name'].str.contains(small_market_str), 'country_name'] = 'SM'
    covid_df = covid_df.groupby(['date', 'country_name']).sum().reset_index()
    covid_df #.country_name.unique()

    covid_df.rename(columns={'date':'ds', 'country_name': 'market'}, inplace=True)
    covid_df['ds'] = pd.to_datetime(covid_df.ds)
    covid_df = covid_df.sort_values(['ds']).reset_index(drop = True)
    for i in covid_df[covid_df.new_confirmed<0].index:
        covid_df.loc[i, 'new_confirmed'] = round((covid_df.loc[i-1, 'new_confirmed'] + covid_df.loc[i+1, 'new_confirmed'])/2, 0)

    covid_df.new_confirmed = covid_df.new_confirmed.fillna(0)
    # df.loc[(df['ds'] > '2021-01-01'), 'new_confirmed'] = df.loc[(df['ds'] >= '2020-12-01') & (df['ds'] <= '2020-12-31'), 'new_confirmed'].mean()
    return covid_df
