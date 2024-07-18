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

import pickle
from prophet import Prophet
from .FileSaving import *
import gcsfs
import numpy as np

# Holidays
Blackfriday = pd.DataFrame({
    'holiday': 'Blackfriday', 
    'ds': pd.to_datetime(['2018-11-23', '2019-11-29', '2020-11-27', '2021-11-26', '2022-11-25','2023-11-24','2024-11-29']),
    'lower_window': -7,
    'upper_window': 7})

CyberMon = pd.DataFrame({
    'holiday': 'CyberMon', 
    'ds': pd.to_datetime(['2018-11-26', '2019-12-02', '2020-11-29', '2021-11-29', '2022-11-28','2023-11-27','2024-11-27']),
    'lower_window': 0,
    'upper_window': 7})

Valentines = pd.DataFrame({
    'holiday': 'Valentines', 
    'ds': pd.to_datetime(['2018-02-14', '2019-02-14', '2020-02-14', '2021-02-14', '2022-02-14','2023-02-14','2024-02-14']),
    'lower_window': -3,
    'upper_window': 1})

Mothers = pd.DataFrame({
    'holiday': 'Mothers',
    'ds': pd.to_datetime(['2018-05-13', '2019-05-12', '2020-05-26', '2021-05-09', '2022-05-08','2023-05-14','2024-05-12']),
    'lower_window': -7,
    'upper_window': 1
})

Independence = pd.DataFrame({
    'holiday': 'Independence',
    'ds': pd.to_datetime(['2018-07-04', '2019-07-04', '2020-07-03', '2021-07-05', '2022-07-04','2023-07-04','2023-07-04','2024-07-04']),
    'lower_window': -7,
    'upper_window': 1
})

Labor = pd.DataFrame({
    'holiday': 'Labor',
    'ds': pd.to_datetime(['2018-09-03', '2019-09-02', '2020-09-07', '2021-09-06', '2022-09-05','2023-09-04','2024-09-02']),
    'lower_window': -7,
    'upper_window': 1
})
Christmas = pd.DataFrame({
    'holiday': 'Christmas',
    'ds': pd.to_datetime(['2018-12-25', '2019-12-25', '2020-12-25', '2021-12-25', '2022-12-25','2023-12-25','2024-12-25']),
    'lower_window': -7,
    'upper_window': 1
})

customized_holidays = pd.concat(( Blackfriday, CyberMon, Valentines, Mothers, Independence, Labor, Christmas))

def CreateModel(training_dataset, row):
    """
    # Model Generator
    # Return with model and training dataset for later use
    
    row: pd series
    # Check if growth is power.
    # If growth is power then transform the X and set frowth to linear.

    """
    pt_kwargs = {k: row[k] for k in row.keys() if k in ['changepoint_prior_scale', 'seasonality_prior_scale', 'holidays_prior_scale', 'seasonality_mode']}
    if row.growth == 'power':
        training_dataset.X = np.power(training_dataset.X, float(row.power))
        growth = 'linear'
    else: 
        growth = row.growth
    if row.include_holiday in ('Default', True):
        model = Prophet(growth = growth,  **pt_kwargs) #
        model.add_country_holidays(country_name = row.market)
    elif row.include_holiday == 'Custom':
        model = Prophet(growth = growth, holidays = customized_holidays, **pt_kwargs) #
    else:
#         print('Holiday not included')
        model = Prophet(growth = growth, **pt_kwargs)
    
    if row.input_mode in ('additive', 'multiplicative'):
        model.add_regressor('X', mode = row.input_mode)
    
    if row.covid_year_exclude == 'Exclude':
        training_dataset.drop(training_dataset[(training_dataset['ds'] >= '2020-01-01')&\
                                               (training_dataset['ds']<='2020-12-31')].index, axis = 0, inplace = True)

    for i in row.keys():
        if i in ['weekday_or_weekend', 'covid_year_dummy', 'anomaly_2021', \
                 'generic_cost_split', 'competitor_cost_split', 'new_confirmed', 'Nov_Dec_2018', 'Jul_Dec_2019']\
            and row[i] in ('additive', 'multiplicative'):
            model.add_regressor(i, mode = row[i])
            
    model.fit(training_dataset)
    
    return model


class GenerateModelFile:
    def __init__(self):
        pass
    
    def _Pickling(self, model_n_data, bucket_name, path_name, base_path = './'):

        with open(base_path + path_name.split('/')[-1], 'wb') as f:
            pickle.dump(model_n_data,f)
        
        fs = gcsfs.GCSFileSystem()
        fs.ls(bucket_name)
        path_name = bucket_name + '/' + path_name
        # print(path_name)
        with fs.open(path_name, 'wb') as handle:
            pickle.dump(model_n_data, handle)
                
    def Generate(self, cuts_df, df, breakdown_ls, bucket_name, folder_name, base_path = './'):
        for key, values in cuts_df.iterrows():
            if len(breakdown_ls) == 4:
                model_df = df[(df[breakdown_ls[0]] == values[breakdown_ls[0]]) & \
                               (df[breakdown_ls[1]] == values[breakdown_ls[1]]) & \
                              (df[breakdown_ls[2]] == values[breakdown_ls[2]]) & \
                              (df[breakdown_ls[3]] == values[breakdown_ls[3]])].reset_index(drop = True)
            elif len(breakdown_ls) == 3:
                model_df = df[(df[breakdown_ls[0]] == values[breakdown_ls[0]]) & \
                               (df[breakdown_ls[1]] == values[breakdown_ls[1]]) & \
                              (df[breakdown_ls[2]] == values[breakdown_ls[2]])].reset_index(drop = True)
            elif len(breakdown_ls) == 2:
                model_df = df[(df[breakdown_ls[0]] == values[breakdown_ls[0]]) & \
                               (df[breakdown_ls[1]] == values[breakdown_ls[1]])].reset_index(drop = True)
            elif len(breakdown_ls) == 1:
                model_df = df[(df[breakdown_ls[0]] == values[breakdown_ls[0]])].reset_index(drop = True)
            
            try:
                training_dataset = model_df[(model_df['ds'] < new_cut_off_point) & (model_df['ds'] >= values.starting_date)]
            except:
                training_dataset = model_df[(model_df['ds'] < values.cutoff_month)]
                
            model = CreateModel(training_dataset, values)

            pickle_file_name = ' '.join([values[i] for i in breakdown_ls])
            path_name = folder_name + '/' + pickle_file_name + '.pkl'
            self._Pickling((model, training_dataset, model_df), bucket_name, path_name, base_path)




