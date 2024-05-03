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

import datetime
import configparser

class MyConfigParser:
    """
    Function used to convert config object to dictonary for the convenience of using.
    Data types are converted according to the nature of keys.
    
    Parameters:
    -------
    config: config
        config object.
    
    Arguments:
    -------
    self.config: config object
        Config object read from config file path.
    
    Returns:
    -------
    dic: dictionary
        converted dictionary
    """
    def __init__(self):
        pass
        # self.config = configparser.ConfigParser()
        # self.config.read(config_file_path)
    
    def GetDict(self, config):
        self.config = config

        sections_dict = {}

        # get sections and iterate over each
        sections = self.config.sections()

        for section in sections:
            options = self.config.options(section)
            temp_dict = {}
            for option in options:
                value = self.config.get(section,option)
                if option in ['base_periods', 'periods', 'weekstarting']:
                    value = float(value)
                elif value == 'null':
                    value = None
                elif option in ['cut_off_point', 'reference_start_date1', 'reference_end_date1', 'reference_start_date2', 'reference_end_date2']:
                    try:
                        value = datetime.datetime.strptime(value, '%Y-%m-%d')#.date()
                    except:
                        pass
                temp_dict[option] = value
                
            sections_dict[section] = temp_dict
        # print(sections_dict)
        return sections_dict

