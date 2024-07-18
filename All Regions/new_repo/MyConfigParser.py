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

import datetime # for date variables 
import configparser # handling configuration files

class MyConfigParser:
    """
    Function used to convert a .ini config object to dictonary for the convenience of using.
    Data is converted according to the nature of keys; str (default), float (for numeric keys), and datetime.datetime for dates, which need to be in '%Y-%m-%d' str format.
    
    Parameters:
    -------
    config_file_path: str 
        .ini config file path

    
    Arguments:
    -------
    self.config: config object
        Config object read from config file path.
    
    Returns:
    -------
    dic: dictionary
        converted dictionary
    """
    
    # initialize the config parser, read the config file 
    def __init__(self, config_file_path): 
        
        # initialize the configparser function
        self.config = configparser.ConfigParser() 
        
        # read the .ini config file 
        self.config.read(config_file_path) 
    
    
     # create a dictionary from the self.config object
    def GetDict(self):
    
        # sets an empty dict to add every section of the config object
        sections_dict = {} 

        # get sections and iterate over each
        sections = self.config.sections() 

        # loop over each section to get each element name and value, parse it and store it into the dict
        for section in sections:
            
            # get the config key names (option) in the section
            options = self.config.options(section)
            
            # create a temp dict that will create the key names and elements (values)
            temp_dict = {}
            
            # loop over each key (option) of the section in the config object
            for option in options:
                
                # get the option value
                value = self.config.get(section,option)
                
                # parse the value to the element type. 
                    #Try to parse as float (number), 
                    # if it does not parse it to float try to parse it to a date type by the format '%Y-%m-%d' 
                    # else parse the element as a str (default), if str.lower() == 'null' parse it as a None value.
                try: 
                    value = float(value)
                except ValueError:
                    try: 
                        value = datetime.datetime.strptime(value, '%Y-%m-%d')
                    except ValueError:
                        if value.lower() == 'null':
                            value = None
                    
                # add the option key and value to the temp dict
                temp_dict[option] = value
            
            # add each section dict to the full dict to be returned 
            sections_dict[section] = temp_dict

        return sections_dict

