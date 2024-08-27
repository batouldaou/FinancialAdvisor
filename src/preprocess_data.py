import os
import pandas as pd
import numpy as np 
import logging
from config import config


class CleanData():
    def __init__(self, config):
        self.config = config 
    
    
    def load_data(self):
        """
            Load the data from csv file -> later this would be linked to SQL
        """
        filename = self.config["filename"]
        file_directory = self.config["data_dir"]
        
        try:
            df = pd.read_csv(os.path.join(file_directory, filename), 
                             skiprows=1,
                             names=self.config["column_names"])
            return df
        except Exception as e:
            logging.exception("Error during loading data:", e)
    
    
    def check_missing_values(self, df):
        pass
    
    
    def manage_missing_values(self, df):
        pass
    
    def check_duplicated_values(self, df):
        pass
    
    def check_outlier(self, df):
        pass
    
    def manage_outlier(self, df):
        pass
    
    def get_scalar(self, df):
        pass
    
    