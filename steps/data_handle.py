import os
import pandas as pd
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(config):
    """
        Load the data from csv file -> later this would be linked to SQL
    """
    filename = config["filename"]
    file_directory = config["data_dir"]
    
    try:
        df = pd.read_csv(os.path.join(file_directory, filename))
                            #skiprows=1,
                            #names=config["column_names"])
        return df
    except Exception as e:
        logging.exception("Error during loading data:", e)


def encode_categories(data:pd.DataFrame, column:str) -> pd.DataFrame:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    with open("encoder.pickle", 'wb') as file:
        pickle.dump(encoder, file)
    return data
    

def seperate_data(data, config):
    features = data.iloc[:,:-config["output_features"]]
    target = data.iloc[:, -config["output_features"]:]
    return features, target


def split_data(features, target, config):
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=config["test_size"], random_state=42)
    return x_train, x_test, y_train, y_test


def deencode_categories(data):
    with open ('encoder.pickle', 'rb') as file:
        encoder = pickle.load(file)
    
    de_ecnoded = encoder.inverse_transform(data)
    return de_ecnoded