import pytest
import os
import pandas as pd
from pathlib import Path
from .data_handle import load_data, encode_categories, seperate_data, split_data, deencode_categories
from unittest.mock import patch, mock_open
from sklearn.model_selection import train_test_split
from pandas.testing import assert_frame_equal, assert_series_equal

# mocking is creating a fack version of an object that mimics the behavior of the real object
@pytest.fixture()
def config():
    data_config = {
        "filename": "personal_budget_dataset.csv",
        "data_dir": os.path.join(Path.cwd().parent, 'data', 'raw_data'),
        "output_features": 1,
        "test_size": 0.2
    }
    return data_config


@pytest.fixture()
def get_data(config):
    data_dir = config['data_dir']
    filename = config['filename']
    return pd.read_csv(f"{data_dir}/{filename}", delimiter=',')


# Test to ensure data loads correctly
def test_loading_data(config):
    data = load_data(config)
    assert not data.empty, "The data should not be empty"
    assert isinstance(data, pd.DataFrame), "The loaded data should be a pandas DataFrame"
    

def test_encode_categories(get_data):
    column_to_encode = 'Category'
    encoded_column = encode_categories(data=get_data, column=column_to_encode)
    
    assert encoded_column[column_to_encode].dtype == int, "Encoded feature should be integer"
    assert os.path.exists('encoder.pickle'), "Encoder file pickle should be created"
    

# Test seperate fata function
def test_split_data(get_data, config):
    features, target = seperate_data(get_data, config)
    x_train, x_test, y_train, y_test = split_data(features, target, config)  
    x_train_test, x_test_test, y_train_test, y_test_test =  train_test_split(features, target, test_size=config["test_size"], random_state=42)
    
    assert_frame_equal(x_train, x_train_test)
    assert_frame_equal(y_train, y_train_test)


@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    if os.path.exists('encoder.pickle'):
        os.remove('encoder.pickle')
        

    
print("found")

