import logging

from zenml import step
import pandas as pd

from src.data_cleaning import DataCleaning, DataDivStrategy, DataPreProcessingStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_df(df: pd.DataFrame) -> tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:

    """
    cleans the data that has been ingested

    Args: df = raw data
    returns: 
    X_train: training data, 
    y_train y_training label
    X_test: testing data, 
    y_test: testing label
    """
    try: 
        process_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_df = data_cleaning.HandleData()

        divide_strategy = DataDivStrategy()
        data_cleaning = DataCleaning(processed_df, divide_strategy)
        X_train, y_train, X_test, y_test = data_cleaning.HandleData()

        logging.info('Data cleaning completed')
        return X_train, y_train, X_test, y_test

    except Exception as e:
        logging.info('Error in data cleaning: {}'.format(e))
        raise e

