import logging

import pandas as pd
from zenml import step


class IngestData:
    # """
    # ingesting data from the data path
    # """
    def __init__(self, data_path: str):
        # """
        # Args:
        #     data_path = path to the data
        # """
        self.data_path = data_path

    def get_data(self):
        # """
        # ingesting the data from the data path
        # """
        logging.info(f'ingesting data from {self.data_path}')
        return pd.read_csv(self.data_path) #i added from index_col to parese dates
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    # """
    # ingesting data from the data path

    # """

    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f'Error while ingesting data {e}')
        return e