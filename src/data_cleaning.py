import logging
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    class defining strategy for handling data

    """
    @abstractmethod
    def HandleData(self, data = pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessingStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """
    def HandleData(self, data = pd.DataFrame) -> pd.DataFrame:

        """
        proprocess data

        """
        try:
            data = data.drop(['order_approved_at',
                              'order_delivered_carrier_date',
                              'order_estimated_delivery_date',
                              'order_purchase_timestamp',
                              ],
                              axis = 1)
            
            data['product_weight_g'].fillna(data['product_weight_g'].median(), inplace = True)
            data['product_length_cm'].fillna(data['product_length_cm'].median(), inplace = True)
            data['product_height_cm'].fillna(data['product_height_cm'].median(), inplace = True)
            data['product_width_cm'].fillna(data['product_width_cm'].median(), inplace = True)
            data['review_comment_message'].fillna('no review', inplace = True)
            
            data = data.select_dtypes(include = [np.number])
            cols_to_drop = ['customer_zip_code_prefix', 'order_item_id']
            data = data.drop(cols_to_drop, axis = 1)

            return data
        
        except Exception as e:
            logging.error('Error in preprocessing data: {}'.format(e))
            raise e
        

class DataDivStrategy(DataStrategy):
        
        """
        class for train test split of data
        """
        def HandleData(self, data=pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
                """
                train test and split the dataset
                """
                try:
                     X = data.drop(['review_score'], axis = 1)
                     y = data['review_score']

                     X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                         test_size= 0.2,
                                                                         random_state= 42)
                     return X_train, X_test, y_train, y_test
                except Exception as e:
                     logging.info('error in dividing data: {}'.format(e))

                     raise e
                
class DataCleaning:
     """
     Class which preprocess the data and divide it into train test and split
     """
     def __init__(self, data: pd.DataFrame, strategy = DataStrategy):
          self.data = data
          self.strategy = strategy

     def HandleData(self) -> Union[pd.DataFrame, pd.Series]:
          """
          handle data by cleaning and dividing
          """
          try:
               return self.strategy.HandleData(self.data)
          except Exception as e:
               logging.info('Error in handling data: {e}'.format(e))

"""
if __name__ == '__main__':
     
     data = pd.read_csv('url to data')
     datacleaning = DataCleaning(data, DataPreProcessingStrategy())
     datacleaning.HandleData()
"""
                


