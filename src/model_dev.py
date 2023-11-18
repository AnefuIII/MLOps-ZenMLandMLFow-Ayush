import logging
from zenml import steps
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression

class Model(ABC):

    """abstract class for all models"""
    @abstractmethod
    def train(self, X_train, y_train):
        """
        X_train: training data
        y_train: training label

        """
        pass

class LinearRegressionModel(Model):
        """
        trains the model 

        args: X_train, y_train
        returns: None

        """
        def train(self, X_train, y_train, **kwargs):
            """
            trains the model

            Args:
            X_train: trainind data
            y_train: training label

            return: None
            """
             
            try:
                reg = LinearRegression(**kwargs)
                reg.fit(X_train, y_train)
                logging.info('Model training completed')

                return reg
            except Exception as e:
                logging.info('Error in training model: {}'.format(e))
                raise e

