import logging
import numpy as np
from abc import ABC, abstractmethod

from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    abstract class defining strategy for evaluating our models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):

        """
        calculate the scores of the model

        Arg: y_true: for true labels,
             y_pred: for predicted labels
        returns:
             None
        """
        pass

class MSE(Evaluation):

    """
    Mean Squared Error Evaluation Strategy
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        try:

            logging.info('Calculating MSE')
            mse = mean_squared_error(y_true, y_pred)
            logging.info('mse: {}'.format(mse))
            return mse
        except Exception as e:
            logging.info('Error in calculating mse: {}'.format)
            raise e
        

class R2(Evaluation):

    """
    R2 score Evaluation Strategy
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        try:

            logging.info('Calculating R2_score')
            r2 = r2_score(y_true, y_pred)
            logging.info('R2_score: {}'.format(r2))
            return r2
        except Exception as e:
            logging.info('Error in calculating R2_score: {}'.format)
            raise e
        
class RMSE(Evaluation):

    """
    Root Mean Squared Error Evaluation Strategy
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        try:

            logging.info('Calculating RMSE')
            rmse = mean_squared_error(y_true, y_pred, squared = False)
            logging.info('RMSE: {}'.format(rmse))
            return rmse
        except Exception as e:
            logging.info('Error in calculating RMSE: {}'.format)
            raise e
        
