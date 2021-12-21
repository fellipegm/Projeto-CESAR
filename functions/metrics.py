import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error


class Metrics:

    
    def __init__(self, y_true, y_pred):
        '''
        Calculate the metrics from a regression problem
        :param y_true: Numpy.ndarray or Pandas.Series
        :param y_pred: Numpy.ndarray or Pandas.Series
        :return: Dict with metrics:
        '''
        self.y_true = y_true
        self.y_pred = y_pred


    def fit(self):
        '''
        Calculate the metrics from a regression problem
        :param y_true: Numpy.ndarray or Pandas.Series
        :param y_pred: Numpy.ndarray or Pandas.Series
        :return: Dict with metrics
        '''


        mean_abs_err = mean_absolute_error(self.y_true, self.y_pred)
        mean_sqr_err = mean_squared_error(self.y_true, self.y_pred)
        r2 = r2_score(self.y_true, self.y_pred)
        maxerror = max_error(self.y_true, self.y_pred)
        residual_mean = (self.y_true - self.y_pred).mean()
        residual_std = (self.y_true - self.y_pred).std()
        
        return {'mean_abs_err' : mean_abs_err, 'mean_sqr_err' : mean_sqr_err,
                'r2': r2, 'max_error': maxerror, 
                'residual_mean': residual_mean, 'residual_std': residual_std,
                'residual': (self.y_true - self.y_pred)}
    