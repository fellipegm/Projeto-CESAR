# Based on https://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas
import numpy as np

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas series objects of equal length

    Returns
    ----------
    crosscorr : float
    """

    minlen = min(datax.shape[0], datay.shape[0])
    
    # if datax.shape != datay.shape:
    #     raise IndexError('The arrays should have the same length')

    
    return np.corrcoef(np.array(datax[0:minlen].shift(-lag, fill_value=0)), np.array(datay[0:minlen]))[0,1]