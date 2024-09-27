'''
Useful functions for fitting negative binomial distributions, since there is no "nice" MLE available.

We need to "know" one of the parameters in advance, so a good start is to use Method of Moments to generate initial guesses,
i.e. for the shape parameter r = mean**2/(var - mean)

We can then solve the likelihood function for the shape parameter.
'''

import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.optimize import newton
from typing import Tuple


def nb_method_of_moments(data: np.ndarray | pd.DataFrame | pd.Series) -> Tuple[float, float]:
    '''
    Use method of moments estimator to attain parameters of negative binomial distribution.

    Parameters
    ----------
    data: np.ndarray | pd.DataFrame | pd.Series
        Data to use

    Returns
    -------
    r, k: float, float
        Shape and scale of the negative binomial distribution (see scipy.stats.nbinom)
    '''
    mean, var = data.mean(), data.var()
    r = mean**2 / (var - mean)
    k = mean / var
    return (r, k)


def nb_likelihood(r: float, data: np.ndarray | pd.DataFrame | pd.Series) -> float:
    '''
    Likelihood function based on initial known shape parameter r (as per scipy implementation) and provided data.

    Parameters
    ----------
    r: float
        Shape parameter for negative binomial distribution (see scipy.stats.nbinom)
    data: np.ndarray | pd.DataFrame | pd.Series
        Data vector to use

    Returns
    -------
    likelihood: float
        Value of the likelihood function given shape, r
    '''

    N = data.shape[0]
    sum_digamma = np.sum(digamma(data + r)) - N * digamma(r)
    log = N * np.log(r / (r + np.sum(data / N)))
    return sum_digamma + log


def nb_mle_get_prob(r: float, data: np.ndarray | pd.DataFrame | pd.Series) -> float:
    '''
    Returns the scale parameter given the shape parameter, r.

    Parameters
    ----------
    r: float
        Shape parameter for negative binomial distribution (see scipy.stats.nbinom)
    data: np.ndarray | pd.DataFrame | pd.Series
        Data vector to use

    Returns
    -------
    shape: float
        Shape parameter
    '''
    N = data.shape[0]
    return N * r / (N * r + np.sum(data))


def get_frequency_measures(data: pd.DataFrame | pd.Series | np.ndarray,
                           excl_nil: bool = True,
                           prompt: str = "sample") -> Tuple[float, float] | float:
    '''
    Attempt to fit frequency distributions to provided data. The negative binomial distribution is prioritized and will be returned if there are no errors in convergence or no invalid parameters are returned. Otherwise, we will return a Poisson distribution parameter.

    Distributional parameters all determined by maximum likelihood.

    Parameters
    ----------
    data: pd.DataFrame | pd.Series | np.ndarray
        Data to be fitted
    excl_nil: bool = True
        Exclude nil data points (i.e. exclude when data == 0) in fit
    prompt: str = "sample"
        A label for the dataset (optional), to aid with diagnostic reports.

    Returns
    -------
    estimators: Tuple[float, float] | float
        The negative binomial shape and scale paramters if succesful on fitting a negative binomial distribution. Otherwise, return the Poisson mu parameter.
    '''

    _data = data[data > 0] if excl_nil else data

    N_mean = _data.mean()

    # NB fits by MLE are iterative, initialize with initial guess via Method of Moments
    nb_init = nb_method_of_moments(_data)

    try:
        r = newton(nb_likelihood, nb_init[0], args=(_data))
        k = nb_mle_get_prob(r, _data)
    except RuntimeError:  # Failure to converge results in this error
        print('MLE convergence failed for {prompt}. Reverting to Poisson frequency.')
        return N_mean

    if r < 0:
        print('Invalid negative binomial shape parameter ({k:,.5f}) for {prompt}. Reverting to Poisson frequency.')
        return N_mean

    return (r, k)
