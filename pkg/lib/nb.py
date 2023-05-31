'''
Useful functions for fitting negative binomial distributions, since there is no "nice" MLE available.

We need to "know" one of the parameters in advance, so a good start is to use Method of Moments to generate initial guesses,
i.e. for the shape parameter r = mean**2/(var - mean)

We can then solve the likelihood function for the shape parameter.
'''

import numpy as np
from scipy.special import digamma
from typing import Tuple


def nb_method_of_moments(data: np.ndarray) -> Tuple[float, float]:
    '''
    Use method of moments estimator to attain parameters of negative binomial distribution.

    Parameters
    ----------
    data: np.ndarray
        Data to use

    Returns
    -------
    r, k: float, float
        Shape and scale of the negative binomial distribution (see scipy.stats.nbinom)
    '''
    mean, var = data.mean(), data.var()
    r = mean**2/(var - mean)
    k = mean/var
    return (r, k)


def nb_likelihood(r: float, data: np.ndarray) -> float:
    '''
    Likelihood function based on initial known shape parameter r (as per scipy implementation) and provided data.

    Parameters
    ----------
    r: float
        Shape parameter for negative binomial distribution (see scipy.stats.nbinom)
    data: np.ndarray
        Data vector to use

    Returns
    -------
    likelihood: float
        Value of the likelihood function given shape, r
    '''

    N = data.shape[0]
    sum_digamma = np.sum(digamma(data + r)) - N*digamma(r)
    log = N*np.log(r/(r + np.sum(data/N)))
    return sum_digamma + log


def nb_mle_get_prob(r: float, data: np.ndarray) -> float:
    '''
    Returns the scale parameter given the shape parameter, r.

    Parameters
    ----------
    r: float
        Shape parameter for negative binomial distribution (see scipy.stats.nbinom)
    data: np.ndarray
        Data vector to use

    Returns
    -------
    shape: float
        Shape parameter
    '''
    N = data.shape[0]
    return N*r/(N*r + np.sum(data.values))
