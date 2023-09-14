import numpy as np
from typing import Tuple
from scipy.stats import norm


def black_scholes_price(
        S0: float,
        K: float,
        r: float,
        sigma: float,
        t: float,
        q: float = 0,
        name: str = 'call'
        ) -> Tuple[float, float, float]:
    '''
    Calculate the price of a European vanilla option via the Garman-Kohlhagen solution to the Black-Scholes partial differential equation.

    Paramaters
    ----------
    S0: float
        Price of underlying stock.
    K: float
        Strike price.
    r: float
        Assumed risk-free interest rate (constant over time period).
    sigma: float
        Assumed volatility of underlying stock price (constant over time period).
    t: float
        Duration of the option (expiry).
    q: float = 0
        Assumed constant and continuous dividend rate (defaulted to 0).
    name: str = call
        Call or put option

    Returns
    -------
    price: float
        Price of option,
    Phi2: float,
        Risk-neutral probability that option is exercised.
    Phi1: float
        Delta of the option.
    '''
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if name.lower() == 'call':
        Phi1 = norm.cdf(d1)
        Phi2 = norm.cdf(d2)
        price = S0*np.exp(-q*t)*Phi1 - K*np.exp(-r*t)*Phi2
    elif name.lower() == 'put':
        Phi1 = norm.cdf(-d1)
        Phi2 = norm.cdf(-d2)
        price = K*np.exp(-r*t)*Phi2 - S0*np.exp(-q*t)*Phi1
    else:
        raise AssertionError("Error: name must be either 'put' or 'call'.")

    return price, Phi2, Phi1
