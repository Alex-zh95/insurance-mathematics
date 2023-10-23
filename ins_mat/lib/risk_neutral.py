import numpy as np
from typing import Tuple
from scipy.stats import norm
from scipy.optimize import newton


def black_scholes_price(
        S0: float,
        K: float,
        r: float,
        sigma: float,
        t: float,
        q: float = 0,
        name: str = 'call') -> Tuple[float, float, float]:
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
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if name.lower() == 'call':
        Phi1 = norm.cdf(d1)
        Phi2 = norm.cdf(d2)
        price = S0 * np.exp(-q * t) * Phi1 - K * np.exp(-r * t) * Phi2
    elif name.lower() == 'put':
        Phi1 = norm.cdf(-d1)
        Phi2 = norm.cdf(-d2)
        price = K * np.exp(-r * t) * Phi2 - S0 * np.exp(-q * t) * Phi1
    else:
        raise AssertionError("Error: name must be either 'put' or 'call'.")

    return price, Phi2, Phi1


def wang_transform(P: np.ndarray | float,
                   sharpe_ratio: float,
                   inverse: bool = False) -> np.ndarray | float:
    '''
    Wang transform converts a given probability into a risk-adjusted probability.

    Implementation is $G(x) = N(N^{-1}(F(x)) + sharpe_ratio)$

    where $N$ is standard normal cumulative distribution function.

    This function can effectively be used as a change-of-measure transform (i.e. a Radon-Nikodym derivative), in this case between a risk-neutral (risk-adjusted) measure and an actuarial measure.

    Parameters
    ----------
    P: np.ndarray | float
        Probabilite to convert.
    sharpe_ratio. float
        Also termed "market price of risk".
    inverse: bool = False
        Decide whether to compute forward or inverse transform.

    Returns
    -------
    G: np.ndarray | float
        Converted probabilities.
    '''
    if inverse:
        return norm.cdf(norm.ppf(P) - sharpe_ratio)
    else:
        return norm.cdf(norm.ppf(P) + sharpe_ratio)


def implied_asset_volatility(
        vE: np.ndarray,
        sig_e: float,
        debt_face_value: float,
        maturity: float = 1.0,
        r: float = 0.01,
        n_iter: int = 50) -> float:
    '''
    Asset volatility is not observable from market data and so a common approach is to derive an implied estimate using equity information.

    Approach here is analogous to the EM algorithm (2-step optimization):

    1. Solve for asset values using assumed asset volatility,
    2. Solve for asset volatlity using updated asset values.

    Implied asset ($\sigma_A$) and equity volatility ($\sigma_E$) are related by the following:

    $$\sigma_E \times E = \sigma_A \times A \frac{\partial E}{\partial A}$$

    Observe the Radon-Nikodym derivative on the RHS of the above. This can be simplified (in Black-Scholes framework) as the Delta of call option, which we derive in step 1. Thus equation above simplifies to

    $$\sigma_E \times E = \sigma_A \times A N(d_1)$$

    with $N(d_1)$ the Delta. Note: We do not use the simultaneous equation approach here, instead, looping steps 1 and 2 until we get "Cauchy"-convergence of the $\sigma_A$ estimates.

    Idea for implementation in this manner: https://www.bradfordlynch.com/blog/2017/05/20/ProbabilityOfDefault.html

    Parameters
    ----------
    vE: np.ndarray,
        Historic stock value array
    sig_e: float,
        Equity volatility
    debt_face_value: float
        Face value of the debt (used as Strike price in Merton model)
    maturity: float = 1.0
        Duration of debts held, defaulted to 1.0
    r: float = 0.01
        Risk-free rate of return, defaulted to 1%
    n_iter: int = 50
        Maximum number of iterations before we stop the algorithm, defaulted to 50.

    Returns
    -------
    sig_a: float
        Implied asset volatility
    '''

    # Helper functions
    def implied_equity_error(a0: float, e0: float) -> float:
        '''
        Solve Black-Scholes equation with current asset value a0 to get implied current equity. E.g. pass to scipy.optimize.newton to attain roots.

        Return the difference between implied current equity and actual equity, e0.
        '''
        implied_equity, _, _ = black_scholes_price(
            S0=a0,
            K=debt_face_value,
            sigma=sig_a,
            t=maturity,
            r=r)

        return e0 - implied_equity

    def empirical_asset_vol(prev_iter_sig_a: float, vA: np.ndarray) -> Tuple[float, float]:
        '''
        Update the asset volatility given newly calculated asset values.

        prev_iter_sig_a is to keep the old iteration of sig_a
        '''

        # Calculate log returns - np.roll shifts everything up one index with wrapping
        log_returns = np.log(vA / np.roll(vA, 1))
        log_returns[0] = np.nan

        # Calculate new asset volatility
        sig_a = np.nanstd(log_returns)

        return sig_a, np.abs(prev_iter_sig_a - sig_a)

    # Duration of the vector of equity values
    N = len(vE)

    # Use the equity vector as initial guess of asset values
    vA = vE.copy()

    # We there is no leverage (no debt), the asset volatility is the same as equity volatility
    if debt_face_value < 1e-6:
        return sig_e

    # Get initial guess of sig_a, using equity volatility
    sig_a, _ = empirical_asset_vol(prev_iter_sig_a=0, vA=vE)

    for _ in range(n_iter):
        for i in range(N):
            # Calculate using this guess of sig_a the implied asset value
            vA[i] = newton(implied_equity_error, vA[i], args=(vE[i],))

        # Now update sig_a with this vector of asset values
        sig_a, diff = empirical_asset_vol(prev_iter_sig_a=sig_a, vA=vA)

        # If we get convergence before n_iter expires, we exit loop
        if diff < 1e-3:
            break

    return sig_a


def default_probability(
        a0: float,
        mu_a: float,
        sig_a: float,
        debt_face_value: float,
        duration: float = 1.0):
    '''
    Use Merton Distance to Default model to calculate probability of default (risk-neutral). Distance to default, DD, is calculated as:

    $$\text{DD} = \frac{\frac{A_0}{L} + (\mu_A + 0.5\sigma_A^2)T}{\sigma_A \sqrt{T}}$$

    with

    - $A_0$ as current asset value,
    - $\mu_A$ as the expected firm asset drift,
    - $\sigma_A$ as the asset volatility,
    - $T$ the duration we are looking at (default to 1)
    - $L$ is the debt face value.

    The risk neutral probability of default is $N(-\text{DD})$, under assumption that frequency of default is normal.

    Parameters
    ----------
    a0: float,
        Current value of assets
    mu_a: float,
        Expected firm asset drift
    sig_a: float,
        Asset volatility
    debt_face_value: float,
        Value of debt as given
    duration: float = 1,
        Duration of the debt, defaulted to 1

    Returns
    -------
    result: float,
        Risk-neutral probability of default.
    '''
    DD = (np.log(a0 / debt_face_value) + (mu_a + 0.5 * sig_a**2) * duration) / (sig_a * np.sqrt(duration))

    return norm.cdf(-DD)
