'''
Note: for runtime call

for _risk in self.list_risks:
    # Step 1: attain implied volatilities
    self.calculate_implied_volatility(_risk)

    # Step 2: Require sharpe ratios for all risks (no adjustments/overrides possible)
    self.calculate_sharpe_ratio(_risk)

    # Step 3: Generate rates (using actuarial probabilities only)
    self.generate_rate(_risk, use_rn=False)
'''
import numpy as np
from scipy.stats import lognorm, norm
from scipy.optimize import newton

from ins_mat.lib import risk_neutral as rn
from dataclasses import dataclass
from typing import Tuple


@dataclass
class risk:
    '''
    Encapsulate the information that can be obtained from external service providers.
    '''
    name: str
    ticker: str
    sector: str
    shares_issued: int
    market_history: list[float] | np.ndarray
    option_price: float
    option_strike: float
    option_type: str
    option_maturity: float
    currency: str
    assets: float
    liabilities: float
    dividends: float = 0


def get_returns(price_vect: np.ndarray | list[float]) -> Tuple[np.ndarray, dict]:
    '''
    Calculate the YoY percentage change on stock value also empirical estimations of lognormal model.

    Parameters
    ----------
    price_vect: np.ndarray
        Vector of current prices.

    Returns
    -------
    returns: np.ndarray
        YoY percentage on stock price, noting array will have 1 less element than price_vect.
    lg_mdl:
        Dictionary describing the lognormal model, in the following form:

        {
            dist: scipy.stats.lognorm,
            properties: lognormal parameters
        }
    '''
    returns = price_vect[1:] / price_vect[:-1]  # Daily returns
    daily_params = lognorm.fit(returns, floc=0)
    daily_j = lognorm.mean(*daily_params) - 1
    daily_s = lognorm.std(*daily_params)

    # Convert from daily to annual model
    annual_return = (1 + daily_j)**365
    var_return = (1 + 2*daily_j + daily_j**2 + daily_s**2)**365 - (1 + daily_j)**(2*365)

    # Parameterize new lognormal based on the above new moments
    new_sigma = np.sqrt(np.log(var_return/annual_return**2 + 1))
    new_mu = np.log(annual_return) - 0.5*new_sigma**2

    mdl = {
            'dist': lognorm,
            'properties': (new_sigma, 0, np.exp(new_mu))
            }

    return returns, mdl


def inverse_wang_transform(P: np.ndarray | float, sharpe_ratio: float) -> np.ndarray | float:
    '''
    The Wang transform converts a given probability into a risk-adjusted probability.

    Implementation is $G(x) = N(N^{-1}(F(x)) + sharpe_ratios)$

    where N is the standard normal cumulative distribution.

    This function can effectively be used as a change-of-measure transform (i.e. stand-in for Radon-Nikodym derivative).

    We calculate in this function the inverse Wang transform.

    Parameters
    ----------
    P: np.ndarray | float
        Array of probabilities to convert.
    sharpe_ratio: float
        Sharpe ratio required for the risk.

    Returns
    -------
    G: np.ndarray | float
        Risk-adjusted probabilities
    '''
    return norm.cdf(norm.ppf(P) - sharpe_ratio)


class credit_module():
    def __init__(self,
                 risks: list[risk],
                 limit: float = 100e6,
                 debt_maturity: float = 1,
                 r: float = 0.03
                 ):
        # Generate dictionary of the risks
        self.dict_risks = {}
        for rsk in risks:
            self.dict_risks[rsk.name] = rsk

        self.r = r
        self.maturity = debt_maturity
        self.limit = limit

        # Blank result inits
        self.asset_volatilities = {}
        self.equity_volatilities = {}
        self.implied_debts = {}
        self.rn_default_probability = {}
        self.ac_default_probability = {}
        self.sharpe_ratios = {}
        self.returns_mdls = {}
        self.premiums = {}
        self.credit_spread = {}

    def calculate_implied_volatility(self, rsk, override: float | None = None) -> None:
        '''
        Use the Black-Scholes equation to obtain

        - implied equity volatilities for chosen risk (can be overriden if external data available),
        - implied asset volatilities for chosen risk in the list by solving:

        `(equity_volatility * equity)**2 = (asset_volatilities*asset*delta)**2`

        Parameters
        ----------
        rsk: risk
            Selected risk in portfolio.
        '''

        # Helper function wrappers around the Black-Sholes solver
        def bse(v: float, _rsk) -> float:
            out, _, _ = rn.black_scholes_price(
                    S0=_rsk.market_history[0],
                    K=_rsk.option_strike,
                    r=self.r,
                    sigma=v,
                    t=_rsk.option_maturity,
                    q=_rsk.dividends,
                    name=_rsk.option_type
                    )
            return out

        def bse_equity_to_asset(
                asset_volatility: float,
                equity_volatility: float,
                _rsk,
                ) -> float:
            current_assets = _rsk.assets / _rsk.shares_issued
            current_liabilities = _rsk.liabilities / _rsk.shares_issued

            out, Phi2, delta = rn.black_scholes_price(
                    S0=current_assets,
                    K=current_liabilities,
                    r=self.r,
                    sigma=asset_volatility,
                    t=self.maturity,
                    q=0,
                    name='call'
                    )

            current_equity_price = _rsk.market_history[0]
            return ((equity_volatility * current_equity_price) - (asset_volatility * current_assets * delta))

        # Solve iteratively for equity volatility
        if override is None:
            self.equity_volatilities[rsk.name] = newton(lambda v: bse(v, rsk) - rsk.option_price, 1.0)
        else:
            self.equity_volatilities[rsk.name] = override

        # Convert from equity volatility to asset volatility (unobservable)
        try:
            self.asset_volatilities[rsk.name] = newton(bse_equity_to_asset, 1.0, args=(self.equity_volatilities[rsk.name], rsk,))
        except RuntimeError:
            self.asset_volatilities[rsk.name] = self.equity_volatilities[rsk.name]
            print('Warning: Failed to converget for asset volatiltiies, using equity volatility as proxy.')

        # Set up returns model for later use
        _, returns_model = get_returns(rsk.market_history)
        self.returns_mdls[rsk.name] = returns_model

    def calculate_sharpe_ratio(self, rsk, ovr_returns: float | None = None) -> None:
        '''
        Calculate the sharpe ratio for the provided risk.

        Parameters
        ----------
        rsk: risk
            Selected risk object to calculate sharpe ratio.
        ovr_returns: float | None = None
            Override a return for the provided risk if desired. Otherwise, we use the modeled returns.

        Notes
        -----
        The sharpe ratio is calculated as:

        sr = (risk return - risk-free rate) / equity volatility

        It is also termed "market price of risk"
        '''

        if ovr_returns is not None:
            adj_returns = ovr_returns
        else:
            cur_risk_return_mdl = self.returns_mdls[rsk.name]
            adj_returns = lognorm.mean(*cur_risk_return_mdl['properties']) - 1

        risk_free_rate = self.r
        self.sharpe_ratios[rsk.name] = (adj_returns - risk_free_rate) / self.equity_volatilities[rsk.name]

    def generate_rate(self, rsk, use_rn: bool = False) -> None:
        '''
        Calculate the technical premium required for each ticker.

        Parameters
        ----------
        rsk: risk
            Selected risk in portfolio to solve for.
        rn: bool = False
            Whether to use the risk neutral probabilities or not.

        Notes
        -----
        Start using the Merton credit model, to calculate credit risk and default probability for each ticker symbol.

        The Merton model is a structural model using the basis that firm value $F(t)$ at time $t$ is given by:

        $F(t) = B(t) + E(t)$

        where $B(t)$ is the debt at time t and $E(t)$ the equity at time t. The debt will have face value $L$, maturing at time $T$.

        We assume that corporate entity issues both equity and debt. At maturity, assume that company is "wound-up",
        in which case shareholders recive a payoff of $E(T) = max(F(T)-L, 0)$ because debt is senior to equity.

        Under this framework, this is equivalent to treating shareholders as having a European vanilla call option on the assets of the company with maturity $T$ and strike price equal to value of debt.

        Losses are modeled here via frequency-severity, where

        - frequency is given by the default event and its probability as calculated.
        - severity is modeled as a lognormal distribution centered at face value of the debt and standard deviation the implied volatility.
        '''
        debt_face_value = rsk.liabilities / rsk.shares_issued

        # Implied equity on a per share basis
        implied_equity, Phi2, _ = rn.black_scholes_price(
                S0=rsk.assets/rsk.shares_issued,
                K=debt_face_value,
                r=self.r,
                sigma=self.asset_volatilities[rsk.name],
                t=self.maturity,
                q=0
                )

        self.rn_default_probability[rsk.name] = (1 - Phi2)/self.maturity

        y = (1/self.maturity) * np.log(debt_face_value/implied_equity)
        self.credit_spread[rsk.name] = y - self.r

        # Calculate the loss given default (lgd)
        lgd = np.min([self.limit, rsk.liabilities])

        if use_rn:
            self.premiums[rsk.name] = self.rn_default_probability[rsk.name] * lgd
            self.ac_default_probability[rsk.name] = 0
        else:
            # Use the Wang transform to convert from risk-neutral probability to actuarial probability
            # One method is to utilize sharpe_ratio to make converison.
            prob_act_default = inverse_wang_transform(
                    P=self.rn_default_probability[rsk.name],
                    sharpe_ratio=self.sharpe_ratios[rsk.name]
                    )
            self.ac_default_probability[rsk.name] = prob_act_default
            self.premiums[rsk.name] = self.ac_default_probability[rsk.name] * lgd
