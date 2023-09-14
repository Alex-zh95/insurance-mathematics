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
    market_history: list[float]
    option_price: float
    option_strike: float
    option_type: str
    option_maturity: float
    currency: str
    assets: float
    liabilities: float
    dividends: float = 0


def get_returns(price_vect: np.ndarray) -> Tuple[np.ndarray, dict]:
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
    returns = price_vect[1:] / price_vect[:-1]

    mdl = {
            'dist': lognorm,
            'properties': lognorm.fit(np.log(returns), floc=0)
            }

    return returns, mdl


def wang_transform(P: np.ndarray | float, sharpe_ratio: float) -> np.ndarray | float:
    '''
    The Wang transform converts a given probability into a risk-adjusted probability.

    Implementation is $G(x) = N(N^{-1}(F(x)) + sharpe_ratios)$

    where N is the standard normal cumulative distribution.

    This function can effectively be used as a change-of-measure transform (i.e. stand-in for Radon-Nikodym derivative).

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
    return norm.cdf(norm.ppf(P) + sharpe_ratio)


class credit_risk():
    def __init__(self,
                 risks: list[risk],
                 limit: float = 10e6,
                 debt_maturity: float = 1,
                 r: float = 0.03
                 ):
        self.list_risks = risks
        self.r = r
        self.maturity = debt_maturity
        self.limit = limit

        # Blank result inits
        self.asset_volatilities = {}
        self.equity_volatilities = {}
        self.implied_debts = {}
        self.rn_default_probability = {}
        self.rates = {}
        self.sharpe_ratios = {}
        self.returns_mdls = {}

    def calculate_implied_volatility(self, rsk) -> None:
        '''
        Use the Black-Scholes equation to obtain

        - implied equity volatilities for each of the risks in the list,
        - implied asset volatilities for each of the risks in the list by solving:

        `(equity_volatility * equity_volatility)**2 = (asset_volatilities*asset*delta)**2`

        Parameters
        ----------
        rsk: risk
            Selected risk in portfolio.
        '''

        # Helper function wrappers around the Black-Sholes solver
        def bse(v: float, _rsk) -> float:
            out, _, _ = rn.black_scholes_price(
                    S0=_rsk.market_history[-1],
                    K=_rsk.option_strike,
                    r=self.r,
                    sigma=v,
                    t=_rsk.option_maturity,
                    q=_rsk.dividends,
                    name=_rsk.option_type
                    )
            return out

        def bse_equity_to_asset(
                equity_volatility: float,
                asset_volatility: float,
                current_equity_price: float,
                _rsk,
                ) -> float:
            out, Phi2, delta = rn.black_scholes_price(
                    S0=_rsk.assets / _rsk.shares_issued,
                    K=_rsk.liabilities / _rsk.shares_issued,
                    r=self.r,
                    sigma=asset_volatility,
                    t=self.maturity,
                    q=0,
                    name='call'
                    )

            return (equity_volatility * current_equity_price - asset_volatility * _rsk.assets * delta)**2

        # Create initial guess by looking at returns
        _, returns_model = get_returns(rsk.market_history[-1])
        v_init = lognorm.std(*returns_model['properties'])

        # Solve iteratively for equity volatility
        self.equity_volatilities[rsk.name] = newton(lambda v: bse(v) - rsk.option_price, v_init, args=(rsk,))

        self.asset_volatilities[rsk.name] = newton(bse_equity_to_asset, self.equity_volatilities[rsk.name], args=(1.0, rsk.market_history[-1], rsk))

        # Also save the returns model
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
        '''

        if ovr_returns is not None:
            adj_returns = ovr_returns
        else:
            cur_risk_return_mdl = self.returns_mdls[rsk.name]
            adj_returns = lognorm.mean(*cur_risk_return_mdl['properties'])

        self.sharpe_ratios[rsk.name] = (ovr_returns - self.r) / adj_returns

    def generate_rate(self, rsk, rn: bool = True) -> None:
        '''
        Calculate the technical premium required for each ticker.

        Parameters
        ----------
        rsk: risk
            Selected risk in portfolio to solve for.
        rn: bool = True
            Whether to use the risk neutral probabilities or not.

        Notes
        -----
        Start using the Merton credit model, to calculate credit risk and default probability for each ticker symbol.

        The Merton model is a structural model using the basis that firm value $F(t)$ at time $t$ is given by:

        $F(t) = B(t) + E(t)$

        where $B(t)$ is the debt at time t and $E(t)$ the equity at time t. The debt will have face value $L$, maturing at time $T$.

        We assume that corporate entity issues both equity and debt. At maturity, assume that company is "wound-up",
        in which case shareholders recive a payoff of $E(T) = max(F(T)-L, 0)$ because debt is senior to equity.

        Under this framework, this is equivalent to treating shareholders as having a European vanilla call option on the assets of the comapny with maturity $T$ and strike price equal to value of debt.

        Losses are modeled here via frequency-severity, where

        - frequency is given by the default event and its probability as calculated.
        - severity is modeled as a lognormal distribution centered at face value of the debt and standard deviation the implied volatility.
        '''
        debt_face_value = rsk.liabilities / rsk.shares_issued

        implied_equity, Phi2, _ = rn.black_scholes_price(
                S0=rsk.assets / rsk.shares_issued,
                K=debt_face_value,
                r=self.r,
                sigma=self.asset_volatilities[rsk.name],
                t=self.maturity,
                q=0
                )

        self.rn_default_probability[rsk.name] = Phi2,

        y = (1/self.maturity) * np.log(debt_face_value/implied_equity)
        self.credit_spread[rsk.name] = y - self.r

        # Calculate a rate for the contract
        if rn:
            self.rates[rsk.name] = rsk.market_history[-1] * Phi2
        else:
            # Use the Wang transform to convert from risk-neutral probability to actuarial probability
            # One method is to utilize sharpe_ratio to make converison.
            act_phi2 = wang_transform(P=Phi2, sharpe_ratio=self.sharpe_ratios[rsk.name])
            self.rates[rsk.name] = rsk.market_history[-1] * act_phi2

        # Calculate premiums based on the provided limits
        self.premiums[rsk.name] = self.rates[rsk.name] * self.limit

    def pipeline_solve(self) -> None:
        '''
        Pipeline all the actions in this class for all risks in portfolio.
        '''
        for risks in self.list_risks:
            # Step 1: attain implied volatilities
            self.calculate_implied_volatility(risk)

            # Step 2: Require sharpe ratios for all risks (no adjustments/overrides possible)
            self.calculate_implied_volatility(risk)

            # Step 3: Generate rates (using actuarial probabilities only)
            self.generate_rate(risk, rn=True)
