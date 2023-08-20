'''
Modules
-------

1. Credit_Underwriter

'''


import numpy as np
import pandas as pd

import datetime as dt
import matplotlib.pyplot as plt

import glob
from typing import Tuple

from scipy.stats import norm
from scipy.optimize import newton
from ins_mat.special.risk import risk


class Credit_Underwriter():
    def __init__(self,
                 risks: list[risk] | list[str],
                 risk_free_rate: float = 0.03,
                 online_load: bool = True,
                 maturity: float = 1
                 ) -> None:
        '''
        Credit_Underwriter
        ------------------

        - Credit Underwriter evaluates the credit-riskiness of shares using a simplifed model of a company, described completely in terms of its equity and liability.
        - It is assumed that the underlying stocks do not pay dividends (although this could be added in future) and that the company does not acquire further debt over the period of consideration.
        - We assume that markets are efficient, and as such we acquire estimates of volatility via selected option prices (implied volatility) and to evaluate the riskiness of of companies, we calculate the credit spread and the risk-neutral probability of default.
        - To calculate a risk premium, default will be taken as a Poisson process and the claim size would be a proportion of the implied debt, grossed up to maturity.

        Parameters
        ----------
        risks: list[risk] | list[str]
            List of underlying risk objects to use - if a list of strings are passed (i.e. ticker labels), then we will initialize the risk objects here, with default attributes.
        risk_free_rate: float = 0.03
            Assumed underlying risk free rate of return - which we assume to be constant.
        online_load: bool = True
            Whether to refresh the listings from internet.
        maturity: float = 1
            For derivative instruments considered, look at time to expiry.
        '''

        # All dicts specified by ticker symbol key access

        self.risks = {}

        # Check the type of `risks`- initialize where needed
        for _risk in risks:
            if isinstance(_risk, str):
                self.risks[_risk] = risk(_risk, online_load=online_load)
            elif isinstance(_risk, risk):
                self.risks[_risk.symb] = _risk
            else:
                raise TypeError('Only ticker strings and `risk` objects are currently supported.')

        self.risk_free_rate = risk_free_rate
        self.equity_volatility = {}
        self.asset_volatility = {}
        self.merton_price = {}
        self.implied_debt = {}
        self.credit_spread = {}
        self.default_prob = {}
        self.agg_mdls = {}
        self.maturity = maturity

    @classmethod
    def from_file(cls, source_folder: str, risk_free_rate: float = 0.03, maturity=1):
        '''
        Alternative constructor to load all risk information from a given folder.

        Parameters
        ----------
        source_folder: str
            Folder containing all the risk pickles
        '''

        # Loop through all pkl items in the given folder and load
        if not source_folder.endswith("/"):
            source_folder = f'{source_folder}/'

        risk_files = glob.glob(source_folder + "*.pkl")

        symb_list = []
        for risk_file in risk_files:
            _risk = risk.from_file(risk_file)
            symb_list.append(_risk)

        obj = cls(risks=symb_list, risk_free_rate=risk_free_rate, online_load=False, maturity=maturity)
        return obj

    @staticmethod
    def black_scholes_call(S0: float,  # Underlying
                           K: float,  # Strike
                           r: float,  # Risk-free rate
                           sigma: float,  # Volatility
                           t: float,  # Time to maturity
                           q: float,  # Dividends
                           ) -> Tuple[float, float]:
        '''
        Static method to solve the Black-Scholes equation for a vanilla European call option.

        Returns the value fo the call option, the risk-neutral probability that the call expies in the money and the delta of the call.
        '''
        d1 = (np.log(S0/K) + (r-q+0.5*sigma**2)*t)/(sigma*np.sqrt(t))
        d2 = d1 - sigma*np.sqrt(t)

        Phi1 = norm.cdf(d1)
        Phi2 = norm.cdf(d2)

        V = S0*np.exp(-q*t)*Phi1 - K*np.exp(-r*t)*Phi2
        return [V, Phi2, Phi1]

    def mean(self, symb: str):
        '''Return the mean of the credit losses for each symbol.'''
        mdl = self.agg_mdls[symb]
        return mdl['exposure'] * mdl['p']

    def var(self, symb: str):
        '''Return the variance of the credit losses for each symbol.'''
        mdl = self.agg_mdls[symb]
        return mdl['exposure'] * mdl['p'] * (1-mdl['p'])

    def get_option_data(self):
        '''
        Obtain option data for each risk.

        Method via URL: medium.com/[at]txlian13/webscrapping-options-data-with-python-and-yfinance-e4deb0124613

        For reading the option tables on Yahoo Finance, url:
        https://finance.yahoo.com/news/read-options-table-080007410.html?guccounter=1&guce_referrer=aHR0cHM6Ly9kdWNrZHVja2dvLmNvbS8&guce_referrer_sig=AQAAAHbsM5cC5grAC4DpGZVdlWbO3ONiUco2aLkD5JLJt_kmW2JZi7m7ap8BXrDVL8DFQu6Elk9qC2bAW02UAQL7_YjDCkjm4xWll0AVWJHose3hgwjcJpkXrTHAuT0wZquTUK2U8AgW87FcCq3fsRzk9k_3B69NCP8DEzEm5-2K5V6M

        Some key info from the above article:
        Vol (Volume) = how many contracts traded during session.
        Open Int (Interest) = the number of open positions in the contract - we use this as a proxy for demand.

        Volatility here is reported as a float value, not percentage.

        Updated properties
        ------------------
        options: list[pd.DataFrame]
            list of options associated per risk.
        '''

        self.options = {}

        for _risk in self.risks.values():
            # Get the expiry dates (yfinance.options)
            expiry_dates = _risk.options

            # Filter for expiry dates that are not immediate, we look at least 1 month from now
            exp_date_filter = dt.datetime.today() + dt.timedelta(weeks=52/12)

            options = pd.DataFrame()  # Set up empty dataase to store option information
            for T in expiry_dates:
                if dt.datetime.strptime(T, '%Y-%m-%d') < exp_date_filter:
                    continue

                option = _risk.option_chain(T)
                option = pd.concat([option.calls, option.puts], ignore_index=True)
                option['expiry'] = T
                options = pd.concat([options, option], ignore_index=True)

            # Odd issue that yields wrong expiration dates - add 1 day to correct
            options['expiry'] = pd.to_datetime(options['expiry']) + dt.timedelta(days=1)

            # Relative lifetime of option
            options['duration'] = (options['expiry'] - dt.datetime.today()).dt.days / 365

            # Boolean column if we are looking at a call option
            options['IsCall'] = options['contractSymbol'].str[4:].apply(lambda symbol: "C" in symbol)

            # Midpoint of the bid-ask, to be used as the "Price"
            options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
            options['price'] = options[['bid', 'ask']].mean(axis=1)

            # Drop other columns that are not needed
            options = options.drop(columns=[
                'contractSize',
                'currency',
                'change',
                'percentChange',
                'lastTradeDate',
                'lastPrice'
                ])

            # Use openInterest as a credibility measure - represents level of demand
            options = options.sort_values(by='openInterest', ascending=False)

            self.options[_risk.symb] = options

    def visualize_volatility(self, symb: str):
        '''
        Create plots to see volatility by:

        1. expiry dates (in duration)
        2. open interest

        Parameters
        ----------
        symb: str
            Retrieve the relevant ticker symbol to visualize.

        Returns
        -------
        duration_group: pd.DataFrame
            Pivot table of statistics of implied volatility by option duration.
        fig, ax: matplotlib figure and axes
            Returns the visuals themselves.
        '''
        # Retrieve option information
        opt = self.options[symb]

        fig, ax = plt.subplots(1, 2)
        fig.set_figwidth(15)

        # For duration vs volatility, we note that duration is quite regular and so we can calculate averages for each and display those
        ax[0].scatter(opt['duration'], opt['impliedVolatility'], s=2, c='blue', label='Raw')
        ax[0].set_xlabel('Option Duration')

        duration_group = opt.groupby('duration').agg(
                avg_implied_volatility=('impliedVolatility', np.mean),
                stdev=('impliedVolatility', np.std),
                perc75=('impliedVolatility', lambda x: np.quantile(x, 0.75)),
                perc25=('impliedVolatility', lambda x: np.quantile(x, 0.25))
                )

        ax[0].plot(duration_group.index, duration_group['avg_implied_volatility'], linewidth=2, c='red', label='Average')
        ax[0].legend()

        ax[1].scatter(opt['openInterest'], opt['impliedVolatility'], s=2, c='orange')
        ax[1].set_xlabel('Open Interest')

        for _ax in ax:
            _ax.grid()
            _ax.set_ylabel('Implied volatility')

        return duration_group, fig, ax

    def set_volatility(self, eq_vol: None | dict = None) -> None:
        '''
        Set up a list of equity volatilities to test possible implied share prices. Then convert to asset volatility.

        This is related by (equity_volatility*equity)**2 = (asset_volatility*asset*delta)**2

        Parameters
        ----------
        eq_vol: dict | None
            Dictionary of annual volatility inputs for each symbol (key). E.g.

            {
                'stk_symbol': [0.1, 0.05, 0.2],
                'new_symbol': [0.3, 0.5]
            }

            Leave as None for automatic suggestion (implied best estimate volatility, 25th percentile, 75th percentile)
        '''
        for symb, opt in self.options.items():
            _risk = self.risks[symb]
            total_shares = _risk.balance_sheet.loc['Share Issued'].iloc[0]  # Current count of shares

            # We want to consider only current assets and liabs if "short-term"
            if self.maturity > 1:
                debt_face_value = _risk.balance_sheet.loc['Total Debt'].iloc[0] / total_shares  # Strike
                total_assets = _risk.balance_sheet.loc['Total Assets'].iloc[0] / total_shares  # Current underlying
            else:
                debt_face_value = _risk.balance_sheet.loc['Current Debt'].iloc[0] / total_shares  # Strike
                total_assets = _risk.balance_sheet.loc['Current Assets'].iloc[0] / total_shares  # Current underlying

            if eq_vol is not None:
                self.equity_volatility[symb] = eq_vol[symb]
            else:
                # For automatic we will input best estimate (mean), 25th and 75th perentiles as our alternatives
                low, high = np.quantile(opt['impliedVolatility'], [0.25, 0.75])
                self.equity_volatility[symb] = [opt['impliedVolatility'].mean(), low, high]

            # Obtain the implied asset volatility
            sigma_E = self.equity_volatility[symb]

            # This needs to be solved numerically
            def bs_solver(asset_vol, equity_vol):
                _, Phi2, delta = self.black_scholes_call(
                        S0=total_assets,
                        K=debt_face_value,
                        r=self.risk_free_rate,
                        sigma=asset_vol,
                        t=self.maturity,
                        q=0
                        )

                equity = _risk.Price.iloc[0]
                return (equity_vol*equity - asset_vol*total_assets*delta)**2

            sigma_A = [newton(bs_solver, sE, args=(1.0,)) for sE in sigma_E]
            self.asset_volatility[symb] = sigma_A

    def solve(self) -> None:
        '''
        Calculate the technical premium required for each ticker.

        Start using the Merton credit model, to calculate credit risk and default probability for each ticker symbol.

        The Merton model is a structural model using the basis that firm value F(t) at time t is given by:

        F(t) = B(t) + E(t)

        where B(t) is the debt at time t and E(t) the equity at time t. The debt will have face value L, maturing at time T.

        We assume that corporate entity issues both equity and debt. At maturity, assume that company is "wound-up",
        in which case shareholders recive a payoff of E(T) = max(F(T)-L, 0) because debt is senior to equity.

        Under this framework, this is equivalent to treating shareholders as having a European vanilla call option on the assets of the comapny with maturity T and strike price equal to value of debt.

        Losses are modeled here via frequency-severity, where

        - frequency is given by the default event and its probability as calculated.
        - severity is modeled as a lognormal distribution centered at face value of the debt and standard deviation the implied volatility.
        '''

        for symb in self.risks.keys():
            _risk = self.risks[symb]
            total_shares = _risk.balance_sheet.loc['Share Issued'].iloc[0]  # Current count of shares

            # We want to consider only current assets and liabs if "short-term"
            if self.maturity > 1:
                debt_face_value = _risk.balance_sheet.loc['Total Debt'].iloc[0] / total_shares  # Strike
                total_assets = _risk.balance_sheet.loc['Total Assets'].iloc[0] / total_shares  # Current underlying
            else:
                debt_face_value = _risk.balance_sheet.loc['Current Debt'].iloc[0] / total_shares  # Strike
                total_assets = _risk.balance_sheet.loc['Current Assets'].iloc[0] / total_shares  # Current underlying

            _vols = np.array(self.asset_volatility[symb])

            implied_equity, Phi2, _ = self.black_scholes_call(
                    S0=total_assets,
                    K=debt_face_value,
                    r=self.risk_free_rate,
                    sigma=_vols,
                    t=self.maturity,
                    q=0
                    )

            self.merton_price[symb] = implied_equity

            # Calculate the credit spread
            self.implied_debt[symb] = total_assets - implied_equity
            y = (1/self.maturity)*np.log(debt_face_value/self.implied_debt[symb])  # The required yield
            self.credit_spread[symb] = y - self.risk_free_rate
            self.default_prob[symb] = (1 - Phi2)/self.maturity  # 1-year risk-neutral probability of default at maturity

            # Encapsulate losses parameters
            self.agg_mdls[symb] = {
                    'exposure': total_assets - debt_face_value,
                    'p': self.default_prob[symb][0]
                    }

    def present(self, index: int = 0) -> pd.DataFrame:
        '''
        Present for each of the risks on board the Merton implied price, implied volatility, credit spread and risk-neutral probability of default.

        Parameters
        ----------
        index: int = 0
            Present the numbers based on the chosen implied volatility estimate. Default to 0 for the first volatility item in each risk (best estimate if using defaults in 'set_volatility')

        Returns
        -------
        result: pd.DataFrame
            The above results.
        '''
        prices = []
        share_prices = []
        eq_vols = []
        a_vols = []
        spreads = []
        prob_defs = []
        symbs = self.risks.keys()
        risk_prems = []

        for s in symbs:
            prices.append(self.merton_price[s][index])
            share_prices.append(self.risks[s].Price.iloc[0])
            eq_vols.append(self.equity_volatility[s][index])
            a_vols.append(self.asset_volatility[s][index])
            spreads.append(self.credit_spread[s][index])
            prob_defs.append(self.default_prob[s][index])

            # Attain risk premium information (loss cost estimate)
            risk_prems.append(self.mean(s))

        return pd.DataFrame({
            'Ticker': symbs,
            'Merton_Price': prices,
            'Market_Price': share_prices,
            'Implied_Equity_Volatility': eq_vols,
            'Implied_Asset_Volatility': a_vols,
            'Credit_Spread': spreads,
            'Default_Probability': prob_defs,
            'Risk_premium': risk_prems,
            })

    def save_all(self, folder_path: str = "./") -> None:
        '''
        Save all underlying risk objects into chosen folder.

        Parameters
        ----------
        folder_path: str = "./"
            Specify current directory to store all risk pickles. Defaults to current working directory.
        '''
        p = f'{folder_path}' if folder_path.endswith("/") else f'{folder_path}/'

        for _, _risk in self.risks.items():
            _risk.export(path=f'{p}{_risk.symb}.pkl')
