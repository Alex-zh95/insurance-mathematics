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

from scipy.stats import norm, lognorm
from ins_mat.special.risk import risk
from ins_mat.agg_dist.fft_poisson import poisson_fft_agg


class Credit_Underwriter():
    def __init__(self,
                 risks: list[risk] | list[str],
                 risk_free_rate: float = 0.03,
                 online_load: bool = True
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
        self.Volatility = {}
        self.merton_price = {}
        self.implied_debt = {}
        self.credit_spread = {}
        self.default_prob = {}
        self.severity = {}
        self.agg_mdls = {}

    @classmethod
    def from_file(cls, source_folder: str, risk_free_rate: float = 0.03):
        '''
        Alternative constructor to load all risk information from a given folder.

        Parameters
        ----------
        source_folder: str
            Folder containing all the risk pickles
        '''

        # Loop through all pkl items in the given folder and load
        risk_files = glob.glob(source_folder + "*.pkl")

        symb_list = []
        for risk_file in risk_files:
            _risk = risk.from_file(risk_file)
            symb_list.append(_risk)

        obj = cls(risks=symb_list, risk_free_rate=risk_free_rate, online_load=False)
        return obj

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

    def set_volatility(self, Volatility: None | dict = None) -> None:
        '''
        Set up a list of volatilities to test possible implied share prices.

        Parameters
        ----------
        Volatility: dict | None
            Dictionary of volatility inputs for each symbol (key). E.g.

            {
                'stk_symbol': [0.1, 0.05, 0.2],
                'new_symbol': [0.3, 0.5]
            }

            Leave as None for automatic suggestion (implied best estimate volatility, 25th percentile, 75th percentile)
        '''
        for symb, opt in self.options.items():
            if Volatility is not None:
                self.Volatility[symb] = Volatility[symb]
            else:
                # For automatic we will input best estimate (mean), 25th and 75th perentiles as our alternatives
                low, high = np.quantile(opt['impliedVolatility'], [0.25, 0.75])
                self.Volatility[symb] = [opt['impliedVolatility'].mean(), low, high]

    @staticmethod
    def lognorm_params(mean: float, var: float) -> dict:
        r'''
        Method of moments parameter generation of the lognormal distribution.

        $$E(X) = e^{\mu + 0.5 \sigma^2}$$
        $$V(X) = E(X)^2 (e^{\sigma^2} - 1)$$

        Parameters
        ----------
        mean: float
            Observed mean of lognormal distribution
        var: float
            Observed variance of lognormal distribution

        Returns
        -------
        dict:

            In the following form:

            {
                'dist': scipy.stats.lognorm,
                'properties': (s, location=0, scale)
            }

            See the Scipy manual for the layout of the lognorm distribution.
        '''
        sigma = np.sqrt(np.log(var/mean**2 + 1))
        mu = np.log(mean - 0.5*sigma**2)

        return {
                'dist': lognorm,
                'properties': (sigma, 0, np.exp(mu))
                }

    def _calc_aggregate_loss(self, symb: str) -> None:
        '''
        Private helper class to attain the aggregate loss object from the provided frequency-severity parametrizations.
        '''
        aggregate_model = poisson_fft_agg(
                frequency=self.default_prob[symb],
                severity_distribution=self.severity[symb]
                )

        aggregate_model.compile_aggregate_distribution()
        self.agg_mdls[symb] = aggregate_model

        # Throw errors if the model fit produces something that makes no sense
        if np.abs(aggregate_model.diagnostics['Distribution_total'] - 1.) > 0.05:
            raise RuntimeError("Unsuitable aggregate model produced. Discrete probabilities do not sum to 1")

    def solve(self, maturity: float = 1.0) -> None:
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

        Parameters
        ----------
        maturity: float = 1.0
            The maturity to use for all underlying tickers in years. We default to 1 year.
        '''

        for symb in self.risks.keys():
            _risk = self.risks[symb]
            # _opt = self.options[symb]

            # current_share_price = _risk.info['currentPrice']

            # We want to consider only current assets and liabs if "short-term"
            if maturity > 1:
                debt_face_value = _risk.balance_sheet.loc['Total Debt'].iloc[0]  # Strike
                total_assets = _risk.balance_sheet.loc['Total Assets'].iloc[0]  # Current underlying
            else:
                debt_face_value = _risk.balance_sheet.loc['Current Debt'].iloc[0]  # Strike
                total_assets = _risk.balance_sheet.loc['Current Assets'].iloc[0]  # Current underlying

            total_shares = _risk.balance_sheet.loc['Share Issued'].iloc[0]  # Current count of shares
            _vols = np.array(self.Volatility[symb])

            # Solve the Black-Scholes equation
            d1 = (np.log(total_assets/debt_face_value) + (self.risk_free_rate + 0.5*_vols**2)*maturity)/(_vols*np.sqrt(maturity))
            d2 = d1 - _vols*np.sqrt(maturity)

            Phi1 = norm.cdf(d1)
            Phi2 = norm.cdf(d2)

            implied_equity = total_assets * Phi1 - debt_face_value * np.exp(-self.risk_free_rate*maturity) * Phi2

            self.merton_price[symb] = implied_equity / total_shares

            # Calculate the credit spread
            self.implied_debt[symb] = total_assets - implied_equity
            y = (1/maturity)*np.log(debt_face_value/self.implied_debt[symb])  # The required yield
            self.credit_spread[symb] = y - self.risk_free_rate
            self.default_prob[symb] = 1 - Phi2  # Risk-neutral probability of default at maturity
            # Encapsulate severity model
            self.severity[symb] = self.lognorm_params(
                    mean=self.implied_debt[symb],
                    var=self.Volatility[symb]
                    )

            # Generate the aggregate loss model
            self._calc_aggregate_loss(symb)

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
        vols = []
        spreads = []
        prob_defs = []
        symbs = self.risks.keys()
        risk_prems = []
        upper95 = []
        upper99 = []

        for s in symbs:
            prices.append(self.merton_price[s][index])
            share_prices.append(self.risks[s].Price.iloc[0])
            vols.append(self.Volatility[s][index])
            spreads.append(self.credit_spread[s][index])
            prob_defs.append(self.default_prob[s][index])

            # Attain risk premium information (loss cost estimate)
            risk_prems.append(self.agg_mdls[s].mean(theoretical="Partial"))
            upper95.append(self.agg_mdls[s].ppf(q=0.95))
            upper99.append(self.agg_mdls[s].ppf(q=0.99))

        return pd.DataFrame({
            'Ticker': symbs,
            'Merton_Price': prices,
            'Market_Price': share_prices,
            'Implied_Volatility': vols,
            'Credit_Spread': spreads,
            'Default_Probability': prob_defs,
            'Risk_premium': risk_prems,
            '95%': upper95,
            '99%': upper99
            })

    def save_all(self, folder_path: str = "./") -> None:
        '''
        Save all underlying risk objects into chosen folder.

        Parameters
        ----------
        folder_path: str = "./"
            Specify current directory to store all risk pickles. Defaults to current working directory.
        '''

        for _, _risk in self.risks.items():
            _risk.export(path=f'{folder_path}{_risk.symb}.pkl')
