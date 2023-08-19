'''
Wrapper for yfinance ticker objects.
'''

import numpy as np
import yfinance as yf
import pandas as pd
import pickle

from datetime import datetime


class risk(yf.Ticker):
    '''
    `risk` inherits from the yfinance package and extend for storage
    '''

    def __init__(self,
                 symb: str,
                 window: int = 3,
                 online_load: bool = True
                 ) -> None:
        '''
        Inializes an instance of risk and download the relevant stock information, ready for use.

        Parameters
        ----------
        symb: str
            Stock ticker symbol to access (must be one relevant to the stock API library used).
        stock_lib: str, default = "yfinance"
            Underlying API for accessing stock data.
        window: int, default = 3
            Number of years of stock data to download - note the start date always at 1 Jan.
        online_load: bool, default = True
            Load stock information from online resource
        '''
        self.symb = symb
        self.window = window

        super().__init__(symb)

        # Download the underlying stock price history
        if online_load:
            self.refresh_data()

    @classmethod
    def from_file(cls, source_file: str):
        '''
        Alternative constructor that loads stock and share data from a pickle file.

        Parameters
        ----------
        source_file: str
            Path to file containing risk information.
        '''
        with open(source_file, 'rb') as file:
            data = pickle.load(file)

        obj = cls(symb=data['symb'], window=data['window'], online_load=False)

        obj.Price = data['Price']
        obj.Dividend = data['Dividend']
        obj.ccy = data['ccy']

        obj.calculate_returns()

        return obj

    def refresh_data(self, end: datetime = datetime.today()) -> None:
        '''
        Download data via the chosen stock API (see self.stock_lib)

        Parameters
        ----------
        end: datetime, default: datetime.today()
            Cut-off point of data download.

        Updated properties
        ------------------
        Price: np.ndarray
            Underlying daily close stock prices
        Dividend: np.ndarray
            Underlying daily dividend payments
        ccy: str
            Underlying currency
        '''
        # Get the relevant start date based on the provided time window
        start_date = datetime(year=end.year - self.window, month=1, day=1)
        data = self.history(start=start_date, end=end)

        self.Price = data['Close']
        self.Dividend = data['Dividends']
        self.ccy = self.info['currency']

        # Calculate the returns information
        self.calculate_returns()

    def calculate_returns(self) -> None:
        '''
        Calculate the YoY percentage change on stock value.

        Updated properties
        ------------------
        Returns: np.ndarray
            YoY percentage on stock price
        '''
        self.Returns = self.Price.values[1:] / self.Price.values[:-1]

    def describe(self) -> pd.DataFrame:
        '''
        Calculate some basic statistics and return these as a pandas dataframe. Statistics include:

        - Count of ticker points
        - Mean stock price, returns in period
        - Max stock price, returns in period
        - Min stock price, returns in period observed
        - Standard deviation of stock, returns price observed
        '''
        log_returns = np.log(self.Returns.T)
        log_mean, log_sd = np.mean(log_returns), np.std(log_returns)

        return pd.DataFrame({
            'Count': [self.Price.shape[0], self.Returns.shape[0]],
            'Mean': [self.Price.mean(), np.exp(log_mean)],
            'Max': [self.Price.max(), self.Returns.max()],
            'Min': [self.Price.min(), self.Returns.min()],
            'Standard deviation': [np.std(self.Price), np.exp(log_sd)]
            }, index=['Stock_price', 'Stock_returns']).T

    def export(self, path: str) -> None:
        '''
        Export the class attributes and data

        Parameters
        ----------
        path: str
            Path to output.
        '''
        att = self.__dict__.copy()

        # Remove the keys that begin with underscores (mainly modules that cannot be pickled)
        for key in list(att.keys()):
            if key.startswith('_'):
                att.pop(key)

        with open(path, 'wb') as file:
            pickle.dump(att, file)
