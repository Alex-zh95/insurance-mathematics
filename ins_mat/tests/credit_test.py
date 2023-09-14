# %% [markdown]
# Testing the functionality of Credit_Underwriter class, including visuals.
#
# Note: Yahoo finance only provides options data for US entities.

from ins_mat.special.credit import risk, credit_module

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt


def yf_risk_load(yf_ticker_name: str):
    '''
    Load specified risk tickers from yfinance library.
    '''
    ticker = yf.Ticker(yf_ticker_name)

    # Download 3 years of history
    end_date = dt.datetime.today()
    start_date = dt.datetime(year=end_date.year - 3, month=1, day=1)

    dload = ticker.history(start=start_date, end=end_date)

    # Balance sheet attributes for the risk
    try:
        assets = ticker.balance_sheet.loc['Current Assets'].iloc[0]
        liabilities = ticker.balance_sheet.loc['Current Debt'].iloc[0]
    except KeyError:
        assets = ticker.balance_sheet.loc['Total Assets'].iloc[0]
        liabilities = ticker.balance_sheet.loc['Total Debt'].iloc[0]
    shares_issued = ticker.balance_sheet.loc['Share Issued'].iloc[0]

    # Option prices (optimize for lowest volatility)
    options = pd.DataFrame()

    # Get expiry dates
    expiry_dates = ticker.options

    for T in expiry_dates:
        cur_T_options = ticker.option_chain(T)
        cur_T_options = pd.concat([cur_T_options.calls, cur_T_options.puts], ignore_index=True)
        cur_T_options['expiry'] = T
        options = pd.concat([options, cur_T_options], ignore_index=True)

    # Odd issue that yields wrong expiration dates so add 1 day to correct
    options['expiry'] = pd.to_datetime(options['expiry']) + dt.timedelta(days=1)
    options['duration'] = ((options['expiry']) - end_date).dt.days / 365

    # Labeling call and puts
    options['is_call'] = options['contractSymbol'].str[4:].apply(lambda symb: 'C' in symb)

    # Use the mid-point of price as strike, while also ensuring correct data-type for bid, as, strike
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['price'] = options[['bid', 'ask']].mean(axis=1)

    # Attain the option with the highest trade volume - measure of credibility
    options = options.sort_values(by='openInterest', ascending=False)

    # Obtain the option parameters
    opt_price = options['price'].iloc[0]
    opt_type = 'call' if options['is_call'].iloc[0] else 'put'
    opt_maturity = options['duration'].iloc[0]
    opt_strike = options['strike'].iloc[0]

    yf_risk = risk(name=yf_ticker_name,
                   ticker=yf_ticker_name,
                   sector=ticker.info['sector'],
                   shares_issued=shares_issued,
                   market_history=dload['Close'],
                   option_price=opt_price,
                   option_strike=opt_strike,
                   option_type=opt_type,
                   option_maturity=opt_maturity,
                   currency=ticker.info['currency'],
                   assets=assets,
                   liabilities=liabilities,
                   dividends=dload['Dividends'].iloc[-1]
                   )

    return yf_risk


def credit_risk_generation(yf_risks, limit=10e6, debt_maturity=1, risk_free_rate=0.03):
    '''
    Run the credit risk pipeline from the chosen list of yf_risks.
    '''

    uw = credit_module(yf_risks, limit=limit, debt_maturity=debt_maturity, r=risk_free_rate)
    uw.pipeline_solve()

    return uw


def main_test():
    names = ['SBUX', 'BAC']
    portfolio_lst = [yf_risk_load(s) for s in names]

    uw = credit_risk_generation(portfolio_lst)

    # Print some results:
    ac_probs = [uw.ac_default_probability[s] for s in names]
    sharpe_ratios = [uw.sharpe_ratios[s] for s in names]
    prems = [uw.premiums[s] for s in names]

    results = pd.DataFrame({
        'Risk': names,
        'Default_probs_act': ac_probs,
        'Sharpe': sharpe_ratios,
        'Premiums': prems
        })

    print(results)


if __name__ == "__main__":
    main_test()
