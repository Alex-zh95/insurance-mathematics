# %% [markdown]
# Testing the functionality of Credit_Underwriter class, including visuals.
#
# Note: Yahoo finance only provides options data for US entities.
from context import access_root_dir
access_root_dir(1)

from ins_mat.special.credit import risk, credit_module

import pandas as pd
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
    assets = ticker.balance_sheet.loc['Total Assets'].iloc[0]
    long_liabs = ticker.balance_sheet.loc['Total Debt'].iloc[0]
    shares_issued = ticker.balance_sheet.loc['Share Issued'].iloc[0]

    # Typical trigger is 100% of short + 50% of long-term liabs payable
    try:
        short_liabs = ticker.balance_sheet.loc['Current Debt'].iloc[0]
        alpha = 0.5
    except KeyError:
        # When short term liabs not available, we take 75% of long-term instead
        short_liabs = 0
        alpha = 0.75

    liabilities = short_liabs + alpha*long_liabs  # Typical trigger (KMV)

    # Option prices (optimize for lowest volatility)
    options = pd.DataFrame()

    # Get expiry dates
    expiry_dates = ticker.options

    # Filter for expiry dates not immediate, so we look at least 1 month from now
    # exp_date_filter = end_date + dt.timedelta(weeks=52/12)

    for T in expiry_dates:
        # if dt.datetime.strptime(T, '%Y-%m-%d') < exp_date_filter:
            # continue

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

    # Attain the option details with the highest trade volume - measure of credibility
    options = options.sort_values(by='openInterest', ascending=False)

    # Obtain the option parameters
    opt_price = options['price'].iloc[0]
    opt_type = 'call' if options['is_call'].iloc[0] else 'put'
    opt_maturity = options['duration'].iloc[0]
    opt_strike = options['strike'].iloc[0]

    # Obtain mean implied volatility
    opt_implied = options['impliedVolatility'].iloc[0]

    yf_risk = risk(name=yf_ticker_name,
                   ticker=yf_ticker_name,
                   sector=ticker.info['sector'],
                   shares_issued=shares_issued,
                   market_history=dload['Close'].values,
                   option_price=opt_price,
                   option_strike=opt_strike,
                   option_type=opt_type,
                   option_maturity=opt_maturity,
                   currency=ticker.info['currency'],
                   assets=assets,
                   liabilities=liabilities,
                   dividends=dload['Dividends'].iloc[0]
                   )

    return (yf_risk, opt_implied)


def credit_risk_generation(yf_risks, limit=100e6, debt_maturity=1., risk_free_rate=0.03, impl_vol_overriders=None):
    '''
    Run the credit risk pipeline from the chosen list of yf_risks.
    '''

    uw = credit_module(yf_risks, limit=limit, debt_maturity=debt_maturity, r=risk_free_rate)

    for i in range(len(yf_risks)):
        rsk = yf_risks[i]
        ovr = impl_vol_overriders[i]

        print(f'Solving for {rsk.name}')
        print('Attaining implied asset volatility...')
        uw.calculate_implied_volatility(rsk, override=ovr)

        print('Attaining sharpe ratio')
        uw.calculate_sharpe_ratio(rsk)

        print('Generating rates\n')
        uw.generate_rate(rsk, use_rn=False)

    return uw


def main_test(names: list[str], limit: float = 100e6):
    portfolio_lst = []
    list_implied_vol = []

    for s in names:
        rsk, implV = yf_risk_load(s)
        portfolio_lst.append(rsk)
        list_implied_vol.append(implV)

    uw = credit_risk_generation(portfolio_lst, limit=limit, debt_maturity=1.0, risk_free_rate=0.03, impl_vol_overriders=list_implied_vol)

    # Print some results:
    ac_probs = [uw.ac_default_probability[s] for s in names]
    rn_probs = [uw.rn_default_probability[s] for s in names]
    e_vols = [uw.equity_volatilities[s] for s in names]
    a_vols = [uw.asset_volatilities[s] for s in names]
    sharpe_ratios = [uw.sharpe_ratios[s] for s in names]
    prems = [uw.premiums[s] for s in names]
    mkt_prices = [uw.dict_risks[s].market_history[0] for s in names]

    results = pd.DataFrame({
        'Risk': names,
        'Implied_equity_volatility': e_vols,
        'Implied_asset_volatility': a_vols,
        'Default_probs_rn': rn_probs,
        'Default_probs_act': ac_probs,
        'Sharpe': sharpe_ratios,
        'Premiums': prems,
        'Market_price': mkt_prices,
        })

    print("Result table:\n")
    print(results)


if __name__ == "__main__":
    in_str = input('Ticker strings (separate by commas):')
    names = in_str.split(', ')
    main_test(names)
