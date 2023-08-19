# %% [markdown]
# Testing the functionality of Credit_Underwriter class, including visuals.
#
# Note: Yahoo finance only provides options data for US entities.

from ins_mat.special.underwriter import Credit_Underwriter
import pandas as pd


def test(
        online_refresh: bool = True,  # Toggle whether to reload risk data
        profit_load: float = 0.4,   # General profit load (typical for reinsurance)
        comms: float = 0.35,  # Commission load (typical brokerage commission)
        Symb: list = ['TGT', 'BAC', 'MCD', 'SBUX', 'AAPL', 'MSFT', 'GOOG', 'TSLA']
        ) -> pd.DataFrame:
    multi_uw = Credit_Underwriter(Symb, risk_free_rate=0.05)
    multi_uw.get_option_data()

    # Accept the suggested volatility estimates (or insert own)
    multi_uw.set_volatility()

    # Solve and generate risk premium
    multi_uw.solve(maturity=10)

    output = multi_uw.present()
    print(output)

    # Convert from loss cost to technical premium
    prems = output[['Ticker', 'Risk_Premium']]
    prems['GWP'] = prems['Risk_Premium'] / (1 - profit_load - comms)

    print(f"Total Premium = {prems['GWP'].sum():,.0f}")

    multi_uw.save_all(folder_path="tests/pkls/*")

    return output
