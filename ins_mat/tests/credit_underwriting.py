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
        Symb: list = ['TGT', 'BAC', 'AAPL', 'MSFT', 'GOOG', 'TSLA', 'XOM', 'VMW', 'ALL', 'AIG', 'SAP']
        ) -> pd.DataFrame:
    if online_refresh:
        multi_uw = Credit_Underwriter(Symb, risk_free_rate=0.05, maturity=10)
    else:
        multi_uw = Credit_Underwriter.from_file("ins_mat/tests/pkls", risk_free_rate=0.05, maturity=10)

    multi_uw.save_all(folder_path="ins_mat/tests/pkls")

    multi_uw.get_option_data()

    # Accept the suggested volatility estimates (or insert own)
    multi_uw.set_volatility()

    # Solve and generate risk premium
    multi_uw.solve()

    output = multi_uw.present()
    print(output)

    # Convert from loss cost to technical premium
    output['GWP'] = output['Risk_premium'] / (1 - profit_load - comms)

    return output
