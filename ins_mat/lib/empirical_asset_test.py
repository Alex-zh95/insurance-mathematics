# TODO: More live testing with actual data from Yahoo Finance

import numpy as np
from risk_neutral import implied_asset_volatility, default_probability

# Example data - AMZN equity data
E = np.array([131.83, 132.33, 129.79, 132.55, 131.47, 128.13, 128.40, 125.17])

debt = 140
implied_volatility = 0.35  # For options expiring at least 1 year from now

assets = 462

# Parameters
r = 0.05
# daily_r = (1 + r)**(1. / 365) - 1
daily_r = r

implied_asset_vol = implied_asset_volatility(
    vE=E,
    sig_e=implied_volatility,
    debt_face_value=debt,
    r=daily_r)

print(f'Implied asset volatility = {implied_asset_vol:.3%}')

# Assume that peer group asset drift is as given:
mu_group = r

# We find the probability of default
prob_default = default_probability(
    a0=assets,
    mu_a=mu_group,
    sig_a=implied_asset_vol,
    debt_face_value=debt)

print(f'P(default) = {prob_default:.5%}')
