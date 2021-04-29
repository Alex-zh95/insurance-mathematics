"""
Filename:   ibnr_triangle_free.py
Author:     Alex-zh

Date:       2020-05-02

Using an assumed observed exponential distribution for reporting delays, calculate IBNR (Incurred But Not Reported) on either volume or number of claims.

The observed exponential distribution is likely to be biased as we can never observe the delays that are longer than observed, trivially - hence unbiasing will be done with Bayesian techniques.

Source: Parodi, Pietro: "Triangle-free reserving: a non-traditional framework for estimating reserves and reserve uncertainty.", Date: 2013-02-04, Presented to the Institute and Faculty of Actuaries.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import newton # Secant method of root finding
import json
import argparse

parser = argparse.ArgumentParser(description="Calculate IBNR on the assumption that reporting delays behave according to an exponential distribution")
parser.add_argument('-i', type=str, help="Path to file containing policy information JSON")
parser.add_argument('-p', type=str, help="Toggle whether plots are to be shown - default is blank for no plots")
parser.add_argument('-o', type=str, help="Path to output file for results saving.")

args = parser.parse_args()
json_file = args.i
out_file = args.o
plotOn = args.p

# Import
with open(json_file) as file:
    policy = json.load(file)['policy_data']

# To unbias the observed parameter, find the roots of the following function
def bias_to_debias_difference(t_True, t_Obs, a):
    return t_True * (1 + (np.exp(-a/t_True) - t_True/a * (1 - np.exp(-a/t_True)) / (1 - (t_True/a) * (1 - np.exp(-a/t_True))))) - t_Obs

# OBSERVED PARAMETERS - Time parameters in years
tObserved = policy['Average_reporting_delay']
upper = policy['Policy_length']

print("Observed delay:\t\t\t", format(tObserved, ".3f"))
print("Policy history (years):\t\t", format(upper, ".3f"))
print("Observed claims count:\t\t", format(policy["Total_claims"], ".3f"))

# Method requires an initial guess - as we prognose the true value to be slightly above tObserved, we add a small amount to tObserved
t0 = tObserved + 0.01

# Plot the Bayesian unbiasing function to see visually where we have roots (Newton and secant methods do not unconditionally converge)
if plotOn is not None:
    x = np.linspace(0,upper,500)
    y = bias_to_debias_difference(x, tObserved, upper)
    plt.plot(x,y,color='red')
    plt.grid()
    plt.title("Delay bias correction function")
    plt.xlabel("Delay, t")
    plt.ylabel("Bias")
    plt.show()

tTrue = newton(bias_to_debias_difference, t0, args=[tObserved, upper])
print("Calculated true average delay:\t", format(tTrue, ".3f"))

# Projecting the claims upwards to the true upper limit (e.g. grossing up), we can use the tTrue parameter, assuming that the unbiased delay distribution is still exponential
def project(reported_tUpper, project_tUpper, param, reported_count):
    return reported_count * (project_tUpper / (project_tUpper - param * ( np.exp((project_tUpper - reported_tUpper)/param) - np.exp(-reported_tUpper/param))))

reported = 4 
out = project(policy['Reporting_years'], policy['Policy_length'], tTrue, policy['Total_claims'])
print("Uplifted:\t\t\t", format(out, ".3f"))
print("IBNR:\t\t\t\t", format(out - policy['Total_claims'], ".3f"))

ibnr_standard_error = np.sqrt(out - policy['Total_claims'])
print("Standard error:\t\t\t", format(ibnr_standard_error, ".3f"))

# Write to output if not empty
if out_file is not None:
    # Open the file output as specified by user, store in "f"
    with open(out_file, 'w') as f:
        # Create the dataframe for export
        result = {
            "Policy years": policy['Policy_length'],
			"Reported years": policy['Reporting_years'],
            "Observed frequency": policy['Total_claims'],
            "Average observed delay": tObserved,
            "Unbiased delay estimate": tTrue,
            "Frequency with IBNR": out,
            "Standard error": ibnr_standard_error
        }

        # Dump the result
        json.dump(result, f)