"""
Filename:   loss_distribution.py
Author:     Alex-zh

Date:       2020-04-21

Calculates a compound Poisson loss distribution using given selected inputs and severity distributions, outputting information into a JSON file.

Due to the number of input parameters required, will import the necessary parameters using JSON.
"""

import numpy as np
import pandas as pd
from scipy import stats
import bisect

import argparse, json

# Get the input arguments to the json file containing the insurance parameters
parser = argparse.ArgumentParser(description="Generate an aggregate loss distribution estimate using FFT.")

parser.add_argument('-i', type=str, help="Path to insurance structure JSON.")
parser.add_argument('-o', type=str, help="Output file name", default=None)
parser.add_argument('-t', type=str, help="Export file type (JSON, CSV)", default="JSON")
args = parser.parse_args()

json_file = args.i
out_file = args.o
out_type = args.t

# Load the json file
with open(json_file) as in_file:
    insurance_structure = json.load(in_file)['insurance_structure']

# Discretization parameters
M = 2**18           # Grid size 
h = 0.02            # Grid step size

# Exposure and probability of claim - frequency parameters
N = insurance_structure['Exposure']
pr = insurance_structure['Claim_unit_probability']

# Calculate the count
fLambda = N*pr

# Insurance parameters
Limit = insurance_structure['Limit_of_liability'] # Limit of liability - needs to be a multiple of h
Excess = insurance_structure['Excess'] # Also needs to be a multiple of h

Limit_h = int(Limit/h)
Excess_h = int(Excess/h)

# Severity parameters
s_params = [insurance_structure['Severity_param0'], insurance_structure['Severity_param1'], insurance_structure['Severity_param2']]
s_dist = insurance_structure['Severity_distribution']

## Discretize the severity distribution
def discretize_pdf(X, h, severity_distribution, params):
    if severity_distribution == "lognorm":
        pdf = stats.lognorm.cdf(X+h/2, s=params[0], scale=params[1]) - stats.lognorm.cdf(X-h/2, s=params[0], scale=params[1])
    elif severity_distribution == "gamma":
        pdf = stats.gamma.cdf(X+h/2, a=params[0], scale=1/params[1]) - stats.gamma.cdf(X-h/2, a=params[0], scale=1/params[1])
    elif severity_distribution == "pareto":
        pdf = stats.pareto.cdf(X+h/2, b=params[0], scale=params[1]) - stats.pareto.cdf(X-h/2, b=params[0], scale=params[1])
    elif severity_distribution == "gpd":
        pdf = stats.genpareto.cdf(X+h/2, c=params[0], loc=params[1], scale=params[2]) - stats.genpareto.cdf(X-h/2, c=params[0], loc=params[1], scale=params[2])
    return pdf

def discretize_loss(k,h):
    return k*h

x = np.linspace(h, M*h, M)
x_pdf = discretize_pdf(x,h, severity_distribution=s_dist, params=s_params)

# Retained distribution models below excess losses
x_ret_pdf = np.zeros(M)
x_ret_pdf[:] = x_pdf[:] 

# Obtain survival of excess
pAboveExcess = 1 - np.sum(x_pdf[0:Excess_h])
x_ret_pdf[Excess_h] = pAboveExcess

x_ced_pdf = np.zeros(M) # Ceded distribution models above excess but below the limit

# Losses above the excess have probability of 0 for retained distribution
x_ret_pdf[Excess_h+1:] = 0

# Ceded distribution contains losses given they are above excess, hence probabilities need to be scaled
x_ced_pdf[0:Limit_h] = np.array([x_pdf[Excess_h+k]/pAboveExcess for k in range(Limit_h)])

pAboveLimit = 1 - sum(x_ced_pdf)
x_ced_pdf[Limit_h] = pAboveLimit

## Calculate the aggregated loss distribution with the FFT

# Transform to Fourier domain
FT_x_pdf = np.fft.fft(x_pdf)
FT_x_ret_pdf = np.fft.fft(x_ret_pdf)
FT_x_ced_pdf = np.fft.fft(x_ced_pdf)

# Calculated the transformed aggregate loss distribution
FT_s_pdf = np.exp(fLambda*(FT_x_pdf-1))
FT_s_ret_pdf = np.exp(fLambda*(FT_x_ret_pdf-1))

# Ceded frequency = Gross frequency * above deductible survival
c_fLambda = fLambda * pAboveExcess
FT_s_ced_pdf = np.exp(c_fLambda*(FT_x_ced_pdf-1))

# Aggregate loss distribution 
s_pdf = np.real(np.fft.ifft(FT_s_pdf))
s_ret_pdf = np.real(np.fft.ifft(FT_s_ret_pdf))
s_ced_pdf = np.real(np.fft.ifft(FT_s_ced_pdf))

## Obtain the results for output

# Mean losses
al_gross_mean = np.sum(s_pdf*x)
al_ret_mean = np.sum(s_ret_pdf*x)
al_ced_mean = np.sum(s_ced_pdf*x)

# Generate the standard deviations
al_gross_std = np.sqrt( np.sum(s_pdf*x**2) - al_gross_mean**2 )
al_ret_std = np.sqrt( np.sum(s_ret_pdf*x**2) - al_ret_mean**2 )
al_ced_std = np.sqrt( np.sum(s_ced_pdf*x**2) - al_ced_mean**2 )

# Summarize the mean and standard deviations into one table
al_stats = pd.DataFrame({
    'Percentiles': ('Mean', 'Std'),
    'Gross Losses': (al_gross_mean, al_gross_std),
    'Retained Losses': (al_ret_mean, al_ret_std),
    'Ceded Losses': (al_ced_mean, al_ced_std)
})

# Generate the aggregate loss cdf
s_cdf = np.cumsum(s_pdf)
s_ret_cdf = np.cumsum(s_ret_pdf)
s_ced_cdf = np.cumsum(s_ced_pdf)

percentiles = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99, 0.999]

# Each vector of indices will have the same length by design (i.e. length of percentiles) hence can be contained in one loop
gross_loss = []
ret_loss = []
ced_loss = []

for p in percentiles:
    gross_index = min(bisect.bisect(s_cdf, p),M-1)
    gross_loss.append(x[gross_index])

    ret_index = min(bisect.bisect(s_ret_cdf, p), M-1)
    ret_loss.append(x[ret_index])

    ced_index = min(bisect.bisect(s_ced_cdf, p), M-1)
    ced_loss.append(x[ced_index])

agg_loss_tab = pd.DataFrame({
    'Percentiles': percentiles,
    'Gross Losses': gross_loss,
    'Retained Losses': ret_loss,
    'Ceded Losses': ced_loss
})

# Print parameters
print("Aggregate Loss Costing -- FFT based approach")
print("Excess\t:", Excess)
print("Limit\t:", format(Limit, ".0f"))
print("Frequency\t:", format(fLambda, ".3f"))
print("Exposure\t:", N)

print("Chosen distribution\t:", s_dist)
print("Parameters\t:", format(s_params[0], ".3f"), ",", format(s_params[1], ".3f"), ",", format(s_params[2], ".3f"))

print("Gross mean\t:", format(al_gross_mean, ",.3f"))
print("Retained mean\t:", format(al_ret_mean, ",.3f"))
print("Ceded mean\t:", format(al_ced_mean, ",.3f"))

# Concatenate the two data frames
frames = [agg_loss_tab, al_stats]
total_tab = pd.concat(frames, ignore_index=True)
print(total_tab)

# Want to summarize the above results into a JSON file
if out_file is not None:
    if out_type == "CSV":
        total_tab.to_csv(out_file)
    else:
        total_tab.to_json(out_file, orient="records")
