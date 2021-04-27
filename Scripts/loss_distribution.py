"""
Filename:   loss_distribution.py
Author:     Alex-zh

Date:       2020-04-21

Calculates a compound Poisson loss distribution using given selected inputs and severity distributions, outputting information into a JSON file.

Due to the number of input parameters required, will import the necessary parameters using JSON.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import multiprocessing
from joblib import Parallel, delayed

import argparse, json

# Get the input arguments to the json file containing the insurance parameters
parser = argparse.ArgumentParser(description="Generate an aggregate loss distribution estimate using FFT.")

parser.add_argument('-i', type=str, help="Path to insurance structure JSON.")
parser.add_argument('-o', type=str, help="Output file name (JSON)", default="Output.JSON")
args = parser.parse_args()

json_file = args.i
out_file = args.o

# Load the json file
with open(json_file) as in_file:
    insurance_structure = json.load(in_file)['insurance_structure']

# Parallel processing
u_cores = int(multiprocessing.cpu_count())

# Discretization parameters
M = 65536*4           # Grid size 
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
def discretize_pdf(k, h, severity_distribution, params):
    if severity_distribution == "lognorm":
        pdf = stats.lognorm.cdf(k*h+h/2, s=params[0], scale=params[1]) - stats.lognorm.cdf(k*h-h/2, s=params[0], scale=params[1])
    elif severity_distribution == "gamma":
        pdf = stats.gamma.cdf(k*h+h/2, a=params[0], scale=1/params[1]) - stats.gamma.cdf(k*h-h/2, a=params[0], scale=1/params[1])
    elif severity_distribution == "pareto":
        pdf = stats.pareto.cdf(k*h+h/2, b=params[0], scale=params[1]) - stats.pareto.cdf(k*h-h/2, b=params[0], scale=params[1])
    elif severity_distribution == "gpd":
        pdf = stats.genpareto.cdf(k*h+h/2, c=params[0], loc=params[1], scale=params[2]) - stats.genpareto.cdf(k*h-h/2, c=params[0], loc=params[1], scale=params[2])
    return pdf

def discretize_loss(k,h):
    return k*h

x_pdf = Parallel(n_jobs=u_cores)(delayed(discretize_pdf)(k, h, s_dist, s_params) for k in range(M))
#x = Parallel(n_jobs=u_cores)(delayed(discretize_loss)(k,h) for k in range(M))
x = np.linspace(10, M*h, M)

x_ret_pdf = np.zeros(M) # Retained distribution models below excess losses
x_ced_pdf = np.zeros(M) # Ceded distribution models above excess but below the limit

# Obtain survival of excess
pBelowExcess = 0
for k in range(Excess_h):
    pBelowExcess += x_pdf[k]
pAboveExcess = 1 - pBelowExcess

# Losses above the excess have probability of 0 for retained distribution
for k in range(M):
    x_ret_pdf[k] = x_pdf[k]
    if k == Excess_h:
        x_ret_pdf[k] = pAboveExcess
        break # All other values are 0 by default

# Ceded distribution contains losses given they are above excess, hence probabilities need to be scaled
for k in range(M):
    if k < Limit_h:
        x_ced_pdf[k] = x_pdf[Excess_h+k] / pAboveExcess
    else:
        break

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
    'Layer': ('Gross', 'Retained', 'Ceded'),
    'Mean': (al_gross_mean, al_ret_mean, al_ced_mean),
    'Std': (al_gross_std, al_ret_std, al_ced_std)
})

# Generate the aggregate loss cdf
s_cdf = np.cumsum(s_pdf)
s_ret_cdf = np.cumsum(s_ret_pdf)
s_ced_cdf = np.cumsum(s_ced_pdf)

# Obtain the quantile descriptors
def find_index(value, array):
    N = len(array)
    cur_ind = 0
    temp_min = 1e6
    for i in range(1,N):
        temp_diff = np.abs(array[i]-value)
        if temp_min >= temp_diff:
            cur_ind = i
            temp_min = temp_diff
    return cur_ind

percentiles = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99, 0.999]
gross_indices = []
ret_indices = []
ced_indices = []
for p in percentiles:
    g_index = find_index(p, s_cdf)
    gross_indices.append(g_index)

    r_index = find_index(p, s_ret_cdf)
    ret_indices.append(r_index)

    c_index = find_index(p, s_ced_cdf)
    ced_indices.append(c_index)

gross_loss = []
for i in gross_indices:
    gross_loss.append(x[i])

ret_loss = []
for i in ret_indices:
    ret_loss.append(x[i])

ced_loss = []
for i in ced_indices:
    ced_loss.append(x[i])

agg_loss_tab = pd.DataFrame({
    'Percentiles': percentiles,
    'Gross Losses': gross_loss,
    'Retained Losses': ret_loss,
    'Ceded Losses': ced_loss
})

print("Aggregate Loss Costing -- FFT based approach")
print("Excess\t:", Excess)
print("Limit\t:", format(Limit, ".0f"))
print("Frequency\t:", format(pr, ".3f"))
print("Exposure\t:", N)

print("Chosen distribution\t:", s_dist)
print("Parameters\t:", format(s_params[0], ".3f"), ",", format(s_params[1], ".3f"), ",", format(s_params[2], ".3f"))

print("Gross mean\t:", format(al_gross_mean, ",.3f"))
print("Retained mean\t:", format(al_ret_mean, ",.3f"))
print("Ceded mean\t:", format(al_ced_mean, ",.3f"))

# Apply formatting for display purposes
disp_tab = agg_loss_tab.to_string(formatters={
    'Percentiles': '{:,.3f}'.format,
    'Gross Losses': '{:,.3f}'.format,
    'Retained Losses': '{:,.3f}'.format,
    'Ceded Losses': '{:,.3f}'.format
})
print(disp_tab)

# Want to summarize the above results into a JSON file
with open(out_file, "w") as out_f:
    # The JSON file will contain 2 layers: 1 for the aggregate loss table and then the summary statistics
    result = {'Aggregate_table': agg_loss_tab, 'Summary_aggregate': al_stats}
    json.dump(result, out_f)
    