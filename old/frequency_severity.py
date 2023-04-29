"""
Filename:   frequency_severity.py
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
    jsonDict = json.load(in_file)
    insurance_structure = jsonDict['insurance_structure']
    grid_params = jsonDict['Grid']

# Discretization parameters
M = grid_params['Grid_width']
h = grid_params['Grid_step']

# Claim frequency
fLambda = insurance_structure['Frequency']

# Insurance layer parameters
Limits = np.array(insurance_structure['Limits'])
Limits_h = (Limits / h).astype(int) # Conversion to steps

# Severity parameters
s_params = [insurance_structure['Severity_param0'], insurance_structure['Severity_param1'], insurance_structure['Severity_param2']]
s_dist = insurance_structure['Severity_distribution']

# Outputs for percentiles 
percentiles = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99, 0.995, 0.999]

## Function for getting discretized pdf from given distribution
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

# Build up the PDF of the gross severity distribution
x = np.linspace(h, M*h, M)
x_pdf = discretize_pdf(x,h, severity_distribution=s_dist, params=s_params)

# Run a check to ensure that x_pdf sums close to 1 (add slight tolerance, defined by step size h, given inaccurancies will be caused by this even on correct generation)
assert (x_pdf.sum() >= 1 - h) and (x_pdf.sum() <= 1 + h), 'Incorrect PDF generated, consider revising grid parameters in JSON file'

# Probability of exceeding each limit
pExceedLims = [1-np.sum(x_pdf[:Lh]) for Lh in Limits_h]

# Store all results from following for-loop into the following results matrix
result = np.zeros(shape=(len(percentiles)+4, Limits_h.shape[0]))

for k in range(len(Limits_h)):
    if k == 0:
        # For the first item, generate the gross layer
        x_layer_pdf = x_pdf.copy()
        c_fLambda = fLambda
    else:
        x_layer_pdf = np.zeros(M)

        # Apply the attachment/excess
        x_layer_pdf[:Limits_h[k]] = np.array([x_pdf[Limits_h[k-1]+l]/pExceedLims[k-1] for l in range(Limits_h[k])])

        # Apply the upper limit
        x_layer_pdf[Limits_h[k]] = 1 - x_layer_pdf.sum()

        # Adapt the claim frequency for the higher excess
        c_fLambda = fLambda * pExceedLims[k-1]

    # Apply Fourier transform
    hatX_layer_pdf = np.fft.fft(x_layer_pdf)

    # Obtain transformed aggregate loss distribution
    hatS_layer_pdf = np.exp(c_fLambda * (hatX_layer_pdf - 1))

    # Obtain aggregate loss distribution by inverting Fourier transform
    s_layer_pdf = np.real(np.fft.ifft(hatS_layer_pdf)) # note: throw away imaginary part artefact

    # Calculate usual statistics: mean, standard deviation
    layer_mean = np.sum(s_layer_pdf * x)
    result[-4, k] = layer_mean

    layer_sd = np.sqrt(np.sum(s_layer_pdf * x**2) - layer_mean**2)
    result[-3, k] = layer_sd

    # Obtain percentiles
    s_layer_cdf = np.cumsum(s_layer_pdf)

    for i in range(len(percentiles)):
        # Obtain relevant percentile index (including interpolation if needed)
        ind = min(bisect.bisect(s_layer_cdf, percentiles[i]), M-1)

        # Save the percentile
        result[i, k] = x[ind]

    # For results, insert the layer information 
    result[-2, k] = 0 if k == 0 else Limits[k-1]
    result[-1, k] = np.Inf if k == 0 else Limits[k]

# Summarize the mean and standard deviations into one table
colNames = [f'Layer {i:.0f} Losses' if i > 0 else 'Gross Losses' for i in range(len(Limits))]
indNames = [f'{p:.1%}' for p in percentiles]
indNames.append('Mean')
indNames.append('Sd')
indNames.append('Lower Limit')
indNames.append('Upper Limit')
results_table = pd.DataFrame(data=result, columns=colNames, index=indNames)

# Print parameters
print('Aggregate Loss Costing -- FFT based approach')
print(f'Chosen distribution\t: {s_dist}')
print(f'Frequency\t\t: {fLambda:.3f}')
print(f'Parameters\t\t: {s_params[0]:,.3f}, {s_params[1]:,.3f}, {s_params[2]:,.3f}') 

print(results_table)

# Want to summarize the above results into a JSON file
if out_file is not None:
    if out_type == "CSV":
        results_table.to_csv(out_file)
    else:
        results_table.to_json(out_file, orient="records")
else:
    print('No output file created. If this is not desired, rerun and issue a file path via the -o flag.')
