# Insurance Mathematics

This repository showcases scripts and tools used for pricing insurance contracts. Areas of consideration include loss modeling following a frequency/severity approach as well as distributional considerations for reporting delay (IBNR). 

# Scripts
## 1. Triangle-free IBNR estimation
This particular script calculates IBNR by modeling reporting delays as an exponential distribution, which is often used to model delays and also builds on the assumption of claims frequency following a Poisson process. Given the bias of short-tailed claims reported in recent periods, we attempt to make a correction to the mean delay using Bayesian techniques. The IBNR is then calculated following methods described in [1].

Usage: Call

python ibnr_triangle_free.py -i <POLICY.JSON> -p <1> -o <OUTPUT.JSON>

where the -i flag represents the policy information (could be at portfolio level), the -p flag toggles whether to show plots of the unbiasing function (omit this to not show plots and the -o flag represents the output, which is also in JSON format. This flag can be omitted if no output file is desired.

## 2. Frequency-Severity Modelling 
This script builds an aggregate loss distribution using the Poisson frequency assumption and a selection of possible severity distributions such as lognormal, gamma and generalized Pareto. The output is a simple 3-layer insurance tower, all of which can be specified by the input JSON file. The method used to generate is the Fourier transform, derived by first principles (verification found text in [2]).

This script is perhaps best suited for reinsurance or other excess of loss policies where parameters have been chosen to be scaled by millions (the example JSON file uses this approach). Otherwise, the grid parameters may need modification so that discretization is correct. This error will be flagged and the script will terminate.

Usage: Call

python frequency_severity.py -i <INSURANCE_STRUCTURE.JSON> -o <OUTPUT.JSON>

# Sources

[1] "Triangle-free reserving: a non-traditional framework for estimating reserves and reserve uncertainty" by Pietro Parodi, where a distributional approach was taken for evaluating IBNR

[2] "Pricing in General Insurance" by Pietro Parodi, which contains the theory of how Fourier transforms may be incorporated in producing a aggregate loss distribution when considering a frequency/severity approach to pricing