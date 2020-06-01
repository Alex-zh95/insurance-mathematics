"""
Filename:   claim_size_visualizer.py
Author:     Alex-zh

Date:       2020-04-15

From a list of claims, provide plots of the empirical distribution to give insight/intuition to the claims seen.

Input file expects the following columns ['AD_Total', 'TPPD_Total', 'TPPI_Total'] 
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats
from scipy.optimize import minimize

# Parallel processing
from joblib import Parallel, delayed
import multiprocessing

# Impoort argument parser
import argparse

# Commandline parsing init
parser = argparse.ArgumentParser(description="Visualize and suggest distribution that best fits the claim severity")
parser.add_argument('-r', type=str, help="Path to file containing claims file")
parser.add_argument('-d', type=str, help="Decimal separator used in files, default is '.'", default=".")

# Enable limits of indemnity to remove outliers with 0 default meaning no limit
parser.add_argument('--adl', type=int, help="AD severity limit", default=0)
parser.add_argument('--tppil', type=int, help="TPPI severity limit", default=0)
parser.add_argument('--tppdl', type=int, help="TPPD severity limit", default=0)

args = parser.parse_args()

file = args.r
decimal_encoding = args.d
df = pd.read_csv(file, decimal=decimal_encoding)

def clip(x, level):
    """
    Clips data at the maximum level, by setting all values >= level as 0
    """
    if x >= level:
        return 0
    else:
        return x

# Set limits when needed - set invalid entries to 0, so they are removed
u_cores = int(multiprocessing.cpu_count() / 2)
if args.adl > 0:
    temp = df['AD_Total'].values
    t_vect = Parallel(n_jobs=u_cores)(delayed(clip)(t, args.adl) for t in temp)
    df['AD_Total'] = t_vect

if args.tppil > 0:
    temp = df['TPPI_Total'].values
    t_vect = Parallel(n_jobs=u_cores)(delayed(clip)(t, args.tppil) for t in temp)
    df['TPPI_Total'] = t_vect
    
if args.tppdl > 0:
    temp = df['TPPD_Total'].values
    t_vect = Parallel(n_jobs=u_cores)(delayed(clip)(t, args.tppdl) for t in temp)
    df['TPPD_Total'] = t_vect

# Require valid data (i.e. no non-positives)
df_new = df[df>0]
if (len(df_new) < len(df)):
    print("Invalid data (non-positive values) have been found and excluded from analysis.")
df = df_new
del df_new

# Set up negative log-likelihood functions for the usual distributions: Gamma and Lognormal

def gnlogl(theta, x):
    """
    Negative of the loglikelihood function for Gamma distribution.
    """
    # Get the number of available processors - will utilize half
    u_cores = int(multiprocessing.cpu_count() / 2)

    alpha = theta[0]
    beta = theta[1]
    s_vect = Parallel(n_jobs=u_cores)(delayed(stats.gamma.logpdf)(xi, a=alpha, loc=0, scale=1/beta) for xi in x)
    return -np.sum(s_vect)

def lognorm_logl(theta, x):
    """
    Negative of the loglikelihood function for Lognormal distribution
    """
    # Get the number of available processors - will utilize half
    u_cores = int(multiprocessing.cpu_count() / 2)

    s = theta[0]
    scale = theta[1]
    s_vect = Parallel(n_jobs=u_cores)(delayed(stats.lognorm.logpdf)(xi, s=s, loc=0, scale=scale) for xi in x)
    return -np.sum(s_vect)

# Function for fitting the Gamma distribution
def gamma_mle(series):
    alpha_start = series.mean()**2 / series.var()
    beta_start = series.mean()**2 / series.var()
    theta_start = [alpha_start, beta_start]

    ad_output = minimize(gnlogl, x0=theta_start, args=(series), method="Nelder-Mead")
    return ad_output

# Function for fitting lognormal distribution
def lognorm_mle(series):
    mu_start = np.mean(np.log(series.values))
    sigma_start = np.std(np.log(series.values))
    theta_start = [mu_start, sigma_start]

    ad_output = minimize(lognorm_logl, x0=theta_start, args=(series), method="Nelder-Mead")
    return ad_output

print("\n")
ad_df = df['AD_Total'].dropna()

print("Fitting the Gamma distribution to the AD data")
ad_gfit = gamma_mle(ad_df)
ad_gParams = [ad_gfit.x[0], 0, ad_gfit.x[1]]
print("Gamma shape\t:", format(ad_gParams[0], ".3f"))
print("Gamma scale\t:", format(ad_gParams[2], ".3f"))

print("Fitting the Lognormal distribution to the AD data")
ad_lnfit = lognorm_mle(ad_df)
ad_lnParams = [ad_lnfit.x[0], 0, ad_lnfit.x[1]]
print("Lognormal mu\t:", format(ad_lnParams[0], ".3f"))
print("Lognormal sigma\t:", format(ad_lnParams[2], ".3f"))

# TPPD Data
print("\n")
tppd_df = df['TPPD_Total'].dropna()

print("Fitting the Gamma distribution to the TPPD data")
tppd_gfit = gamma_mle(tppd_df)
tppd_gParams = [tppd_gfit.x[0], 0, 1/tppd_gfit.x[1]] # We have fitted rate, Python lib uses scale = 1/rate
print("Gamma shape\t:", format(tppd_gParams[0], ".3f"))
print("Gamma scale\t:", format(tppd_gParams[2], ".3f"))

print("Fitting the Lognormal distribution to the TPPD data")
tppd_lnfit = lognorm_mle(tppd_df)
tppd_lnParams = [tppd_lnfit.x[0], 0, tppd_lnfit.x[1]]
print("Lognormal mu\t:", format(tppd_lnParams[0], ".3f"))
print("Lognormal sigma\t:", format(tppd_lnParams[2], ".3f"))

# TPPI Data
tppi_df = df['TPPI_Total'].dropna()
print("\n")
print("Fitting the Gamma distribution to the TPPI data")
tppi_gfit = gamma_mle(tppi_df)
tppi_gParams = [tppi_gfit.x[0], 0, tppi_gfit.x[1]]
print("Gamma shape\t:", format(tppi_gParams[0], ".3f"))
print("Gamma scale\t:", format(tppi_gParams[2], ".3f"))

print("Fitting the Lognormal distribution to the TPPI data")
tppi_lnfit = lognorm_mle(tppi_df)
tppi_lnParams = [tppi_lnfit.x[0], 0, tppi_lnfit.x[1]]
print("Lognormal mu\t:", format(tppi_lnParams[0], ".3f"))
print("Lognormal sigma\t:", format(tppi_lnParams[2], ".3f"))

# Plot histograms (densities) with the selected fitting distributions
fig, ax = plt.subplots(1,3, figsize=(15,5))
p = np.linspace(0.001, 0.999, 999)

ax[0].hist(ad_df, density=True, label="Empirical", color="lightblue")
x = np.linspace(ad_df.min(), ad_df.max(), 1000)
pg = stats.gamma.pdf(x, ad_gParams[0], ad_gParams[1], ad_gParams[2])
pln = stats.lognorm.pdf(x, ad_lnParams[0], ad_lnParams[1], ad_lnParams[2])
ax[0].plot(x, pg, color="red", label="Gamma")
ax[0].plot(x, pln, color="green", label="Lognormal")
ax[0].legend()
ax[0].set_title("AD Claims Distribution")
ax[0].set_xlabel("Claim Severity")
ax[0].set_ylabel("p")

ax[1].hist(tppi_df, density=True, label="Empirical", color="lightblue")
x = np.linspace(tppi_df.min(), tppd_df.max(), 1000)
pg = stats.gamma.pdf(x, tppi_gParams[0], tppi_gParams[1], tppi_gParams[2])
pln = stats.lognorm.pdf(x, tppi_lnParams[0], tppi_lnParams[1], tppi_lnParams[2])
ax[1].plot(x, pg, color="red", label="Gamma")
ax[1].plot(x, pln, color="green", label="Lognormal")
ax[1].legend()
ax[1].set_title("TPPI Claims Distribution")
ax[1].set_xlabel("Claim Severity")
ax[1].set_ylabel("p")

ax[2].hist(tppd_df, density=True, label="Empirical", color="lightblue")
x = np.linspace(tppd_df.min(), tppd_df.max(), 1000)
pg = stats.gamma.pdf(x, tppd_gParams[0], tppd_gParams[1], tppd_gParams[2])
pln = stats.lognorm.pdf(x, tppd_lnParams[0], tppd_lnParams[1], tppd_lnParams[2])
ax[2].plot(x, pg, color="red", label="Gamma")
ax[2].plot(x, pln, color="green", label="Lognormal")
ax[2].legend()
ax[2].set_title("TPPD Claims Distribution")
ax[2].set_xlabel("Claim Severity")
ax[2].set_ylabel("p")
fig.show()