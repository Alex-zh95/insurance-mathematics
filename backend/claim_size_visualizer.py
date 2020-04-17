"""
Filename:   claim_size_visualizer.py
Author:     Alex-zh

Date:       2020-04-15

From a list of claims, provide plots of the empirical distribution to give insight/intuition to the claims seen. Standard distributions are suggested with visualizations to observe goodness-of-fits.

Input file expects the following columns ['AD_Total', 'TPPD_Total', 'TPPI_Total'] 
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.optimize import minimize

# Parallel processing
from joblib import Parallel, delayed
import multiprocessing

file = "./Fakedata/INDIVIDUAL_CLAIMS.csv"
decimal_encoding = "."
df = pd.read_csv(file, decimal=decimal_encoding)

# Require valid data (i.e. no non-positives)
df_new = df[df>0]
if (len(df_new) < len(df)):
    print("Invalid data (non-positive values) have been found and excluded from analysis.")
df = df_new
del df_new

# Plot histograms
fig, ax = plt.subplots(1,3, figsize=(15,5))
sns.distplot(df['AD_Total'], kde=False, ax=ax[0])
sns.distplot(df['TPPI_Total'], kde=False, ax=ax[1])
sns.distplot(df['TPPD_Total'], kde=False, ax=ax[2])
fig.show()

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
def gamma_mle(df, column):
    alpha_start = df[column].mean()**2 / df[column].var()
    beta_start = df[column].mean()**2 / df[column].var()
    theta_start = [alpha_start, beta_start]

    ad_output = minimize(gnlogl, x0=theta_start, args=(df[column]), method="Nelder-Mead")
    return ad_output

# Function for fitting lognormal distribution
def lognorm_mle(df, column):
    mu_start = np.mean(np.log(df[column].values))
    sigma_start = np.std(np.log(df[column].values))
    theta_start = [mu_start, sigma_start]

    ad_output = minimize(lognorm_logl, x0=theta_start, args=(df[column]), method="Nelder-Mead")
    return ad_output

print("\n")
print("Fitting the Gamma distribution to the AD data")
ad_gfit = gamma_mle(df, 'AD_Total')
ad_gParams = [ad_gfit.x[0], 0, ad_gfit.x[1]]
print("Gamma shape\t:", format(ad_gParams[0], ".3f"))
print("Gamma scale\t:", format(ad_gParams[2], ".3f"))

# Evaluation with the KS test
ad_gRes = stats.kstest(df['AD_Total'], 'gamma', args=ad_gParams)
print("KS Statistic\t:", format(ad_gRes.statistic, ".3f"))
print("KS p-value\t:", format(ad_gRes.pvalue, ".3f"))

print("Fitting the Lognormal distribution to the AD data")
ad_lnfit = lognorm_mle(df, 'AD_Total')
ad_lnParams = [ad_lnfit.x[0], 1, ad_lnfit.x[1]]
print("Lognormal mu\t:", format(ad_lnParams[0], ".3f"))
print("Lognormal sigma\t:", format(ad_lnParams[2], ".3f"))

# Evaluation with the KS test
ad_lnRes = stats.kstest(df['AD_Total'], 'lognorm', args=ad_lnParams)
print("KS Statistic\t:", format(ad_lnRes.statistic, ".3f"))
print("KS p-value\t:", format(ad_lnRes.pvalue, ".3f"))

# Suggest the better fit by observing the distribution with the higher p-value
if ad_lnRes.pvalue > ad_gRes.pvalue:
    print("Recommended distribution: lognormal")
    ad_dist = "lognormal"
elif ad_lnRes.pvalue < ad_gRes.pvalue:
    print("Recommended distribution: gamma")
    ad_dist = "gamma"
else:
    print("Both distributions provide a reasonable fit.")

# TPPD Data
print("\n")
print("Fitting the Gamma distribution to the TPPD data")
tppd_gfit = gamma_mle(df, 'TPPD_Total')
tppd_gParams = [tppd_gfit.x[0], 1, 1/tppd_gfit.x[1]] # We have fitted rate, Python lib uses scale = 1/rate
print("Gamma shape\t:", format(tppd_gParams[0], ".3f"))
print("Gamma scale\t:", format(tppd_gParams[2], ".3f"))

# Evaluation with the KS test
tppd_gRes = stats.kstest(df['TPPD_Total'], 'gamma', args=tppd_gParams)
print("KS Statistic\t:", format(tppd_gRes.statistic, ".3f"))
print("KS p-value\t:", format(tppd_gRes.pvalue, ".3f"))

print("Fitting the Lognormal distribution to the TPPD data")
tppd_lnfit = lognorm_mle(df, 'TPPD_Total')
tppd_lnParams = [tppd_lnfit.x[0], 0, tppd_lnfit.x[1]]
print("Lognormal mu\t:", format(tppd_lnParams[0], ".3f"))
print("Lognormal sigma\t:", format(tppd_lnParams[2], ".3f"))

# Evaluation with the KS test
tppd_lnRes = stats.kstest(df['TPPD_Total'], 'lognorm', args=tppd_lnParams)
print("KS Statistic\t:", format(tppd_lnRes.statistic, ".3f"))
print("KS p-value\t:", format(tppd_lnRes.pvalue, ".3f"))

# Suggest the better fit by observing the distribution with the higher p-value
if tppd_lnRes.pvalue > tppd_gRes.pvalue:
    print("Recommended distribution: lognormal")
    tppd_dist = "lognormal"
elif tppd_lnRes.pvalue < tppd_gRes.pvalue:
    print("Recommended distribution: gamma")
    tppd_dist = "gamma"
else:
    print("Both distributions provide a reasonable fit.")

# TPPI Data
print("\n")
print("Fitting the Gamma distribution to the TPPI data")
tppi_gfit = gamma_mle(df, 'TPPI_Total')
tppi_gParams = [tppi_gfit.x[0], 0, tppi_gfit.x[1]]
print("Gamma shape\t:", format(tppi_gParams[0], ".3f"))
print("Gamma scale\t:", format(tppi_gParams[2], ".3f"))

# Evaluation with the KS test
tppi_gRes = stats.kstest(df['TPPI_Total'], 'gamma', args=tppi_gParams)
print("KS Statistic\t:", format(tppi_gRes.statistic, ".3f"))
print("KS p-value\t:", format(tppi_gRes.pvalue, ".3f"))

print("Fitting the Lognormal distribution to the TPPI data")
tppi_lnfit = lognorm_mle(df, 'TPPI_Total')
tppi_lnParams = [tppi_lnfit.x[0], 1, tppi_lnfit.x[1]]
print("Lognormal mu\t:", format(tppi_lnParams[0], ".3f"))
print("Lognormal sigma\t:", format(tppi_lnParams[2], ".3f"))

# Evaluation with the KS test
tppi_lnRes = stats.kstest(df['TPPI_Total'], 'lognorm', args=tppi_lnParams)
print("KS Statistic\t:", format(tppi_lnRes.statistic, ".3f"))
print("KS p-value\t:", format(tppi_lnRes.pvalue, ".3f"))

# Suggest the better fit by observing the distribution with the higher p-value
if tppi_lnRes.pvalue > tppi_gRes.pvalue:
    print("Recommended distribution: lognormal")
    tppi_dist = "lognormal"
elif tppi_lnRes.pvalue < tppi_gRes.pvalue:
    print("Recommended distribution: gamma")
    tppi_dist = "gamma"
else:
    print("Neither provides a better fit")

# Plot histograms (densities) with the selected fitting distributions
fig, ax = plt.subplots(1,3, figsize=(15,5))
p = np.linspace(0.001, 0.999, 999)

ax[0].hist(df['AD_Total'], density=True, label="Empirical", color="lightblue")
x = np.linspace(df['AD_Total'].min(), df['AD_Total'].max(), 1000)
pg = stats.gamma.pdf(x, ad_gParams[0], ad_gParams[1], ad_gParams[2])
pln = stats.lognorm.pdf(x, ad_lnParams[0], ad_lnParams[1], ad_lnParams[2])
ax[0].plot(x, pg, color="red", label="Gamma")
ax[0].plot(x, pln, color="green", label="Lognormal")
ax[0].legend()
ax[0].set_title("AD Claims Distribution")
ax[0].set_xlabel("Claim Severity")
ax[0].set_ylabel("p")

ax[1].hist(df['TPPI_Total'], density=True, label="Empirical", color="lightblue")
x = np.linspace(df['TPPI_Total'].min(), df['TPPI_Total'].max(), 1000)
pg = stats.gamma.pdf(x, tppi_gParams[0], tppi_gParams[1], tppi_gParams[2])
pln = stats.lognorm.pdf(x, tppi_lnParams[0], tppi_lnParams[1], tppi_lnParams[2])
ax[1].plot(x, pg, color="red", label="Gamma")
ax[1].plot(x, pln, color="green", label="Lognormal")
ax[1].legend()
ax[1].set_title("TPPI Claims Distribution")
ax[1].set_xlabel("Claim Severity")
ax[1].set_ylabel("p")

ax[2].hist(df['TPPD_Total'], density=True, label="Empirical", color="lightblue")
x = np.linspace(df['TPPD_Total'].min(), df['TPPD_Total'].max(), 1000)
pg = stats.gamma.pdf(x, tppd_gParams[0], tppd_gParams[1], tppd_gParams[2])
pln = stats.lognorm.pdf(x, tppd_lnParams[0], tppd_lnParams[1], tppd_lnParams[2])
ax[2].plot(x, pg, color="red", label="Gamma")
ax[2].plot(x, pln, color="green", label="Lognormal")
ax[2].legend()
ax[2].set_title("TPPD Claims Distribution")
ax[2].set_xlabel("Claim Severity")
ax[2].set_ylabel("p")
fig.show()