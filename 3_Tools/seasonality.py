#%%
'''
Filename:           seasonality.py

About:              Functions as a workbook to demonstrate the usage of the Fourier transform within Python's scipy library for removing noise from time series. Especially, if the noise introduces some level of sub-seasonality.
'''

import numpy as np 
from scipy import stats, fft
from typing import Tuple 

import matplotlib.pyplot as plt 
import seaborn as sns 

# Generate some data - and provide some "background info"
print("Generating time data - each data point represents 1 day")
t = np.linspace(0, 364, 365)
noise = stats.norm.rvs(loc=0, scale=8.8, size=len(t))
T = 2*np.sin(t*np.pi/180)**2 + np.cos((t+noise)*np.pi/180)

# Visuaize original plot
sns.lineplot(x=t, y=T, label="Original")
plt.title("Original time series - daily values")
plt.grid('on')
plt.show()

# %% Smoothing
# Obtain smoothing
T_hat = fft.rfft(T)
xi = fft.rfftfreq(n=t.shape[0], d=1/365)

sns.lineplot(x=xi, y=T_hat)
plt.title("Fourier-transformed time series")
plt.grid('on')
plt.show()

# Basic smoothing by setting higher-xi values of T_hat to zero
T2_hat = T_hat.copy()
T2_hat[5:] = 0

T2 = fft.irfft(T2_hat)

sns.lineplot(x=t, y=T, label="Original")
sns.lineplot(x=t[1:], y=T2, label="Smoothed")
plt.title("Original vs smoothed series - daily values")
plt.grid('on')
plt.show()

# %%
