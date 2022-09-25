# %%
'''
Filename:       smooth_PeronaMalik.py

About:          Provide a framework for smoothing out small fluctuations in data while also preserving the larger fluctuations.

                We do this by solving a modified Perona-Malik partial differential equation provides a filter that provides the smoothing capabilities of the above.

Source:         https://www.mia.uni-saarland.de/weickert/Papers/book.pdf

Implementation source: https://wire.insiderfinance.io/preserving-edges-when-smoothening-time-series-data-90f9d965132e 
'''

from cProfile import label
import numpy as np 
import pandas as pd
from typing import Tuple

import yfinance as yf
import matplotlib.pyplot as plt 
import seaborn as sns

def get_fd_matrices(n: int) -> Tuple[np.array, np.array]:
    '''
    Obtain first and second derivative matrices, truncated so that only interior points are used for differentiation.

    To apply differentiation, apply Dx @ f for a discretized function f - here @ is the numpy operator for matrix multiplication.

    Inputs:
    ---------------
    n: int
        Total number of discrete points 

    Outputs:
    ---------------
    [Dx, Dxx]: np.ndarray
        First and second derivative matrices, respectively.
    '''

    Dx = (np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1))/2
    Dxx = (np.diag(np.ones(n-1), 1) - 2*np.diag(np.ones(n), 0) + np.diag(np.ones(n-1), -1))

    return Dx[1:-1,:], Dxx[1:-1,:]

def conv_heat_eqn(U: np.ndarray, sigma: float = 1, k: float = 0.05) -> np.ndarray:
    '''
    Perform convolution by solving the heat equation with Neumann boundary conditions.
    
    Inputs:
    ---------------
    U: np.ndarray
        Array to convolve.
    sigma: float, optional
        Standard deviation of Gaussian convolution.
    k : float, optional
        Step-size for finite different scheme

    Output:
    ---------------
    U: np.ndarray
        Convolved function.
    '''

    t = 0
    t_end = sigma**2 / 2

    while t < t_end:
        # Neumann b.c.s
        U[0] = 2*k*U[1] + (1 - 2*k)*U[0]
        U[-1] = 2*k*U[-2] + (1 - 2*k)*U[-1]

        # Scheme on the interior nodes
        U[1:-1] = k*(U[2:] + U[:-2]) + (1-2*k)*U[1:-1]
        t += k

    return U

def perona_malik_smooth(p: np.ndarray, alpha: float=10.0, k: float=0.05, t_end: float = 5.0) -> np.ndarray:
    '''
    Solve the Gaussian convolved Perona-Malik PDE using finite difference scheme.

    Inputs:
    ---------------
    p: np.ndarray
        Array to smooth 
    alpha: float, optional
        The Perona-Malik PDE converges to the heat equation as alpha tends to infinity
    t_end: float, optional
        Stopping point of the algorithm. Increase for stronger smoothness.

    Output:
    ---------------
    U: np.ndarray
        The smoothed time series
    '''

    Dx, Dxx = get_fd_matrices(p.shape[0])

    U = p.copy()
    t = 0

    while t < t_end:
        # Convolve U to allow for well-posedness
        f = conv_heat_eqn(U.copy())

        # Obtain derivatives 
        fx = Dx @ f
        fxx = Dxx @ f

        Ux = Dx @ f
        Uxx = Dxx @ f

        # Substitute into finite difference scheme
        fds = alpha*Uxx / (alpha + fx**2) - 2*alpha*Ux*fx*fxx/(alpha + fx**2)**2 

        # Now solve for next time-step:
        U = np.hstack((
            np.array([p[0]]),
            U[1:-1] + k*fds,
            np.array([p[-1]])
        ))

        t += k

    return U

def main():
    ''' Testing function if library is run '''
    print('Testing smoothing data with AAPL stock prices via YFINANCE')

    df = yf.download('AAPL').reset_index()

    # Truncate to last 100 days
    df = df[-100:]

    v = df['Close'].values

    df['perona_malik1'] = perona_malik_smooth(p=v, t_end=2)
    df['perona_malik2'] = perona_malik_smooth(p=v, t_end=5)
    df['perona_malik3'] = perona_malik_smooth(p=v, t_end=10)

    sns.lineplot(data=df, x='Date', y='Close')
    sns.lineplot(data=df, x='Date', y='perona_malik1', label='Perona malik with t=2')
    sns.lineplot(data=df, x='Date', y='perona_malik2', label='Perona malik with t=5')
    sns.lineplot(data=df, x='Date', y='perona_malik3', label='Perona malik with t=10')
    plt.title("AAPL Stock Price")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

# %%
