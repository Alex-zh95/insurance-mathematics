import numpy as np

from scipy.interpolate import interp1d
from insurance_mathematics.agg_dist.fft_poisson import Agg_PoiFft


class MertonJump_CompoundPoisson():
    jump_fr: float                  # Jump frequency (Poisson)
    jump_cf: np.ndarray             # Jump severity characteristic fn
    jump_mean: float                # Jump severity mean
    rn_mdl: list[float]             # Risk-free rate and volatility
    xi: np.ndarray                  # Frequency domain of losses
    pExercise: float | None         # Probability of exercise
    delta: float | None             # Delta of the option
    lr: float                       # Loss ratio
    asset: float                    # Current level of assets

    # Discretization params used by FS
    M: float
    h: float

    def __init__(self,
                 _compPois_mdl: Agg_PoiFft,
                 _rf: float,
                 _sig: float,
                 _lr: float,
                 _asset: float = 1):
        '''
        Merton jump diffusion model with standard geometric brownian motion (GBM) and an asset value jump, 
        specified by a compound Poisson distribution, drive by possibly large/catastrophe losses not sufficiently
        reflected by using just the GBM in isolation.

        We assume we are already in a risk-neutral world.

        Params
        ------
        _compPois_mdl: Agg_PoiFft   Aggregate loss distribution object
        _rf: float                  Risk-free rate
        _sig: float                 Volatility for GBM
        _lr: float                  Ultimate loss ratio
        _assets: float = 1          Asset amount
        '''
        self.rn_mdl = [_rf, _sig]

        self.lr = _lr
        self.asset = _asset
        self.delta = None
        self.pExercise = None

        self.init_jumpMdl(_compPois_mdl)
    
    def init_jumpMdl(self, _compPois_mdl: Agg_PoiFft):
        '''
        Need to conver the loss model to a jump model
        The frequency element stays the same as the occurrence of jumps = loss occurrence.

        Severity becomes an 1 - severity / asset. We assume that asset amount is const
        as is the case in this class. We extract the loss characteristic function
        '''

        # Discretized severity characteristic function (Xi, cf)
        self.M = _compPois_mdl.M
        self.h = _compPois_mdl.h

        # FFT Frequency domain for x but shifted by very small amt to avoid div by 0
        Xi = np.fft.fftfreq(int(self.M), self.h) + 0.001 # Avoid div by 0
        self.xi = Xi

        _compPois_mdl.discretize_pdf()
        severity_cf = np.fft.fft(_compPois_mdl.severity_dpdf)

        # Convert characteristic's severity input to jump input
        Xi_jump = -Xi / self.asset
        cf_jump_real = interp1d(Xi_jump, np.real(severity_cf), kind='linear', fill_value="extrapolate")
        cf_jump_imag = interp1d(Xi_jump, np.imag(severity_cf), kind='linear', fill_value="extrapolate")

        # Recombined complex interp - Severity CF on jump vec
        severity_cf_on_jump = cf_jump_real(np.real(Xi_jump)) + 1j * cf_jump_imag(np.real(Xi_jump))

        # Find jump output
        self.jump_cf = np.exp(1j * Xi) * severity_cf_on_jump

        # Remaining attributes and the discretization grid to inherit
        self.jump_fr = _compPois_mdl.get_frequency_mean()
        self.jump_mean = _compPois_mdl.get_severity_mean()

    def cf(self, xi: np.ndarray, t: float = 1.0) -> np.ndarray:
        '''
        Define the characteristic function of the jump process.

        This is of the form:
        exp(t*(GBM part + jump part))
        '''
        rf, sigE = self.rn_mdl
        lambd = self.jump_fr
        k = self.jump_mean

        # Interpolate jump cf at shifted xi since cf is discretized - separate for real and imag parts
        cf_jump_real = interp1d(self.xi, np.real(self.jump_cf), kind='linear', fill_value="extrapolate")
        cf_jump_imag = interp1d(self.xi, np.imag(self.jump_cf), kind='linear', fill_value="extrapolate")

        # Recombined complex interp
        cf_jump_shifted = cf_jump_real(np.real(xi)) + 1j * cf_jump_imag(np.real(xi))

        gbm_part = 1j * xi * (rf - lambd * k * sigE**2 / 2) - 0.5 * xi**2 * sigE**2
        jump_part = lambd * (cf_jump_shifted - 1)

        return np.exp(t * (gbm_part + jump_part))

    def pi2(self, t: float = 1.0) -> float:
        K = self.lr * self.asset

        def integrand(x):
            return np.exp(-1j * x * np.log(K)) * self.cf(x, t) / (1j * x)

        y = np.real(integrand(self.xi))

        integrated = np.trapezoid(y, self.xi)
        result = np.min([0.5 + 1 / np.pi * integrated, 1.0])
        result = np.max([result, 0.0])

        self.pExercise = result
        return result

    def pi1(self, t: float = 1.0) -> float:
        K = self.lr * self.asset

        def integrand(x):
            return np.exp(-1j * x * np.log(K)) * self.cf(x - 1j, t) / (1j * x * self.cf(-1j, t))

        y = np.real(integrand(self.xi))

        integrated = np.trapezoid(y, self.xi)
        result = 0.5 + 1 / np.pi * integrated

        self.delta = result
        return result

    def price(self, t: float = 1.0) -> float:
        E0 = self.asset
        K = self.lr * E0

        rf = self.rn_mdl[0]
        p1 = self.pi1(t=t) if self.delta is None else self.delta
        p2 = self.pi2(t=t) if self.pExercise is None else self.pExercise
        return np.max([E0 * p1 - K * np.exp(-rf * t) * p2, 0.0])
