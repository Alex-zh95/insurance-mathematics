import numpy as np

from insurance_mathematics.agg_dist.fft_poisson import Agg_PoiFft


class MertonJump_CompoundPoisson():
    jump_mdl: Agg_PoiFft            # Compound Poisson jump model
    rn_mdl: tuple[float, float]     # Risk-free rate and volatility
    xi: np.array                    # Frequency domain of losses
    pExercise: float | None         # Probability of exercise
    delta: float | None             # Delta of the option
    lr: float                       # Loss ratio
    asset: float                    # Current level of assets

    # Discretization params used by FS
    M: int
    h: int

    def __init__(self,
                 _jump_mdl: Agg_PoiFft,
                 _rf: float,
                 _sig: float,
                 _lr: float,
                 _asset: float = 1):
        '''
        Merton jump diffusion model with standard geometric brownian motion (GBM) and a jump, specified by a compound Poisson distribution.

        We assume we are already in a risk-neutral world.

        Params
        ------
        _jump_mdl: Agg_PoiFft       Aggregate loss distribution for jumps
        _rf: float                  Risk-free rate
        _sig: float                 Volatility for GBM
        _lr: float                  ultimate loss ratio
        _assets: float = 1          Asset amount
        '''
        self.jump_mdl = _jump_mdl
        self.rn_mdl = [_rf, _sig]
        self.M = self.jump_mdl.M
        self.h = self.jump_mdl.h

        # FFT Frequency domain for x but shifted by very small amt to avoid div by 0
        self.xi = np.fft.fftfreq(self.jump_mdl.M, self.jump_mdl.h) + 0.001
        self.lr = _lr
        self.asset = _asset
        self.delta = None
        self.pExercise = None

    def cf(self, t: float = 1.0):
        '''
        Define the characteristic function of the jump process.

        This is of the form:
        exp(t*(GBM part + jump part))
        '''
        rf, sigE = self.rn_mdl
        lambd = self.jump_mdl.get_frequency_mean()
        k = self.jump_mdl.get_severity_mean()
        xi = self.xi

        gbm_part = 1j * xi * (rf - lambd * k * sigE**2 / 2) - 0.5 * xi**2 * sigE**2
        jump_part = lambd * (self.jump_mdl.cf - 1)

        return np.exp(t * (gbm_part + jump_part))

    def _shifted_cf(self, t: float = 1.0):
        rf, sigE = self.rn_mdl
        lambd = self.jump_mdl.get_frequency_mean()
        k = self.jump_mdl.get_severity_mean()
        xi_shifted = self.xi - 1j

        gbm_part = 1j * xi_shifted * (rf - lambd * k * sigE**2 / 2) - 0.5 * xi_shifted**2 * sigE**2
        jump_part = lambd * (self.jump_mdl.cf - 1)

        return np.exp(t * (gbm_part + jump_part))

    def _icf(self, t: float = 1.0):
        rf, sigE = self.rn_mdl
        lambd = self.jump_mdl.get_frequency_mean()
        k = self.jump_mdl.get_severity_mean()

        gbm_part = 1j * -1j * (rf - lambd * k * sigE**2 / 2) - 0.5 * (-1j)**2 * sigE**2
        jump_part = lambd * (self.jump_mdl.cf - 1)

        return np.exp(t * (gbm_part + jump_part))

    def pi2(self, t: float = 1.0) -> float:
        K = self.lr * self.asset

        def integrand(x):
            return np.exp(-1j * x * np.log(K)) * self.cf(t) / (1j * x)

        y = np.real(integrand(self.xi))

        integrated = np.trapezoid(y, self.xi)
        result = np.min([0.5 + 1 / np.pi * integrated, 1.0])
        result = np.max([result, 0.0])

        self.pExercise = result
        return result

    def pi1(self, t: float = 1.0) -> float:
        K = self.lr * self.asset

        def integrand(x):
            return np.exp(-1j * x * np.log(K)) * self._shifted_cf(t) / (1j * x * self._icf(t))

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
        return E0 * p1 - K * np.exp(-rf * t) * p2
