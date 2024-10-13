from __future__ import annotations

import numpy as np
from typing import Tuple
import bisect


class AggregateDistribution:
    '''
    Base class for aggregate distributions.

    We define the frequency, severity elements. Possible losses are also to be included as a grid depending on the underlying algorithm.

    Both distributions to be dicts taking the form

    {
        dist: any of scipy.stats,
        properties: statistical parameters required (in dict form)
    }

    See scipy docs for the required properties

    Parameters
    ----------
    frequency_distribution: as described above
    severity_distribution: as described above

    Properties
    ----------
    losses: vector of possible losses (scaled, if needed)
    pdf: accessible discretized aggregate loss pdf

    Callables
    ---------
    Some basic stats can be called for the underlying frequency and severity distributions. These are mostly calls to their scipy.stats functions. Refer to scipy manual for more details.

    Call compile_aggregate_distribution() to generate the aggregate distribution.
    '''

    def __init__(
            self,
            frequency_distribution: dict,
            severity_distribution: dict,
            discretization_step: float = 0.01,
            grid: float = 1048576):
        '''
        Initialize the frequency and severity distributions.
        '''
        self.frequency = frequency_distribution
        self.severity = severity_distribution

        self.h = discretization_step
        self.M = grid

        self.severity_dpdf = None
        self._pdf = None
        self._cdf = None
        self.losses = None

        self.cf = None  # Characteristic function (store the FFT-vector for other purposes)

        self.diagnostics = None
        self._layer = False

    def get_frequency_mean(self) -> float:
        '''
        Returns the mean of the chosen frequency distribution
        '''
        if self.frequency['dist'] is not None:
            return self.frequency['dist'].mean(*self.frequency['properties'])
        else:
            return self.frequency['properties'][0]

    def get_severity_mean(self) -> float:
        '''
        Returns the mean of the chosen severity distribution
        '''
        if self.severity['dist'] is not None:
            return self.severity['dist'].mean(*self.severity['properties'])
        else:
            return self.severity['properties'][0]

    def get_frequency_variance(self) -> float:
        '''
        Returns the variance of the chosen frequency distribution
        '''
        if self.frequency['dist'] is not None:
            return self.frequency['dist'].var(*self.frequency['properties'])
        else:
            return self.frequency['properties'][1]

    def get_severity_variance(self) -> float:
        '''
        Returns the variance of the chosen severity distribution
        '''
        if self.severity['dist'] is not None:
            return self.severity['dist'].var(*self.severity['properties'])
        else:
            return self.severity['properties'][1]

    def mean(self,
             theoretical: str = 'True'
             ):
        '''
        Returns the mean of the aggregate distribution. If `theoretical` is set to true, we return

        E(severity)*E(frequency)

        Set `theoretical` to partial to use the discretized pdf only. This may be useful for looking at modified severity PDFs.

        Otherwise return the approximation
        '''
        if theoretical == 'True':
            return self.get_severity_mean() * self.get_frequency_mean()
        elif theoretical == 'Partial':
            return np.sum(self.severity_dpdf * self.losses) * self.get_frequency_mean()
        else:
            return np.sum(self._pdf * self.losses)

    def var(self,
            theoretical: str = 'True'
            ):
        '''
        Returns the variance of the aggregate distribution. If `theoretical` is set to true, we return

        E(frequency)*V(severity) + V(frequency)*E(severity)^2

        Otherwise return the approximation
        '''
        if theoretical == 'True':
            return self.get_frequency_mean() * self.get_severity_variance() + self.get_frequency_variance() * (self.get_severity_mean())**2
        elif theoretical == 'Partial':
            severity_mean = np.sum(self.severity_dpdf * self.losses)
            severity_var = np.sum(self.severity_dpdf * self.losses**2) - severity_mean**2
            return self.get_frequency_mean() * severity_var + self.get_frequency_variance() * (severity_mean)**2
        else:
            return np.sum(self._pdf * (self.losses**2)) - self.mean(theoretical='False')**2

    def ppf(self, q: float | Tuple):
        '''
        Return the percentage point of aggregate. We can only use the discretized severity for this.

        Parameters
        ----------
        q: float | Tuple
            Percentile (between 0 and 1)

        Returns
        -------
        result: np.ndarray
            Corresponding percentage point on aggregate distribution
        '''
        _q = np.array([q]) if isinstance(q, float) else np.array(q)

        # Obtain the relevant index of the _q, including interpolation if needed
        indices = [bisect.bisect(self._cdf, qi) for qi in _q]

        # Pass corresponding losses
        return self.losses[indices]

    def pdf(self, x: float | Tuple):
        '''
        Return the pdf of aggregate. We can only use the discretized severity for this.

        Parameters
        ----------
        x: float
            Loss amount (must be in range of loss vector)

        Returns
        -------
        result: np.ndarray
            Corresponding pdf on aggregate distribution
        '''
        _x = np.array([x]) if isinstance(x, float) else np.array(x)

        assert ((x >= self.losses.min()).all()) & ((x <= self.losses.max()).all()), "loss x out of scope"

        # Obtain the relevant index of the _x, including interpolation if needed
        indices = [bisect.bisect(self.losses, xi) for xi in _x]

        # Pass corresponding pdf output
        return self._pdf[indices]

    def cdf(self, x: float | Tuple):
        '''
        Return the cdf of aggregate. We can only use the discretized severity for this.

        Parameters
        ----------
        x: float
            Loss amount (must be in range of loss vector)

        Returns
        -------
        result: np.ndarray
            Corresponding cdf on aggregate distribution
        '''
        _x = np.array([x]) if isinstance(x, float) else np.array(x)

        assert ((x >= self.losses.min()).all()) & ((x <= self.losses.max()).all()), "loss x out of scope"

        # Obtain the relevant index of the _x, including interpolation if needed
        indices = [bisect.bisect(self.losses, xi) for xi in _x]

        # Pass corresponding pdf output
        return self._cdf[indices]

    def discretize_pdf(self,
                       _dist=None,
                       _dist_params: list | None = None,
                       X: np.ndarray | None = None,
                       h_step: float | None = None):
        '''
        Discretize a provided severity distribution according to:

        pdf(X) = dist.cdf(X+h/2, dist_params) - dist.cdf(X-h/2, dist_params)

        Parameters
        ----------
        _dist: scipy-like obj [optional]
            Custom distribution for discretization (note it must implement at minimum the .cdf(...) method), if left blank, we use self.severity
        _dist_params: list [optional]
            Optional parameters for the passed in custom distribution in _dist
        X: np.ndarray [optional],
            Input for the pdf. If None is passed, then we use self.losses and update internal discrete severity pdf
        h_step: float [optional]
            Custom discretization step if desired. This is mainly useful for if we want to discretize a custom probability function

        Returns
        -------
        dpdf: np.ndarray | None
            Discretized pdf corresponding to the input X, if provided, otherwise None
        '''
        if (_dist is None) & (h_step is None):
            dist = self.severity['dist']
            dist_params = self.severity['properties']
            h = self.h
        else:
            dist = _dist
            dist_params = _dist_params
            h = h_step

        if X is None:
            dpdf = dist.cdf(self.losses + h / 2, *dist_params) - dist.cdf(self.losses - h / 2, *dist_params)

            # Check validity of the output pdf
            if np.abs(np.sum(dpdf) - 1) > h:
                Warning(f'Possibly invalid discretization produced! SUM(DPDF) = {np.sum(dpdf)}')

            self.severity_dpdf = dpdf
        else:
            dpdf = dist.cdf(X + h / 2, *dist_params) - dist.cdf(X - h / 2, *dist_params)

            # Check validity of the output pdf
            if np.abs(np.sum(dpdf) - 1) > h:
                Warning(f'Possibly invalid discretization produced! SUM(DPDF) = {np.sum(dpdf)}')

            return dpdf

    def setup_layer(self,
                    excess: float,
                    limit: float | None,
                    inplace: bool = True):
        '''
        Set up a suitable insurance structure with each-and-every excess and limits.

        Layer designed by adjusting probability of incurring losses after excess and limit.

        Parameters
        ----------
        excess: float,
            Excess retained (by insured)
        limit: float
            Limit, above which losses are ceded. Use None for no limit.
        inplace: bool [Optional]
            If set, we overwrite the current discretized severity PDF, otherwise, return the discretized PDF as a vector, which can be inserted into another object

        Returns
        -------
        dpdf: np.ndarray | None
            Discretized severity PDF after layer modification (only when inplace is not set to True)
        '''

        self._layer = True

        # Losses: we expect the severity to already be discretized at this stage so lazy it
        if self.severity_dpdf is None:
            self.discretize_pdf()

        # Discretize the limit and excess points
        lh = int(limit / self.h) if limit is not None else self.M - 1
        xh = int(excess / self.h)

        p_xs_survival = 1 - np.sum(self.severity_dpdf[:xh])

        # Treatment of excess
        dpdf = np.zeros(self.M)
        dpdf[:min(lh, self.M - 1 - xh)] = self.severity_dpdf[range(xh, min(xh + lh, self.M - 1 - xh))] / p_xs_survival

        # Treatment of limit
        dpdf[lh] = 1 - dpdf.sum()

        if inplace:
            self.severity_dpdf = dpdf

            # Also override the frequency
            self.thin_frequency(p_xs_survival)
        else:
            return dpdf

    def setup_agg_layer(self,
                        agg_excess: float,
                        agg_limit: float | None,
                        inplace: bool = True):
        '''
        Set up aggregate layer modifiers, overwriting the aggregate pdf and cdf if `inplace` is set to True.

        Parameters
        ----------
        agg_excess: float,
            Aggregate excess retained (by insured)
        agg_limit: float | None
            Aggregate limit, above which losses are ceded. None implies infinite limit.
        inplace: bool [Optional]
            If set, we overwrite the current discretized aggregate PDF, otherwise, return the discretized PDF as a vector, which can be inserted into another object

        Returns
        -------
        dpdf: np.ndarray | None
            Discretized aggregate PDF after layer modification (only when inplace is not set to True)
        '''

        # Function should only be available after compilation
        if self._cdf is None:
            if self._pdf is None:
                self.compile_aggregate_distribution()
            else:
                self._compile_aggregate_cdf()

        # Cut up the aggregate pdf
        M = self._pdf.shape[0]
        alh = int(agg_limit / self.h) if agg_limit is not None else M - 1
        axh = int(agg_excess / self.h)

        p_xs_survival = 1 - np.sum(self._pdf[:axh])

        # Treatment of excess
        dpdf = np.zeros(M)
        dpdf[:min(alh, M - 1 - axh)] = self._pdf[range(axh, min(axh + alh, M - 1 - axh))] / p_xs_survival

        # Treatment of limit
        dpdf[alh] = 1 - dpdf.sum()

        if inplace:
            self._pdf = dpdf

            # Regenerate the CDF
            self._compile_aggregate_cdf()
        else:
            return dpdf

    def add(self, other: AggregateDistribution) -> AggregateDistribution:
        '''
        Add two aggregate distributions together. We assume independence of distributions and so the result is the multiplication of the characteristic functions.

        We initialize the result in another aggregate distribution class, calculate the pdf and cdf also.

        Some undefined elements may still pull through, e.g. layer information, underlying frequency or severity distributions.
        '''
        if ((self.M != other.M) | (self.h != other.h)):
            raise AssertionError('Grid resolution must match!')

        # Create the new object
        result = AggregateDistribution(
            frequency_distribution={
                'dist': None,
                'properties': [
                    self.get_frequency_mean() + other.get_frequency_mean(),
                    self.get_frequency_variance() + other.get_frequency_variance(),
                ]
            },
            severity_distribution={
                'dist': None,
                'properties': [
                    self.get_severity_mean() + other.get_severity_mean(),
                    self.get_severity_variance() + other.get_severity_variance(),
                ]
            },
            discretization_step=self.h,
            grid=self.M
        )

        result.cf = self.cf * other.cf
        result._pdf = np.real(np.fft.ifft(self.cf))
        result.losses = self.losses

        return result

    def appr_finite_ruin_probability(self, init_income: float, premium: float) -> float:
        '''
        Approximate finite ruin probability given initial income and premium rate as defined. This calculation is exact if a compound Poisson distribution is used.

        Parameters
        ----------
        init_income: float
            Initial capital
        premium: float
            Premium income

        Returns
        -------
        Finite ruin probability: float
        '''
        loss_ratio = self.mean('False') / premium

        if loss_ratio >= 1:
            return 1.0

        return loss_ratio * (1 - self.cdf(init_income + premium))

    # ABSTRACT FUNCTIONS

    def compile_aggregate_distribution(self):
        '''
        Classes to inherit and implement this function!
        '''
        raise NotImplementedError

    def thin_frequency(self, n: float):
        '''
        Redefine the frequency distribution as a result of thinning.

        Theorem: E(Thin frequency) = n*E(frequency)

        The variance and definition will depend on the underyling frequency itself so this will need custom implementation!
        '''
        raise NotImplementedError

    # INTERNAL FUNCTIONS

    def _compile_aggregate_cdf(self):
        self._cdf = np.cumsum(self._pdf)

    def _validate_gross(self, theoretical: str = 'True'):
        '''
        Procedure to check that mean and variance of the generated aggregate loss are equal to the theoretical values, within some tolerance.
        '''
        self.diagnostics = {
            'Distribution_total': np.sum(self._pdf),
            'Theoretical_mean': self.mean(theoretical=theoretical),
            'Agg_mean': self.mean(theoretical='False'),
            'Theoretical_var': self.var(theoretical=theoretical),
            'Agg_var': self.var(theoretical='False')
        }

        # Check validity of the output pdf
        if np.abs(self.diagnostics['Distribution_total'] - 1) > self.h:
            Warning(f'Possibly invalid final PDF produced! SUM(AGGPDF) = {self.diagnostics["Distribution_total"]}')
