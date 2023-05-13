import numpy as np
from typing import Tuple
import bisect


class aggregate_distribution:
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
    agg_pdf: accessible discretized aggregate loss pdf

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
            grid: float = 1048576
            ):
        '''
        Initialize the frequency and severity distributions.
        '''
        self.frequency = frequency_distribution
        self.severity = severity_distribution

        self.h = discretization_step
        self.M = grid

        self.severity_dpdf = None
        self.agg_pdf = None
        self.agg_cdf = None
        self.losses = None

        self.diagnostics = None
        self._layer = False

    def get_frequency_mean(self) -> float:
        '''
        Returns the mean of the chosen frequency distribution
        '''
        return self.frequency['dist'].mean(*self.frequency['properties'])

    def get_severity_mean(self) -> float:
        '''
        Returns the mean of the chosen severity distribution
        '''
        return self.severity['dist'].mean(*self.severity['properties'])

    def get_frequency_variance(self) -> float:
        '''
        Returns the variance of the chosen frequency distribution
        '''
        return self.frequency['dist'].var(*self.frequency['properties'])

    def get_severity_variance(self) -> float:
        '''
        Returns the variance of the chosen severity distribution
        '''
        return self.severity['dist'].var(*self.severity['properties'])

    def agg_mean(self,
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
            return np.sum(self.agg_pdf * self.losses)

    def agg_variance(self,
                     theoretical: str = 'True'
                     ):
        '''
        Returns the variance of the aggregate distribution. If `theoretical` is set to true, we return

        E(frequency)*V(severity) + V(frequency)*E(severity)^2

        Otherwise return the approximation
        '''
        if theoretical == 'True':
            return self.get_frequency_mean()*self.get_severity_variance() + self.get_frequency_variance()*(self.get_severity_mean())**2
        elif theoretical == 'Partial':
            severity_mean = np.sum(self.severity_dpdf * self.losses)
            severity_var = np.sum(self.severity_dpdf * self.losses**2) - severity_mean**2
            return self.get_frequency_mean()*severity_var + self.get_frequency_variance()*(severity_mean)**2
        else:
            return np.sum(self.agg_pdf * (self.losses**2)) - self.agg_mean(theoretical=False)**2

    def agg_ppf(self, q: float | Tuple):
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
        indices = [bisect.bisect(self.agg_cdf, qi) for qi in _q]

        # Pass corresponding losses
        return self.losses[indices]

    def get_agg_pdf(self, x: float | Tuple):
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
        return self.agg_pdf[indices]

    def get_agg_cdf(self, x: float | Tuple):
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
        return self.agg_cdf[indices]

    def discretize_pdf(self,
                       X: np.ndarray | None = None
                       ):
        '''
        Discretize a provided severity distribution according to:

        pdf(X) = dist.cdf(X+h/2, dist_params) - dist.cdf(X-h/2, dist_params)

        Parameters
        ----------
        X: np.ndarray [optional],
            Input for the pdf. If None is passed, then we use self.losses and update internal discrete severity pdf

        Returns
        -------
        dpdf: np.ndarray | None
            Discretized pdf corresponding to the input X, if provided, otherwise None
        '''
        dist = self.severity['dist']
        dist_params = self.severity['properties']
        h = self.h

        if X is None:
            dpdf = dist.cdf(self.losses+h/2, *dist_params) - dist.cdf(self.losses-h/2, *dist_params)
            self.severity_dpdf = dpdf
        else:
            dpdf = dist.cdf(X+h/2, *dist_params) - dist.cdf(X-h/2, *dist_params)
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
        dpdf[:min(lh, self.M - 1)] = self.severity_dpdf[range(xh, min(xh + lh, self.M - 1))] / p_xs_survival

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
        Set up aggregate layer modifiers, overwriting the agg_pdf and cdf if `inplace` is set to True.

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
        if self.agg_cdf is None:
            if self.agg_pdf is None:
                self.compile_aggregate_distribution()
            else:
                self._compile_aggregate_cdf()

        # Cut up the aggregate pdf
        M = self.agg_pdf.shape[0]
        alh = int(agg_limit / self.h) if agg_limit is not None else M - 1
        axh = int(agg_excess / self.h)

        p_xs_survival = 1 - np.sum(self.agg_pdf[:axh])

        # Treatment of excess
        dpdf = np.zeros(M)
        dpdf[:min(alh, M - 1)] = self.agg_pdf[range(axh, min(axh + alh, M - 1))] / p_xs_survival

        # Treatment of limit
        dpdf[alh] = 1 - dpdf.sum()

        if inplace:
            self.agg_pdf = dpdf

            # Regenerate the CDF
            self._compile_aggregate_cdf()
        else:
            return dpdf

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
        self.agg_cdf = np.cumsum(self.agg_pdf)

    def _validate_gross(self, theoretical: str = 'True'):
        '''
        Procedure to check that mean and variance of the generated aggregate loss are equal to the theoretical values, within some tolerance.
        '''
        self.diagnostics = {
                'Distribution_total': np.sum(self.agg_pdf),
                'Theoretical_mean': self.agg_mean(theoretical=theoretical),
                'Agg_mean': self.agg_mean(theoretical='False'),
                'Theoretical_var': self.agg_variance(theoretical=theoretical),
                'Agg_var': self.agg_variance(theoretical='False')
                }
