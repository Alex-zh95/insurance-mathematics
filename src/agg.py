import numpy as np
# from scipy import stats
# from typing import Tuple


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
    '''

    def __init__(
            self,
            frequency_distribution: dict,
            severity_distribution: dict
            ):
        '''
        Initialize the frequency and severity distributions.
        '''
        self.frequency = frequency_distribution
        self.severity = severity_distribution

        self.agg_pdf = 0
        self.losses = 0

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
        return self.frequency['dist'].variance(*self.frequency['properties'])

    def get_severity_variance(self) -> float:
        '''
        Returns the variance of the chosen severity distribution
        '''
        return self.severity['dist'].variance(*self.severity['properties'])

    def agg_mean(self,
                 theoretical: bool = True
                 ):
        '''
        Returns the mean of the aggregate distribution. If `theoretical` is set to true, we return

        E(severity)*E(frequency)

        Otherwise return the approximation
        '''
        if theoretical:
            return self.get_severity_mean() * self.get_frequency_mean()
        else:
            return self.agg_pdf * self.losses

    def discretize_pdf(self,
                       X: np.ndarray,
                       h: float,
                       ):
        '''
        Discretize a provided severity distribution according to:

        pdf(X) = dist.cdf(X+h/2, dist_params) - dist.cdf(X-h/2, dist_params)

        Parameters
        ----------
        X: np.ndarray,
            Input for the pdf
        h: float,
            Discretization step

        Returns
        -------
        pdf:
            Discretized pdf corresponding to the inputs X
        '''
        dist = self.severity['dist']
        dist_params = self.severity['properties']

        self.severity_dpdf = dist.cdf(X+h/2, *dist_params) - dist.cdf(X-h/2, *dist_params)

    def generate_aggregate_distribution(self):
        raise NotImplementedError
