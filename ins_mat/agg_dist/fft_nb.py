import numpy as np

from scipy.stats import nbinom
from ins_mat.agg_dist.agg_base import aggregate_distribution


class nb_fft_agg(aggregate_distribution):
    '''
    Define and build an aggregate distribution using a Negative Binomial NB frequency via Fourier transform.

    Inherits: aggregate_distribution

    Distributions take the form:

    {
        dist: scipy.stats
        properties: statistical parameters required (in dict form)
    }

    Parameters
    ----------
    Following parameters for the NB(n,p) distribution:
        n: float
        p: float
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
            n: float,
            p: float,
            severity_distribution: dict,
            discretization_step: float = 0.01,
            grid: float = 1048576
            ):
        '''
        Initialize the frequency and severity distributions.
        '''
        frequency_distribution = {
                'dist': nbinom,
                'properties': [n, p]
                }

        super().__init__(frequency_distribution, severity_distribution, discretization_step, grid)

        self.losses = np.linspace(self.h, self.M*self.h, self.M)

    def thin_frequency(self, k: float):
        '''
        Thinning the NB distribution yields another NB distribution with modified parameter:

        NB(n, k*q/(1-q+k*q))

        where q is the probability of failure given n successes
        '''
        q = 1-self.frequency['properties'][1]  # scipy.stats definition uses p as "probability of success" but we need probability of failure
        self.frequency['properties'][1] = 1-k*q/(1-q+k*q)

    def compile_aggregate_distribution(self):
        '''
        Use the Fast Fourier Transform (FFT) to approximate the aggregate distribution.

        For a NB based frequency, the transformed aggregate distribution has the closed form:

        (1/(1+(cov-1)*(1-severity_pdf_hat)))^(lam/(cov-1))

        where severity_pdf_hat is the transformed discretized severity pdf
        '''
        # Lazy discretize the severity PDF
        if self.severity_dpdf is None:
            self.discretize_pdf()

        lam = nbinom.mean(*self.frequency['properties'])
        cov = nbinom.var(*self.frequency['properties']) / lam

        severity_pdf_hat = np.fft.fft(self.severity_dpdf)
        agg_pdf_hat = (1/(1+(cov-1)*(1-severity_pdf_hat)))**(lam/(cov-1))
        self._pdf = np.real(np.fft.ifft(agg_pdf_hat))

        self._compile_aggregate_cdf()

        if not self._layer:
            self._validate_gross()
        else:
            self._validate_gross('Partial')
