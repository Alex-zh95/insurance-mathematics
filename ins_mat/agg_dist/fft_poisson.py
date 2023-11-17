import numpy as np

from scipy.stats import poisson
from ins_mat.agg_dist.agg_base import aggregate_distribution


class poisson_fft_agg(aggregate_distribution):
    '''
    Define and build an aggregate distribution using a Poisson frequency via Fourier transform.

    Inherits: aggregate_distribution

    Distributions both taking the form:

    {
        dist: scipy.stats
        properties: statistical parameters required (in dict form)
    }

    Parameters
    ----------
    frequency: float
        mean parameter for the Poisson distribution
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
            frequency: float,
            severity_distribution: dict,
            discretization_step: float = 0.01,
            grid: float = 1048576
            ):
        '''
        Initialize the frequency and severity distributions.
        '''
        frequency_distribution = {
            'dist': poisson,
            'properties': [frequency]
        }

        super().__init__(frequency_distribution, severity_distribution, discretization_step, grid)

        self.losses = np.linspace(self.h, self.M*self.h, self.M)

    def thin_frequency(self, n: float):
        '''
        Thinning the Poisson distribution yields another Poisson distribution with modified parameter:

        n*E(frequency)
        '''
        self.frequency['properties'][0] *= n

    def compile_aggregate_distribution(self):
        '''
        Use the Fast Fourier Transform (FFT) to approximate the aggregate distribution.

        For a Poisson based frequency, the transformed aggregate distribution has the closed form:

        exp( E(frequency) * (severity_pdf_hat - 1))

        where severity_pdf_hat is the transformed discretized severity pdf
        '''
        # Lazy discretize the severity PDF
        if self.severity_dpdf is None:
            self.discretize_pdf()

        severity_pdf_hat = np.fft.fft(self.severity_dpdf)
        self.cf = np.exp(self.get_frequency_mean() * (severity_pdf_hat - 1))
        self._pdf = np.real(np.fft.ifft(self.cf))

        self._compile_aggregate_cdf()

        if not self._layer:
            self._validate_gross()
        else:
            self._validate_gross('Partial')
